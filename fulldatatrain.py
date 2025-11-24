import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.fulldata import FullDataMedDataset
from MEDCLIP.biomedclip import create_model
from MEDCLIP.tokenizer import tokenize
from MEDCLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

# Global variables that will be accessible across cells
global_vars = {}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP Testing')
    # General defaults
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16',
                        help="BiomedCLIP model version")    
    parser.add_argument('--text_encoder', type=str, default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
                        help="Text encoder used for BiomedCLIP")

    parser.add_argument('--pretrain', type=str, default='microsoft',
                        help="pretrained checkpoint source")
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help="path to dataset")

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224, 
                        help="BiomedCLIP trained with 224x224 resolution")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12],
                        help="layer features used for adapters")    
    parser.add_argument('--seed', type=int, default=111)

    # Parse arguments
    args, _ = parser.parse_known_args()

    # Print all arguments
    print("\nParsed Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        global_vars[arg] = getattr(args, arg)
    
    # Set up seed
    setup_seed(args.seed)
    print("\nSeed set to:", args.seed)

    # Fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, 
                             device=device, pretrained=args.pretrain, require_pretrained=True)
    #print(clip_model)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = True

    # Optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), 
                                     lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), 
                                     lr=args.learning_rate, betas=(0.5, 0.999))

    # Load datasets
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    
    train_dataset = FullDataMedDataset(
        dataset_path=args.data_path,
        class_name=args.obj,
        resize=args.img_size,
        mode='train',
        augment=True
    )

    valid_dataset = FullDataMedDataset(
        dataset_path=args.data_path,
        class_name=args.obj,
        resize=args.img_size,
        mode='valid',
        augment=False
    )
    
    test_dataset = FullDataMedDataset(
        dataset_path=args.data_path,
        class_name=args.obj,
        resize=args.img_size,
        mode='test',
        augment=False
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                   shuffle=True, **kwargs)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, 
                                                   shuffle=False, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                                  shuffle=False, **kwargs)

    # Losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # Text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    best_result = 0

    for epoch in range(args.epoch):
        print(f'\nepoch {epoch}:')

        loss_list = []
        
        # Training
        for (image, label, mask) in tqdm(train_dataloader, desc="Training"):
            image = image.to(device)
            label = label.to(device)
            mask = mask.to(device)
            
            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                
                # Extract patch tokens (remove CLS token)
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]
                
                # Detection loss
                det_loss = 0
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[layer] / (det_patch_tokens[layer].norm(dim=-1, keepdim=True) + 1e-8)
                    
                    vision_proj = model.visual_proj  # 768 -> 512
                    proj_tokens = det_patch_tokens[layer] @ vision_proj.weight.T
                    anomaly_map = (100.0 * proj_tokens @ text_features).unsqueeze(1)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    anomaly_score = anomaly_score.squeeze()
                    label = label.squeeze().float()

                    det_loss += loss_bce(anomaly_score, label)
                    #det_loss += loss_bce(anomaly_score, label.float())

                if CLASS_INDEX[args.obj] > 0:
                    # Segmentation loss
                    seg_loss = 0
                    mask_processed = mask.clone()
                    mask_processed[mask_processed > 0.5] = 1
                    mask_processed[mask_processed <= 0.5] = 0
                    
                    for layer in range(len(seg_patch_tokens)):
                        seg_patch_tokens[layer] = seg_patch_tokens[layer] / (seg_patch_tokens[layer].norm(dim=-1, keepdim=True) + 1e-8)
                        
                        vision_proj = model.visual_proj
                        proj_tokens = seg_patch_tokens[layer] @ vision_proj.weight.T
                        anomaly_map = (100.0 * proj_tokens @ text_features).unsqueeze(1)
                        print("anomaly_map shape:", anomaly_map.shape)
                        exit()

                        
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        
                        seg_loss += loss_focal(anomaly_map, mask_processed)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask_processed.squeeze(1))
                    
                    loss = seg_loss + det_loss
                    
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    seg_optimizer.step()
                    det_optimizer.step()

                else:
                    loss = det_loss
                    det_optimizer.zero_grad()
                    loss.backward()
                    det_optimizer.step()

                loss_list.append(loss.item())

        print("Loss: ", np.mean(loss_list))

        # Build memory bank from valid data
        seg_features = []
        det_features = []
        for image, _, _ in valid_dataloader:
            image = image.to(device)
            with torch.no_grad():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[:, 1:, :].contiguous() for p in seg_patch_tokens]
                det_patch_tokens = [p[:, 1:, :].contiguous() for p in det_patch_tokens]
                
                # Average across batch
                seg_patch_tokens = [p.mean(dim=0) for p in seg_patch_tokens]
                det_patch_tokens = [p.mean(dim=0) for p in det_patch_tokens]
                
                seg_features.append(seg_patch_tokens)
                det_features.append(det_patch_tokens)
        
        seg_mem_features = [torch.stack([seg_features[j][i] for j in range(len(seg_features))], dim=0) 
                           for i in range(len(seg_features[0]))]
        det_mem_features = [torch.stack([det_features[j][i] for j in range(len(det_features))], dim=0) 
                           for i in range(len(det_features[0]))]

        # Test
        result = test(args, model, test_dataloader, text_features, seg_mem_features, det_mem_features)
        
        if result > best_result:
            best_result = result
            print("✓ Best result saved\n")
            if args.save_model == 1:
                os.makedirs(args.save_path, exist_ok=True)
                ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')
                torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                           'det_adapters': model.det_adapters.state_dict()}, 
                           ckp_path)


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    """Test function for evaluation"""
    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []

    model.eval()
    
    for (image, y, mask) in tqdm(test_loader, desc="Testing"):
        image = image.to(device)
        mask = mask.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

            if CLASS_INDEX[args.obj] > 0:
                # Segmentation + Detection
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    cos = cos_sim(seg_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                        size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                seg_score_map_few.append(score_map_few)

                # Zero-shot segmentation
                anomaly_maps = []
                for layer in range(len(seg_patch_tokens)):
                    seg_patch_tokens[layer] /= (seg_patch_tokens[layer].norm(dim=-1, keepdim=True) + 1e-8)
                    vision_proj = model.visual_proj
                    proj_tokens = seg_patch_tokens[layer] @ vision_proj.weight.T
                    anomaly_map = (100.0 * proj_tokens @ text_features).unsqueeze(0)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                    anomaly_maps.append(anomaly_map.cpu().numpy())
                score_map_zero = np.sum(anomaly_maps, axis=0)
                seg_score_map_zero.append(score_map_zero)

            else:
                # Detection only
                anomaly_maps_few_shot = []
                for idx, p in enumerate(det_patch_tokens):
                    cos = cos_sim(det_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                        size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                score_few_det = anomaly_map_few_shot.mean()
                det_image_scores_few.append(score_few_det)

                # Zero-shot detection
                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] /= (det_patch_tokens[layer].norm(dim=-1, keepdim=True) + 1e-8)
                    vision_proj = model.visual_proj
                    proj_tokens = det_patch_tokens[layer] @ vision_proj.weight.T
                    anomaly_map = (100.0 * proj_tokens @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())

    # Calculate metrics
    gt_list = np.array(gt_list)
    gt_mask_list = np.stack(gt_mask_list, axis=0)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    if CLASS_INDEX[args.obj] > 0:
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min() + 1e-8)
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min() + 1e-8)
    
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc, 4)}')

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')

        return seg_roc_auc + roc_auc_im

    else:
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min() + 1e-8)
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min() + 1e-8)
    
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det, 4)}')

        return img_roc_auc_det


if __name__ == '__main__':
    main()