import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np

CLASS_NAMES = ['Brain', 'Liver', 'Retina_RESC', 'Retina_OCT2017', 'Chest', 'Histopathology']
CLASS_INDEX = {'Brain': 3, 'Liver': 2, 'Retina_RESC': 1, 'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3}

# Dataset structure mappings
DATASET_STRUCTURE = {
    'Brain': {
        'train_good': 'BraTS2021_slice/train/good',
        'valid_good_img': 'BraTS2021_slice/valid/good/img',
        'valid_good_label': 'BraTS2021_slice/valid/good/label',
        'valid_ungood_img': 'BraTS2021_slice/valid/Ungood/img',
        'valid_ungood_label': 'BraTS2021_slice/valid/Ungood/label',
        'test_good_img': 'BraTS2021_slice/test/good/img',
        'test_ungood_img': 'BraTS2021_slice/test/Ungood/img',
        'test_ungood_label': 'BraTS2021_slice/test/Ungood/label',
        'has_mask': True
    },
    'Chest': {
        'train_good': 'Chest-RSNA/train/good',
        'valid_good_img': 'Chest-RSNA/val/good/img',
        'valid_ungood_img': 'Chest-RSNA/val/Ungood/img',
        'test_good_img': 'Chest-RSNA/test/good/img',
        'test_ungood_img': 'Chest-RSNA/test/Ungood/img',
        'has_mask': False
    },
    'Liver': {
        'train_good': 'hist_DIY/train/good',
        'valid_good_img': 'hist_DIY/valid/good/img',
        'valid_good_label': 'hist_DIY/valid/good/label',
        'valid_ungood_img': 'hist_DIY/valid/ungood/img',
        'valid_ungood_label': 'hist_DIY/valid/ungood/label',
        'test_good_img': 'hist_DIY/test/good/img',
        'test_good_label': 'hist_DIY/test/good/label',
        'test_ungood_img': 'hist_DIY/test/ungood/img',
        'test_ungood_label': 'hist_DIY/test/ungood/label',
        'has_mask': True
    },
    'Retina_RESC': {
        'train_good': 'RESC/train/good',
        'valid_good_img': 'RESC/val/good/img',
        'valid_ungood_img': 'RESC/val/ungood/img',
        'valid_ungood_label': 'RESC/val/ungood/label',
        'test_good_img': 'RESC/test/good/img',
        'test_ungood_img': 'RESC/test/ungood/img',
        'test_ungood_label': 'RESC/test/ungood/label',
        'has_mask': True
    },
    'Retina_OCT2017': {
        'train_good': 'OCT2017/train/good',
        'valid_good_img': 'OCT2017/val/good/img',
        'valid_ungood_img': 'OCT2017/val/Ungood/img',
        'test_good_img': 'OCT2017/test/good/img',
        'test_ungood_img': 'OCT2017/test/Ungood/img',
        'has_mask': False
    },
    'Histopathology': {
        'train_good': 'camelyon16_256/train/good',
        'valid_good_img': 'camelyon16_256/valid/good/img',
        'valid_ungood_img': 'camelyon16_256/valid/Ungood/img',
        'test_good_img': 'camelyon16_256/test/good/img',
        'test_ungood_img': 'camelyon16_256/test/Ungood/img',
        'has_mask': False
    }
}


def is_image_file(filename):
    """Check if file is a valid image"""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    return filename.lower().endswith(valid_extensions) and not filename.startswith('.')


class FullDataMedDataset(Dataset):
    """
    Full dataset loader for training on complete data (train + valid)
    instead of just few-shot samples
    """
    
    def __init__(self,
                 dataset_path='/data/',
                 class_name='Brain',
                 resize=224,
                 mode='train',  # 'train', 'valid', or 'test'
                 augment=True):
        """
        Args:
            dataset_path: Root path to dataset
            class_name: Class name from CLASS_NAMES
            resize: Image resize dimension
            mode: 'train', 'valid', or 'test' split
            augment: Whether to apply augmentation (for train/valid only)
        """
        assert class_name in CLASS_NAMES, f'class_name: {class_name}, should be in {CLASS_NAMES}'
        assert mode in ['train', 'valid', 'test'], f'mode should be train, valid, or test'
        
        self.dataset_path = os.path.join(dataset_path, f'{class_name}_AD')
        self.resize = resize
        self.mode = mode
        self.class_name = class_name
        self.seg_flag = CLASS_INDEX[class_name]
        self.augment = augment and mode != 'test'  # No aug for test
        self.structure = DATASET_STRUCTURE[class_name]
        self.has_mask = self.structure.get('has_mask', False)
        
        # Load dataset
        self.images, self.labels, self.masks = self._load_dataset()
        
        print(f"\n{'='*60}")
        print(f"Loaded {class_name} - {mode.upper()} SET")
        print(f"Total images: {len(self.images)}")
        print(f"Normal: {sum(1 for l in self.labels if l == 0)}")
        print(f"Abnormal: {sum(1 for l in self.labels if l == 1)}")
        print(f"{'='*60}\n")
        
        # Transforms
        self.transform_x = self._get_image_transforms()
        self.transform_mask = self._get_mask_transforms()
    
    def _get_image_transforms(self):
        """Get image transformation pipeline"""
        if self.augment:
            return transforms.Compose([
                transforms.Resize((self.resize, self.resize), Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.resize, self.resize), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _get_mask_transforms(self):
        """Get mask transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((self.resize, self.resize), Image.NEAREST),
            transforms.ToTensor()
        ])
    
    def _load_dataset(self):
        """Load images and masks for specified mode"""
        images = []
        labels = []
        masks = []
        
        if self.mode == 'train':
            # Load training data (good images only)
            train_good_path = os.path.join(self.dataset_path, self.structure['train_good'])
            if os.path.exists(train_good_path):
                train_files = [f for f in os.listdir(train_good_path) if is_image_file(f)]
                images.extend([os.path.join(train_good_path, f) for f in train_files])
                labels.extend([0] * len(train_files))
                masks.extend([None] * len(train_files))
                print(f"Loaded {len(train_files)} training good images")
        
        elif self.mode == 'valid':
            # Load validation data (both good and ungood)
            # Load valid good
            valid_good_path = os.path.join(self.dataset_path, self.structure['valid_good_img'])
            if os.path.exists(valid_good_path):
                valid_good_files = [f for f in os.listdir(valid_good_path) if is_image_file(f)]
                images.extend([os.path.join(valid_good_path, f) for f in valid_good_files])
                labels.extend([0] * len(valid_good_files))
                masks.extend([None] * len(valid_good_files))
                print(f"Loaded {len(valid_good_files)} validation good images")
            
            # Load valid ungood
            valid_ungood_path = os.path.join(self.dataset_path, self.structure['valid_ungood_img'])
            if os.path.exists(valid_ungood_path):
                valid_ungood_files = [f for f in os.listdir(valid_ungood_path) if is_image_file(f)]
                images.extend([os.path.join(valid_ungood_path, f) for f in valid_ungood_files])
                labels.extend([1] * len(valid_ungood_files))
                
                # Load corresponding masks if available
                if self.seg_flag > 0 and 'valid_ungood_label' in self.structure:
                    valid_label_path = os.path.join(self.dataset_path, self.structure['valid_ungood_label'])
                    if os.path.exists(valid_label_path):
                        mask_files = [os.path.join(valid_label_path, f) for f in valid_ungood_files]
                        masks.extend(mask_files)
                    else:
                        masks.extend([None] * len(valid_ungood_files))
                else:
                    masks.extend([None] * len(valid_ungood_files))
                
                print(f"Loaded {len(valid_ungood_files)} validation ungood images")
        
        elif self.mode == 'test':
            # Load test data (both good and ungood)
            # Load test good
            test_good_path = os.path.join(self.dataset_path, self.structure['test_good_img'])
            if os.path.exists(test_good_path):
                test_good_files = [f for f in os.listdir(test_good_path) if is_image_file(f)]
                images.extend([os.path.join(test_good_path, f) for f in test_good_files])
                labels.extend([0] * len(test_good_files))
                masks.extend([None] * len(test_good_files))
                print(f"Loaded {len(test_good_files)} test good images")
            
            # Load test ungood
            test_ungood_path = os.path.join(self.dataset_path, self.structure['test_ungood_img'])
            if os.path.exists(test_ungood_path):
                test_ungood_files = [f for f in os.listdir(test_ungood_path) if is_image_file(f)]
                images.extend([os.path.join(test_ungood_path, f) for f in test_ungood_files])
                labels.extend([1] * len(test_ungood_files))
                
                # Load corresponding masks if available
                if self.seg_flag > 0 and 'test_ungood_label' in self.structure:
                    test_label_path = os.path.join(self.dataset_path, self.structure['test_ungood_label'])
                    if os.path.exists(test_label_path):
                        mask_files = [os.path.join(test_label_path, f) for f in test_ungood_files]
                        masks.extend(mask_files)
                    else:
                        masks.extend([None] * len(test_ungood_files))
                else:
                    masks.extend([None] * len(test_ungood_files))
                
                print(f"Loaded {len(test_ungood_files)} test ungood images")
        
        return images, labels, masks
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        mask_path = self.masks[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = self.transform_x(image)
        
        # Load mask
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask)
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        else:
            mask = torch.zeros(1, self.resize, self.resize)
        
        return image, label, mask
    
    def __len__(self):
        return len(self.images)


class FewShotMedDataset(Dataset):
    """
    Original few-shot dataset (kept for compatibility)
    """
    
    def __init__(self,
                 dataset_path='/data/',
                 class_name='Brain',
                 resize=224,
                 shot=4,
                 iterate=-1):
        assert class_name in CLASS_NAMES
        assert shot > 0
        
        self.dataset_path = os.path.join(dataset_path, f'{class_name}_AD')
        self.resize = resize
        self.shot = shot
        self.iterate = iterate
        self.class_name = class_name
        self.seg_flag = CLASS_INDEX[class_name]
        self.structure = DATASET_STRUCTURE[class_name]
        
        # Load test dataset
        self.x, self.y, self.mask = self._load_test_dataset()
        
        print(f"\nTotal images: {len(self.x)}")
        print(f"Total labels: {len(self.y)}")
        print(f"Total masks: {len(self.mask)}\n")
        
        self.transform_x = transforms.Compose([
            transforms.Resize((resize, resize), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize((resize, resize), Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # Few-shot samples
        self.fewshot_norm_img = self._get_few_normal()
        self.fewshot_abnorm_img, self.fewshot_abnorm_mask = self._get_few_abnormal()
    
    def _load_test_dataset(self):
        """Load test dataset"""
        x, y, mask = [], [], []
        
        # Load test good
        test_good_path = os.path.join(self.dataset_path, self.structure['test_good_img'])
        if os.path.exists(test_good_path):
            good_files = sorted([f for f in os.listdir(test_good_path) if is_image_file(f)])
            x.extend([os.path.join(test_good_path, f) for f in good_files])
            y.extend([0] * len(good_files))
            mask.extend([None] * len(good_files))
            print(f"Loaded {len(good_files)} test good images")
        
        # Load test ungood
        test_ungood_path = os.path.join(self.dataset_path, self.structure['test_ungood_img'])
        if os.path.exists(test_ungood_path):
            ungood_files = sorted([f for f in os.listdir(test_ungood_path) if is_image_file(f)])
            x.extend([os.path.join(test_ungood_path, f) for f in ungood_files])
            y.extend([1] * len(ungood_files))
            
            if self.seg_flag > 0 and 'test_ungood_label' in self.structure:
                label_path = os.path.join(self.dataset_path, self.structure['test_ungood_label'])
                mask_files = [os.path.join(label_path, f) for f in ungood_files]
                mask.extend(mask_files)
            else:
                mask.extend([None] * len(ungood_files))
            
            print(f"Loaded {len(ungood_files)} test ungood images")
        
        return x, y, mask
    
    def _get_few_normal(self):
        """Get few-shot normal samples"""
        img_dir = os.path.join(self.dataset_path, self.structure['valid_good_img'])
        
        if not os.path.exists(img_dir):
            return torch.zeros(self.shot, 3, self.resize, self.resize)
        
        files = [f for f in os.listdir(img_dir) if is_image_file(f)]
        actual_shot = min(self.shot, len(files))
        
        if self.iterate < 0:
            selected = random.sample(files, actual_shot)
        else:
            selected = files[:actual_shot]
        
        fewshot_img = []
        for f in selected:
            img = Image.open(os.path.join(img_dir, f)).convert('RGB')
            img = self.transform_x(img)
            fewshot_img.append(img.unsqueeze(0))
        
        return torch.cat(fewshot_img) if fewshot_img else torch.zeros(actual_shot, 3, self.resize, self.resize)
    
    def _get_few_abnormal(self):
        """Get few-shot abnormal samples"""
        img_dir = os.path.join(self.dataset_path, self.structure['valid_ungood_img'])
        
        if not os.path.exists(img_dir):
            return torch.zeros(self.shot, 3, self.resize, self.resize), None
        
        files = [f for f in os.listdir(img_dir) if is_image_file(f)]
        actual_shot = min(self.shot, len(files))
        
        if self.iterate < 0:
            selected = random.sample(files, actual_shot)
        else:
            selected = files[:actual_shot]
        
        fewshot_img = []
        fewshot_mask = []
        
        for f in selected:
            img = Image.open(os.path.join(img_dir, f)).convert('RGB')
            img = self.transform_x(img)
            fewshot_img.append(img.unsqueeze(0))
            
            if self.seg_flag > 0 and 'valid_ungood_label' in self.structure:
                mask_dir = os.path.join(self.dataset_path, self.structure['valid_ungood_label'])
                mask_path = os.path.join(mask_dir, f)
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    mask = self.transform_mask(mask)
                    fewshot_mask.append(mask.unsqueeze(0))
        
        fewshot_img = torch.cat(fewshot_img) if fewshot_img else torch.zeros(actual_shot, 3, self.resize, self.resize)
        fewshot_mask = torch.cat(fewshot_mask) if fewshot_mask else None
        
        return fewshot_img, fewshot_mask
    
    def __getitem__(self, idx):
        img_path = self.x[idx]
        label = self.y[idx]
        mask_path = self.mask[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform_x(image)
        
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask)
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        else:
            mask = torch.zeros(1, self.resize, self.resize)
        
        return image, label, mask
    
    def __len__(self):
        return len(self.x)

'''
# Example usage
if __name__ == "__main__":
    # For full data training
    train_dataset = FullDataMedDataset(
        dataset_path='/kaggle/input/bmaddataset',
        class_name='Brain',
        resize=224,
        mode='train',  # Use full training data
        augment=True
    )
    
    valid_dataset = FullDataMedDataset(
        dataset_path='/kaggle/input/bmaddataset',
        class_name='Brain',
        resize=224,
        mode='valid',
        augment=False
    )
    
    test_dataset = FullDataMedDataset(
        dataset_path='/kaggle/input/bmaddataset',
        class_name='Brain',
        resize=224,
        mode='test',
        augment=False
    )
    
    # For few-shot learning (original)
    fewshot_dataset = FewShotMedDataset(
        dataset_path='/kaggle/input/bmadd',
        class_name='Brain',
        resize=224,
        shot=4
    )
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    for batch in train_loader:
        images, labels, masks = batch
        print(f"Batch shape - Images: {images.shape}, Labels: {labels.shape}, Masks: {masks.shape}")
        break
        
'''