import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np
from typing import List, Optional, Callable

# IMPORTANT: This order MUST match the order used during model training
# Standard ChestX-ray14 class order
disease_list = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# For 4-class subset, we need to map to these indices in the 14-class model
# Pneumonia=12, Effusion=4, Atelectasis=0, Cardiomegaly=1
target_4class_order = ['Pneumonia', 'Effusion', 'Atelectasis', 'Cardiomegaly']

def get_transforms(img_size=224, train=False):
    """Get image transforms for training or testing."""
    if train:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])

def load_data(data_dir, csv_file=None, seed=42):
    """Load and prepare the ChestX-ray14 dataset."""
    if csv_file is None:
        csv_file = os.path.join(data_dir, "Data_Entry_2017.csv")
    
    df = pd.read_csv(csv_file)
    
    # Get list of all image folders from images_001 to images_012
    image_folders = [os.path.join(data_dir, f"images_{str(i).zfill(3)}", "images") for i in range(1, 13)]
    
    # Create a dictionary mapping image filenames to their folder paths
    image_to_folder = {}
    for folder in image_folders:
        if os.path.exists(folder):
            for img_file in os.listdir(folder):
                if img_file.endswith('.png'):
                    image_to_folder[img_file] = folder
    
    # Filter the CSV to include only images that are present in the folders
    df = df[df['Image Index'].isin(image_to_folder.keys())]
    df = df[df['View Position'].isin(['PA', 'AP'])]
    
    # Unique patient IDs
    unique_patients = df['Patient ID'].unique()
    
    # Split patients — not rows
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=0.02, random_state=seed
    )
    
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.052, random_state=seed
    )
    
    # Use those patients to filter full image rows
    train_df = df[df['Patient ID'].isin(train_patients)]
    val_df = df[df['Patient ID'].isin(val_patients)]
    test_df = df[df['Patient ID'].isin(test_patients)]
    
    return train_df, val_df, test_df, image_to_folder

# Function to convert label string to a vector
def get_label_vector(labels_str, target_classes=None):
    """Convert label string to a binary vector.
    
    Args:
        labels_str: Pipe-separated string of labels (e.g., 'Pneumonia|Effusion')
        target_classes: List of class names to include. If None, uses all disease_list classes.
    
    Returns:
        Binary vector with 1s for present diseases, 0s otherwise
    """
    if target_classes is None:
        target_classes = disease_list
    
    labels = labels_str.split('|')

    if labels == ['No Finding']:
        return [0] * len(target_classes)
    else:
        return [1 if disease in labels else 0 for disease in target_classes]

# Unified Dataset class for ChestX-ray14
class CheXNetDataset(Dataset):
    """Flexible ChestX-ray14 dataset supporting both legacy and modern usage patterns."""
    def __init__(self, dataframe, image_to_folder, target_classes=None, data_ratio=1., transform=None):
        """
        Args:
            dataframe: DataFrame with columns 'Image Index', 'Finding Labels', optionally 'ImageIndex', 'img_path', etc.
            image_to_folder: Dict mapping image filenames to their folder paths
            target_classes: Optional list of specific disease classes to use. If None, uses all 14 classes.
            data_ratio: Fraction of dataset to use (for quick testing)
            transform: Optional torchvision transforms to apply
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_to_folder = image_to_folder
        self.target_classes = target_classes if target_classes is not None else disease_list
        self.transform = transform
        self.data_ratio = data_ratio

    def __len__(self):
        return int(len(self.dataframe) * self.data_ratio)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Handle both 'Image Index' and 'ImageIndex' column names
        img_name = row.get('Image Index') or row.get('ImageIndex')
        
        # Get image path - either from 'img_path' column or construct from image_to_folder
        if 'img_path' in row and pd.notna(row['img_path']):
            img_path = row['img_path']
        else:
            folder = self.image_to_folder[img_name]
            img_path = os.path.join(folder, img_name)
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Get labels - handle both 'Finding Labels' and 'FindingLabels' columns
        labels_str = row.get('Finding Labels') or row.get('FindingLabels')
        label_vector = get_label_vector(labels_str, self.target_classes)
        labels = torch.tensor(label_vector, dtype=torch.float)

        # Return metadata as dict for modern usage, or just path for legacy usage
        meta = {
            "ImageIndex": row.get("ImageIndex") or row.get("Image Index"),
            "ViewPosition": row.get("ViewPosition") or row.get("View Position", "UNK"),
            "n_pathologies_total": int(row.get("n_pathologies_total", 0)),
        }
        
        return image, labels, meta

def make_loaders(data_dir, csv_file=None, img_size=224, batch_size=16, num_workers=8, seed=42):
    """Create train, val, and test data loaders."""
    train_df, val_df, test_df, image_to_folder = load_data(data_dir, csv_file, seed)
    
    transform_train = get_transforms(img_size=img_size, train=True)
    transform_test = get_transforms(img_size=img_size, train=False)
    
    train_dataset = CheXNetDataset(train_df, image_to_folder, transform=transform_train)
    val_dataset = CheXNetDataset(val_df, image_to_folder, transform=transform_test)
    test_dataset = CheXNetDataset(test_df, image_to_folder, transform=transform_test)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, valloader, testloader
