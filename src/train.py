import os
import time
import csv
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from torchvision.models import densenet121, DenseNet121_Weights


# Configuration settings
CONFIG = {
    "model": "train_chexnet",
    "batch_size": 16,
    "learning_rate": 0.001,
    "epochs": 20,
    "num_workers": 8,
    "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "/home/qinc/Dataset/Chest14",
    "wandb_project": "X-Ray Classification",
    "patience": 5,
    "seed": 42,
    "image_size": 224,
}

# List of diseases we're classifying
DISEASE_LIST = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]


def get_transforms():
    """Get training and testing transforms."""
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return transform_train, transform_test


def get_label_vector(labels_str):
    """Convert label string to a vector."""
    labels = labels_str.split('|')

    if labels == ['No Finding']:
        return [0] * len(DISEASE_LIST)
    else:
        return [1 if disease in labels else 0 for disease in DISEASE_LIST]


class CheXNetDataset(Dataset):
    """Custom Dataset class for CheXNet."""
    
    def __init__(self, dataframe, image_to_folder, data_ratio=1., transform=None):
        self.dataframe = dataframe
        self.image_to_folder = image_to_folder
        self.transform = transform
        self.data_ratio = data_ratio

    def __len__(self):
        return int(len(self.dataframe) * self.data_ratio)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Image Index']
        folder = self.image_to_folder[img_name]

        img_path = os.path.join(folder, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels_str = self.dataframe.iloc[idx]['Finding Labels']
        label_vector = get_label_vector(labels_str)
        labels = torch.tensor(label_vector, dtype=torch.float)

        return image, labels


def prepare_data(data_dir, seed=42):
    """Prepare and split the dataset."""
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


def create_dataloaders(train_df, val_df, test_df, image_to_folder, batch_size, num_workers):
    """Create DataLoaders for training, validation, and testing."""
    transform_train, transform_test = get_transforms()

    train_dataset = CheXNetDataset(train_df, image_to_folder, transform=transform_train)
    val_dataset = CheXNetDataset(val_df, image_to_folder, transform=transform_test)
    test_dataset = CheXNetDataset(test_df, image_to_folder, transform=transform_test)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader


def create_model(num_classes=14, device='cpu'):
    """Create and initialize the DenseNet121 model."""
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model = model.to(device)
    return model


def evaluate(model, dataloader, criterion, device, desc="[Test]"):
    """Evaluate the model on a given dataset."""
    model.eval()
    running_loss = 0.0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=desc, leave=True)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.sigmoid(outputs)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    test_loss = running_loss / len(dataloader)

    # Compute AUC for each class
    auc_scores = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(14)]
    avg_auc = np.mean(auc_scores)

    for i, disease in enumerate(DISEASE_LIST):
        print(f"{desc} {disease} AUC-ROC: {auc_scores[i]:.4f}")

    auc_dict = {DISEASE_LIST[i]: auc_scores[i] for i in range(14)}

    # Compute binary predictions for all classes
    preds_binary = (all_preds > 0.5).astype(int)

    # Per-class F1 scores
    f1_scores = [f1_score(all_labels[:, i], preds_binary[:, i]) for i in range(14)]
    avg_f1 = np.mean(f1_scores)

    # Print per-class F1
    for i, disease in enumerate(DISEASE_LIST):
        print(f"{desc} {disease} F1 Score: {f1_scores[i]:.4f}")

    # Build F1 dictionary
    f1_dict = {DISEASE_LIST[i]: f1_scores[i] for i in range(14)}
    print(f"{desc} Loss: {test_loss:.4f}, Avg AUC-ROC: {avg_auc:.4f}, Avg F1 Score: {avg_f1:.4f}")

    return test_loss, avg_auc, avg_f1, auc_dict, f1_dict


def train_epoch(epoch, model, trainloader, optimizer, criterion, config):
    """Train the model for one epoch."""
    device = config["device"]
    model.train()

    running_loss = 0.0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{config['epochs']} [Train]", leave=True)

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": running_loss / (i + 1)})

    train_loss = running_loss / len(trainloader)
    return train_loss


def validate(model, valloader, criterion, device):
    """Validate the model."""
    val_loss, val_auc, val_f1, auc_dict, f1_dict = evaluate(model, valloader, criterion, device, desc="[Validate]")
    return val_loss, val_auc, val_f1, auc_dict, f1_dict


def train_model(config=None):
    """Main training function."""
    if config is None:
        config = CONFIG

    # Prepare data
    print("Preparing data...")
    train_df, val_df, test_df, image_to_folder = prepare_data(config["data_dir"], config["seed"])
    
    # Create dataloaders
    print("Creating dataloaders...")
    trainloader, valloader, testloader = create_dataloaders(
        train_df, val_df, test_df, image_to_folder,
        config["batch_size"], config["num_workers"]
    )

    # Create model
    print(f"Creating model on device: {config['device']}")
    model = create_model(num_classes=14, device=config["device"])

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.1)

    # Create checkpoint directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"run-{timestamp}"
    checkpoint_dir = os.path.join("models", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize metrics CSV
    metrics_csv = os.path.join(checkpoint_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_auc", "val_f1"])

    # Training loop
    best_val_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(config["epochs"]):
        train_loss = train_epoch(epoch, model, trainloader, optimizer, criterion, config)
        val_loss, val_auc, val_f1, auc_dict, f1_dict = validate(model, valloader, criterion, config["device"])
        scheduler.step(val_loss)

        # Append to CSV
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_auc, val_f1])

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            timestamp_checkpoint = time.strftime("%Y%m%d-%H%M%S")

            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{timestamp_checkpoint}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print("Early stopping triggered.")
                break

    # Evaluate the best model on test set
    print("\nEvaluating best model on test set...")
    best_checkpoint_path = sorted(
        [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith('best_model_')]
    )[-1]
    
    model.load_state_dict(torch.load(best_checkpoint_path))
    test_loss, test_auc, test_f1, auc_dict, f1_dict = evaluate(
        model, testloader, criterion, config["device"], desc="[Test]"
    )

    # Save test metrics
    with open(os.path.join(checkpoint_dir, "test_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["test_loss", test_loss])
        writer.writerow(["test_auc", test_auc])
        writer.writerow(["test_f1", test_f1])

    # Save per-class metrics
    per_class_csv = os.path.join(checkpoint_dir, "test_per_class.csv")
    with open(per_class_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["disease", "auc", "f1"])
        for d in DISEASE_LIST:
            writer.writerow([d, auc_dict[d], f1_dict[d]])

    print(f"\nTraining complete! Results saved to {checkpoint_dir}")
    return model, checkpoint_dir


if __name__ == "__main__":
    train_model()
