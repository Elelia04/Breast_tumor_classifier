import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pages.model import get_resnet50
from dataset import BreastCancerDataset
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_class_weights(csv_file):
    df = pd.read_csv(csv_file)
    y = df['label']
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    return torch.tensor(weights, dtype=torch.float)

def run_training(csv_train, csv_val, num_classes=2, epochs=20, batch_size=16, model_save_path="best_model.pt", patience=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = BreastCancerDataset(csv_train, transform=transform)
    val_ds = BreastCancerDataset(csv_val, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    class_weights = get_class_weights(csv_train).to(device)
    model = get_resnet50(num_classes, fine_tune_layers=True).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_f1 = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_losses, preds, targets = [], [], []
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
        train_f1 = f1_score(targets, preds, average='weighted')
        train_acc = accuracy_score(targets, preds)
        print(f"Epoch {epoch+1}: Train Loss={np.mean(train_losses):.4f}, F1={train_f1:.4f}, Acc={train_acc:.4f}")

        # Validation
        model.eval()
        val_losses, vpreds, vtargets = [], [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_losses.append(loss.item())
                vpreds.extend(torch.argmax(out, 1).cpu().numpy())
                vtargets.extend(labels.cpu().numpy())
        val_f1 = f1_score(vtargets, vpreds, average='weighted')
        val_acc = accuracy_score(vtargets, vpreds)
        print(f"Epoch {epoch+1}: Val Loss={np.mean(val_losses):.4f}, F1={val_f1:.4f}, Acc={val_acc:.4f}")

        if val_f1 > best_f1:
            torch.save(model.state_dict(), model_save_path)
            best_f1 = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best validation F1: {best_f1:.4f}")
                break

    print("Training finished. Best validation F1:", best_f1)

if __name__ == "__main__":
    # Example for 40X
    run_training('40X_train_mini.csv', '40X_val_mini.csv', num_classes=2, model_save_path="best_model_40X.pt")
