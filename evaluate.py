import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import BreastCancerDataset
from pages.model import get_resnet50
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(csv_test, model_path="best_model.pt", num_classes=2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = BreastCancerDataset(csv_test, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=16, num_workers=2)
    model = get_resnet50(num_classes, fine_tune_layers=True).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    cm = confusion_matrix(targets, preds)
    print("Test set results:")
    print("Accuracy:", acc)
    print("F1 score:", f1)
    print("Confusion matrix:\n", cm)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [f"Class {i}" for i in range(num_classes)])
    plt.yticks(tick_marks, [f"Class {i}" for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    # Example for 40X
    evaluate_model('40X_test_mini.csv', model_path="best_model_40X.pt", num_classes=2)

