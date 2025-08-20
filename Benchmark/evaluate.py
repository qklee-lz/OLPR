import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, balanced_accuracy_score

from dataset import OLPRDataset
from models import build_model

def evaluate(model, dataloader, dataset_size, device):
    criterion = nn.CrossEntropyLoss()
    running_loss, running_corrects = 0.0, 0
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / dataset_size
    val_acc = running_corrects.double().item() / dataset_size
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    mean_acc = balanced_accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0,1,2])

    print(f"[EVAL] Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {f1:.4f} | "
          f"Mean Acc: {mean_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./root_data")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = OLPRDataset(args.data_root, "test", img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model(args.model, num_classes=3, pretrained=False).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)

    evaluate(model, dataloader, len(dataset), device)
