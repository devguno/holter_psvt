import os
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, classification_report
)

def evaluate_mil(model, test_loader, result_dir, device):
    model = model.to(device).eval()
    preds, gts = [], []
    with torch.no_grad():
        for bags, labels in test_loader:
            bags, labels = bags.to(device), labels.to(device)
            logits, _ = model(bags)
            preds += torch.sigmoid(logits).cpu().tolist()
            gts += labels.cpu().tolist()
    bin_preds = [1 if p >= 0.5 else 0 for p in preds]
    auc = roc_auc_score(gts, preds)
    auprc = average_precision_score(gts, preds)
    acc = accuracy_score(gts, bin_preds)
    prec = precision_score(gts, bin_preds, zero_division=0)
    rec = recall_score(gts, bin_preds, zero_division=0)
    f1 = f1_score(gts, bin_preds, zero_division=0)
    cls = classification_report(gts, bin_preds, digits=4)

    with open(os.path.join(result_dir, "test_metrics.txt"), "w") as f:
        f.write(f"AUROC: {auc:.4f}\nAUPRC: {auprc:.4f}\nAccuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}\n\n")
        f.write("Classification Report:\n" + cls)
