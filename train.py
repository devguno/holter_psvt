import os, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from glob import glob
from datetime import datetime
import ray

from data_loader import MILBagDataset, worker_init_fn
from models import CNNEncoder, TimeAttentionMIL
from utils import evaluate_mil

# ───── 설정 ─────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
ray.init(ignore_reinit_error=True)

# 하이퍼파라미터
SEGMENTS_PER_RECORD = 120
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
loader_gen = torch.Generator().manual_seed(SEED)

def main():
    # 데이터 로드
    base = "/home/coder/workspace/data_readwrite/10s_segment_final"
    neg_all = sorted(glob(os.path.join(base, "not_psvt", "*.h5")))
    pos = sorted(glob(os.path.join(base, "psvt", "*.h5")))
    neg = neg_all[:len(pos)]  # 1:1 비율로 neg 샘플링

    data = [(p, 1) for p in pos] + [(n, 0) for n in neg]
    random.shuffle(data)
    tr, va = int(0.7 * len(data)), int(0.85 * len(data))
    train_records, val_records = data[:tr], data[tr:va]

    train_ds = MILBagDataset(train_records, segments_per_record=SEGMENTS_PER_RECORD)
    val_ds = MILBagDataset(val_records, segments_per_record=SEGMENTS_PER_RECORD)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, worker_init_fn=worker_init_fn, generator=loader_gen)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, worker_init_fn=worker_init_fn, generator=loader_gen)

    # 모델 초기화
    encoder = CNNEncoder()
    model = TimeAttentionMIL(encoder).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 결과 저장 디렉토리
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f""
    os.makedirs(result_dir, exist_ok=True)

    best_val_auc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for bags, labels in train_loader:
            bags, labels = bags.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(bags)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f}")

        # 검증
        print(f"Evaluating on validation set...")
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for bags, labels in val_loader:
                bags, labels = bags.to(DEVICE), labels.to(DEVICE)
                logits, _ = model(bags)
                preds += torch.sigmoid(logits).cpu().tolist()
                gts += labels.cpu().tolist()
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(gts, preds)

        print(f"Validation AUROC: {val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(result_dir, "best.pt"))
            print("Best model updated!")

    print("Training complete. Best model saved to", result_dir)

    # 평가
    print("Evaluating best model on validation set...")
    model.load_state_dict(torch.load(os.path.join(result_dir, "best.pt"), map_location=DEVICE))
    evaluate_mil(model, val_loader, result_dir, DEVICE)

if __name__ == "__main__":
    main()
