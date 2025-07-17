import os, random
import torch
from torch.utils.data import DataLoader
from glob import glob
import ray

from data_loader import MILBagDataset, worker_init_fn
from models import CNNEncoder, TimeAttentionMIL
from utils import evaluate_mil

ray.init(ignore_reinit_error=True)
SEED = 42
torch.manual_seed(SEED)

# Config
SEGMENTS_PER_RECORD = 120
BATCH_SIZE = 4
DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
loader_gen = torch.Generator().manual_seed(SEED)

def main():
    ckpt_path = "/your/checkpoint/path.pth"  # Update with your checkpoint path
    result_dir = os.path.dirname(ckpt_path)
    base = "/home/coder/workspace/data_readwrite/10s_segment_final"

    neg_all = sorted(glob(os.path.join(base, "not_psvt", "*.h5")))
    pos = sorted(glob(os.path.join(base, "psvt", "*.h5")))
    neg = neg_all[:len(pos)]  # 1:1 비율로 neg 샘플링
    
    data = [(p,1) for p in pos] + [(n,0) for n in neg]
    random.shuffle(data)

    tr, va = int(0.7 * len(data)), int(0.85 * len(data))
    test_records = data[va:]

    test_ds = MILBagDataset(test_records, segments_per_record=SEGMENTS_PER_RECORD)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, worker_init_fn=worker_init_fn,
                             generator=loader_gen)

    model = TimeAttentionMIL(CNNEncoder())
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    evaluate_mil(model, test_loader, result_dir, DEVICE)

if __name__ == "__main__":
    main()
