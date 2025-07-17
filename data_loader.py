import os, random, h5py, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import ray

SEGMENT_LENGTH = 1250
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

@ray.remote
def scan_record(fp, label, segments_per_record):
    try:
        with h5py.File(fp, 'r') as f:
            if 'segments' not in f:
                return None
            valid_keys = [
                k for k in f['segments'].keys()
                if any(s == b'S' for s in f['segments'][k]['annotation']['Symbol'][...])
            ]
            if len(valid_keys) < 3:
                return None

            total = segments_per_record
            v = len(valid_keys)
            if v >= total:
                selected = random.sample(valid_keys, total)
                flags = [False] * total
            else:
                miss = total - v
                num_aug = miss // 2
                num_rand = miss - num_aug
                aug_keys = random.choices(valid_keys, k=num_aug)

                all_keys = list(f['segments'].keys())
                invalid_keys = [k for k in all_keys if k not in valid_keys]
                rand_keys = (
                    random.sample(invalid_keys, num_rand)
                    if len(invalid_keys) >= num_rand
                    else invalid_keys + random.sample(valid_keys, num_rand - len(invalid_keys))
                )
                selected = valid_keys + rand_keys + aug_keys
                flags = ([False] * (v + len(rand_keys))) + ([True] * num_aug)
            return (fp, label, selected, flags)
    except:
        return None

class MILBagDataset(Dataset):
    def __init__(self, records, segments_per_record):
        tasks = [scan_record.remote(fp, label, segments_per_record) for fp, label in records]
        results = []
        pending = tasks.copy()
        with tqdm(total=len(tasks), desc="Scanning records") as pbar:
            while pending:
                ready, pending = ray.wait(pending, num_returns=1)
                res = ray.get(ready[0])
                if res is not None:
                    results.append(res)
                pbar.update(1)
        self.records = sorted(results, key=lambda x: x[0])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fp, label, keys, flags = self.records[idx]
        bag = []
        with h5py.File(fp, 'r') as f:
            for k, aug in zip(keys, flags):
                x = f['segments'][k]['signal'][:]
                x = torch.tensor(x, dtype=torch.float32).permute(1, 0)
                x = x[:, :SEGMENT_LENGTH] if x.shape[1] > SEGMENT_LENGTH else F.pad(x, (0, SEGMENT_LENGTH - x.shape[1]))
                bag.append(x)
        return torch.stack(bag), torch.tensor(label, dtype=torch.float32)

def worker_init_fn(worker_id):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
