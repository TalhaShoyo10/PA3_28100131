import os
import random

import numpy as np
import torch

SEED = 6304


def set_all_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
