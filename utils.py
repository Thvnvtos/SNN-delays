import numpy as np
import random, sys
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_versions():
    python_version = sys.version .split(' ')[0]
    print("============== Checking Packages versions ================")
    print(f"python {python_version}")
    print(f"numpy {np.__version__}")
    print(f"pytorch {torch.__version__}")



def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # This flag only allows cudnn algorithms that are determinestic unlike .benchmark
    torch.backends.cudnn.deterministic = True

    #this flag enables cudnn for some operations such as conv layers and RNNs, 
    # which can yield a significant speedup.
    torch.backends.cudnn.enabled = False

    # This flag enables the cudnn auto-tuner that finds the best algorithm to use
    # for a particular configuration. (this mode is good whenever input sizes do not vary)
    torch.backends.cudnn.benchmark = False

    # I don't know if this is useful, look it up.
    #os.environ['PYTHONHASHSEED'] = str(seed)