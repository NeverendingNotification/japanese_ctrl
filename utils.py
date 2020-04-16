
import os
import random

import yaml

def load_yaml(param_file):    
    with open(param_file, "r") as hndl:
        params = yaml.load(hndl, Loader=yaml.SafeLoader)
    return params

def save_yaml(params, out_file=None):
    if out_file is None:
        out_file = os.path.join(params["main_params"]["log_dir"], "params.yml")
    with open(out_file, "w") as hndl:
        hndl.write(yaml.dump(params))


def restart_params(log_dir):
    params = load_yaml(os.path.join(log_dir, "init_params.yml"))
    params["restart"] = True
    return params

def set_seed(seed):
    if isinstance(seed, int):
        seed = [seed] * 5
    else:
        assert isinstance(seed, list)

    print("Sed seed ", seed)
    random.seed(seed[0])
    os.environ['PYTHONHASHSEED'] = str(seed[1])
    import numpy as np
    np.random.seed(seed[2])

    import torch 
    torch.manual_seed(seed[3])
    torch.cuda.manual_seed(seed[4])
    torch.backends.cudnn.deterministic = True    
    torch.backends.cudnn.benchmark = False