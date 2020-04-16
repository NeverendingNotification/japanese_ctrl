
import os
import argparse

from utils import load_yaml, restart_params, set_seed, save_yaml
from data_loaders import get_loaders
from models import get_models
from runners import get_runners

def initialize(params):
    main_params = params["main_params"]
    set_seed(main_params["seed"])


def run(params, log_dir=None):
    if log_dir is None:
        log_dir = params["main_params"]["log_dir"]
    else:
        params["main_params"]["log_dir"] = log_dir
    os.makedirs(log_dir, exist_ok=True)

    save_yaml(params, out_file=os.path.join(log_dir, "init_params.yml"))
    initialize(params)
    print("loading data")
    datasets = get_loaders(params)
    models = get_models(params, datasets)
    runners = get_runners(params, datasets, models)

    for runner in runners:
        runner()

def main(args, params=None):
    if params is None:
        if args.log_dir is not None:
            params = restart_params(args.log_dir)
        else:
            params = load_yaml(args.param_file)
    run(params)    


def get_args():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--param_file', 
                        default="setting.yml",
                        help='parameter_filename')
    parser.add_argument('--log_dir', 
                        default=None,
                        help='log directory name')

    args = parser.parse_args()
    return args
 
if __name__ == "__main__":
    args = get_args()
    main(args)
