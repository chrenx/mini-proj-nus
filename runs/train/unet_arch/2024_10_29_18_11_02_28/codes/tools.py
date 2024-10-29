import importlib, logging, os, random, shutil, sys, yaml
from datetime import datetime

import torch
import numpy as np


class DotDict(dict):
    """Dictionary with dot notation access to nested keys."""
    def __init__(self, data=None):
        super().__init__()
        data = data or {}
        for key, value in data.items():
            # If the value is a dictionary, convert it to a DotDict recursively
            self[key] = DotDict(value) if isinstance(value, dict) else value

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        self[attr] = DotDict(value) if isinstance(value, dict) else value

    def __delattr__(self, attr):
        del self[attr]


class ModelLoader:
    def __init__(self, exp_name, opt=None):
        self.model = None
        self.load_model(exp_name, opt)
        
    def load_model(self, exp_name, opt):
        tmp = exp_name.split('_') # latent_ode_v1
        model_name = '_'.join(tmp[:-1]) # except the version
        match model_name:
            case 'latent_ode':
                class_name = 'LatentODE'
            case _:
                raise ValueError(f"Unknown model type in experiment name: {exp_name}")

        # Dynamically import the module and class
        module = importlib.import_module(f'model.{exp_name}')
        model_class = getattr(module, class_name)
        
        # Instantiate the model class
        match model_name:
            case 'latent_ode': 
                self.model = model_class(
                                input_dim = len(opt.feats) * 3,
                                feat_dim  = opt.fog_model_feat_dim,
                                nheads    = opt.fog_model_nheads,
                                nlayers   = opt.fog_model_nlayers,
                                dropout   = opt.fog_model_encoder_dropout,
                                clip_dim  = opt.clip_dim,
                                feats     = opt.feats,
                                txt_cond  = opt.txt_cond,
                                clip_version = opt.clip_version,
                                activation = opt.activation
                            )

        
        assert self.model is not None, "Error when loading model"



####################################################################################################
def create_dirs_save_files(opt):
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(opt.weights_dir, exist_ok=True)
    os.makedirs(opt.codes_dir, exist_ok=True)
    with open(os.path.join(opt.codes_dir, "config.yaml"), "w") as file:
        yaml.dump(opt, file)
    # Save some important code
    source_files = [f'train.sh',
                    f'train.py', 
                    f'utils/tools.py']
    for file_dir in source_files:
        shutil.copy2(file_dir, opt.codes_dir)

def cycle_dataloader(dl):
    while True:
        for data in dl:
            yield data

def get_cur_time():
    # output: e.g. 2024_11_01_13_14_01
    cur_time = datetime.now()
    cur_time = '{:%Y_%m_%d_%H_%M_%S}_{:02.0f}'.\
                format(cur_time, cur_time.microsecond / 10000.0)
    return cur_time


def set_redirect_printing(opt):
    training_info_log_path = os.path.join(opt.save_dir, "training_info.log")
    
    if not opt.disable_wandb:
        sys.stdout = open(training_info_log_path, "w")
        
        logging.basicConfig(filename=os.path.join(training_info_log_path),
                    filemode='a',
                    format='%(asctime)s.%(msecs)02d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d-%H:%M:%S',
                    level=os.environ.get("LOGLEVEL", "INFO"))
    else:
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def set_seed(seed=None):
    assert seed is not None, "no seed is given"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

