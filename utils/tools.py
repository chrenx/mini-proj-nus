import importlib
from datetime import datetime


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


