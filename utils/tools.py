import gc, importlib, logging, os, random, shutil, sys, time, yaml
from datetime import datetime

import scipy.sparse, torch
import numpy as np
import pandas as pd


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


##################################### General ######################################################
def create_dirs_save_files(opt):
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(opt.weights_dir, exist_ok=True)
    os.makedirs(opt.codes_dir, exist_ok=True)
    with open(os.path.join(opt.codes_dir, "config.yaml"), "w") as file:
        yaml.dump(opt, file, default_flow_style=False, sort_keys=False)
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


##################################### Task specific ################################################
def get_group_id(metadata):
    # return: [0, 0, 2, 2, 1, ...] w/ len metadata.shape[0]
    days = [2, 3, 4, 7, 10]
    donors = [27678, 32606, 13176, 31800]
    technologies = ["citeseq", "multiome"]

    group_id = 0
    ret = np.full(len(metadata), fill_value=-1)
    for technology in technologies:
        for donor in donors:
            for day in days:
                selector = (metadata["technology"] == technology) & \
                           (metadata["donor"] == donor) & (metadata["day"] == day)
                ret[selector.values] = group_id
                group_id += 1
    assert (ret != -1).all()
    return ret # len: metadata.shape[0]

def load_dataset(data_dir, task_type, cell_type="all", split="train"):
    if (task_type == "multi") and (split == "train"):
        inputs_path = "train_multi_inputs_values.sparse.npz"
        inputs_idxcol_path = "train_multi_inputs_idxcol.npz"
        targets_path = "train_multi_targets_values.sparse.npz"
        cell_statistics_path = "normalized_multi_cell_statistics.parquet"
        batch_statistics_path = "normalized_multi_batch_statistics.parquet"
        batch_inputs_path = None
    elif (task_type == "multi") and (split == "test"):
        inputs_path = "test_multi_inputs_values.sparse.npz"
        inputs_idxcol_path = "test_multi_inputs_idxcol.npz"
        targets_path = None
        cell_statistics_path = "normalized_multi_cell_statistics.parquet"
        batch_statistics_path = "normalized_multi_batch_statistics.parquet"
        batch_inputs_path = None
    elif (task_type == "cite") and (split == "train"):
        inputs_path = "train_cite_inputs_values.sparse.npz"
        inputs_idxcol_path = "train_cite_inputs_idxcol.npz"
        targets_path = "train_cite_targets_values.sparse.npz"
        cell_statistics_path = "normalized_cite_cell_statistics.parquet"
        batch_statistics_path = "normalized_cite_batch_statistics.parquet"
        batch_inputs_path = "normalized_cite_batch_inputs.parquet"
    elif (task_type == "cite") and (split == "test"):
        inputs_path = "test_cite_inputs_values.sparse.npz"
        inputs_idxcol_path = "test_cite_inputs_idxcol.npz"
        targets_path = None
        cell_statistics_path = "normalized_cite_cell_statistics.parquet"
        batch_statistics_path = "normalized_cite_batch_statistics.parquet"
        batch_inputs_path = "normalized_cite_batch_inputs.parquet"
    else:
        assert task_type == "multi"
        assert split == "train"
        raise ValueError(f"invalid task type or split. {task_type}, {split}")
    input_index = np.load(os.path.join(data_dir, inputs_idxcol_path), allow_pickle=True)["index"]
    metadata_df = pd.read_parquet(os.path.join(data_dir, "metadata.parquet"))
    metadata_df = metadata_df.set_index("cell_id")
    metadata_df = metadata_df.loc[input_index, :]
    cell_statistics_df = pd.read_parquet(os.path.join(data_dir, cell_statistics_path))
    metadata_df = pd.merge(metadata_df, cell_statistics_df, left_index=True, right_index=True)
    group_ids = get_group_id(metadata_df)
    metadata_df["group"] = group_ids
    # print("before", metadata_df["group"].unique())
    if batch_statistics_path is not None:
        batch_statistics_df = pd.read_parquet(os.path.join(data_dir, batch_statistics_path))
        metadata_df = pd.merge(metadata_df, batch_statistics_df, left_on="group", right_index=True)
    # print("after", metadata_df["group"].unique())
    if batch_inputs_path is not None:
        batch_inputs_df = pd.read_parquet(os.path.join(data_dir, batch_inputs_path))
        metadata_df = pd.merge(metadata_df, batch_inputs_df, left_on="group", right_index=True)
    assert len(metadata_df) == len(input_index)
    print("load input values")
    start_time = time.time()
    train_inputs = scipy.sparse.load_npz(os.path.join(data_dir, inputs_path))
    elapsed_time = time.time() - start_time
    print(f"completed loading input values. elapsed time:{elapsed_time: .1f}")
    if targets_path is not None:
        print("load targets values")
        start_time = time.time()
        train_target = scipy.sparse.load_npz(os.path.join(data_dir, targets_path))
        elapsed_time = time.time() - start_time
        print(f"completed loading targets values. elapsed time:{elapsed_time: .1f}")
    else:
        train_target = None
    gc.collect()
    if cell_type != "all":
        s = metadata_df["cell_type"] == cell_type
        train_inputs = train_inputs[s]
        train_target = train_target[s]
        metadata_df = metadata_df[s]
    return train_inputs, metadata_df, train_target





