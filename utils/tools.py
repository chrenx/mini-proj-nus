import argparse, gc, importlib, logging, os, pickle, random, shutil, sys, time, yaml
from datetime import datetime

import scipy.sparse, torch
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, ShuffleSplit

from utils.suzuki.metric.correlation_score import correlation_score
from utils.suzuki.utility.get_selector_with_metadata_pattern import get_selector_with_metadata_pattern


#!#################################### Template ####################################################
def create_dirs_save_files(opt):
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(opt.weights_dir, exist_ok=True)
    os.makedirs(opt.codes_dir, exist_ok=True)
    with open(os.path.join(opt.codes_dir, "updated_config.yaml"), "w") as file:
        yaml.dump(opt, file, default_flow_style=False, sort_keys=False)
    # Save some important code
    source_files = [f'train_day7.sh', 'train_unseen_donor.sh',
                    f'train.py']
    for file_dir in source_files:
        shutil.copy2(file_dir, opt.codes_dir)
    source_dirs = ['utils/']
    for source_dir in source_dirs:
        source_item = os.path.join(source_dir)
        destination_item = os.path.join(opt.codes_dir, source_item)
        shutil.copytree(source_item, destination_item, dirs_exist_ok=True)

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

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # project information ==========================================================================
    parser.add_argument('--exp_name', type=str, help='wandb project name')
    parser.add_argument('--description', type=str, help='important notes')
    parser.add_argument('--save_dir', type=str, 
                             help='save important files, weights, etc., will be intialized if null')
    parser.add_argument('--wandb_pj_name', type=str, help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, help='wandb account')
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--seed', type=int, help='initializing seed')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--problem_type', type=str, help='problem type')
    
    # data =========================================================================================
    parser.add_argument('--data_dir', type=str, help='data directory')

    # GPU ==========================================================================================
    parser.add_argument('--cuda_id', type=str, help='assign gpu')
    
    # training =====================================================================================
    parser.add_argument('--save_best_model', action='store_true', 
                                             help='save best model during training')
    parser.add_argument('--skip_test_prediction', action='store_true')
    parser.add_argument("--task_type", choices=["multi", "cite"])
    parser.add_argument("--cell_type", choices=["all", "hsc", "eryp", "neup", 
                                                "masp", "mkp", "bp", "mop"])
    parser.add_argument('--model', choices=["ead", "unet"])
    parser.add_argument('--train_batch_size', type=int, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, help='batch size for testing')
    parser.add_argument('--epoch', type=int, help='num of epoch')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--backbone', type=str)

    # ==============================================================================================

    args = vars(parser.parse_args())
    cli_args = [k for k in args.keys() if args[k] is not None and args[k] is not False]
    
    with open('utils/config.yaml', 'r') as f:
        opt = yaml.safe_load(f)
        opt = DotDict(opt)
        for arg in cli_args: # update arguments if passed from command line; otherwise, use default
            opt[arg] = args[arg]
    opt.exp_name = opt.exp_name + '_' + opt.task_type
    opt.data_dir += opt.problem_type

    cur_time = get_cur_time()
    opt.cur_time = cur_time
    if opt.save_dir is None:
        opt.save_dir = os.path.join('runs/train', opt.exp_name, cur_time)
    opt.weights_dir = os.path.join(opt.save_dir, 'weights')
    opt.codes_dir = os.path.join(opt.save_dir, 'codes')
    opt.device_info = torch.cuda.get_device_name(int(opt.cuda_id)) 
    opt.device = f"cuda:{opt.cuda_id}"

    if not opt.debug:
        if os.path.exists(os.path.join(opt.data_dir, f"{opt.task_type}_whole_preprocess_obj.pickle")):
            opt.fast_process_exist_1 = True
        else:
            opt.fast_process_exist_1 = False
        a=os.path.exists(os.path.join(opt.data_dir, f"{opt.task_type}_preprocessed_inputs_values.pickle"))
        b=os.path.exists(os.path.join(opt.data_dir, f"{opt.task_type}_preprocessed_targets_values.pickle"))
        # c=os.path.exists(os.path.join(opt.data_dir, f"{opt.task_type}_preprocessed_test_inputs.pickle"))
        if a and b: # and c:
            opt.fast_process_exist_2 = True
        else:
            opt.fast_process_exist_2 = False

    return opt

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

#*--------------------------------------------------------------------------------------------------

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

#!#################################### Proj specific ###############################################
def build_model(model_class, params):
    model = model_class(params)
    return model

def build_pre_post_process(pre_post_process_class, params):
    pre_post_process = pre_post_process_class(params)
    return pre_post_process

def get_params_core(model_class, pre_post_process_class, opt):
        model_params = model_class.get_params(opt)
        pre_post_process_params = pre_post_process_class.get_params(opt)
        return {"model": model_params, "pre_post_process": pre_post_process_params}

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

def load_data(data_dir, task_type, problem_type, cell_type="all", split="train"):
    if (task_type == "multi") and (split == "train"):
        inputs_path = "train_multi_inputs_values.sparse.npz"
        inputs_idxcol_path = f"train_multi_inputs_idxcol.npz"
        targets_path = "train_multi_targets_values.sparse.npz"
        cell_statistics_path = "normalized_multi_cell_statistics.parquet"
        batch_statistics_path = "normalized_multi_batch_statistics.parquet"
        batch_inputs_path = None
    # elif (task_type == "multi") and (split == "test"):
    #     inputs_path = "test_multi_inputs_values.sparse.npz"
    #     inputs_idxcol_path = "test_multi_inputs_idxcol.npz"
    #     targets_path = None
    #     cell_statistics_path = "normalized_multi_cell_statistics.parquet"
    #     batch_statistics_path = "normalized_multi_batch_statistics.parquet"
    #     batch_inputs_path = None
    # elif (task_type == "cite") and (split == "train"):
    #     inputs_path = "train_cite_inputs_values.sparse.npz"
    #     inputs_idxcol_path = "train_cite_inputs_idxcol.npz"
    #     targets_path = "train_cite_targets_values.sparse.npz"
    #     cell_statistics_path = "normalized_cite_cell_statistics.parquet"
    #     batch_statistics_path = "normalized_cite_batch_statistics.parquet"
    #     batch_inputs_path = "normalized_cite_batch_inputs.parquet"
    # elif (task_type == "cite") and (split == "test"):
    #     inputs_path = "test_cite_inputs_values.sparse.npz"
    #     inputs_idxcol_path = "test_cite_inputs_idxcol.npz"
    #     targets_path = None
    #     cell_statistics_path = "normalized_cite_cell_statistics.parquet"
    #     batch_statistics_path = "normalized_cite_batch_statistics.parquet"
    #     batch_inputs_path = "normalized_cite_batch_inputs.parquet"
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
    assert len(metadata_df) == len(input_index), "metadata_df len != input_index len"
    print(f"load {split} input values...")
    start_time = time.time()
    train_inputs = scipy.sparse.load_npz(os.path.join(data_dir, inputs_path))
    elapsed_time = time.time() - start_time
    print(f"completed loading input values. elapsed time:{elapsed_time: .1f}")
    if targets_path is not None:
        print(f"load {split} targets values...")
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

def load_processed_inputs_targets(opt):
    print("load processed_inputs_targets instance ...")

    input_path = os.path.join(opt.data_dir, f"{opt.task_type}_preprocessed_inputs_values.pickle")
    target_path = os.path.join(opt.data_dir, f"{opt.task_type}_preprocessed_targets_values.pickle")

    with open(input_path, 'rb') as f:
        preprocessed_inputs_values = pickle.load(f)
    with open(target_path, 'rb') as f:
        preprocessed_targets_values = pickle.load(f)
    # with open(f"data/fast_preprocess/{opt.task_type}_preprocessed_test_inputs.pickle", 'rb') as f:
    #     preprocessed_test_inputs = pickle.load(f)
    # return preprocessed_inputs_values, preprocessed_targets_values, preprocessed_test_inputs
    return preprocessed_inputs_values, preprocessed_targets_values

def load_pre_post_process_instance(opt):
    print("load pre_post_process_class instance ...")
    dpath = os.path.join(opt.data_dir, f"{opt.task_type}_whole_preprocess_obj.pickle")
    with open(dpath, 'rb') as f:
        class_instance = pickle.load(f)
    return class_instance

# def save_processed_inputs_targets(preprocessed_inputs_values, preprocessed_targets_values, 
#                                   preprocessed_test_inputs, opt):
def save_processed_inputs_targets(preprocessed_inputs_values, preprocessed_targets_values, opt):
    if opt.fast_process_exist_2:
        return
    
    train_input_path = os.path.join(opt.data_dir, f"{opt.task_type}_preprocessed_inputs_values.pickle")
    target_path = os.path.join(opt.data_dir, f"{opt.task_type}_preprocessed_targets_values.pickle")
    # test_input_path = os.path.join(opt.data_dir, f"{opt.task_type}_preprocessed_test_inputs.pickle")

    if os.path.exists(train_input_path) and os.path.exists(target_path):
        return

    print("save processed_inputs_targets ...")
    os.makedirs(opt.data_dir, exist_ok=True)
    with open(train_input_path, 'wb') as f:
        pickle.dump(preprocessed_inputs_values, f)
    with open(target_path, 'wb') as f:
        pickle.dump(preprocessed_targets_values, f)
    # with open(test_input_path, 'wb') as f:
    #     pickle.dump(preprocessed_test_inputs, f)

def save_pre_post_process_class_instance(pre_post_process, opt):
    if opt.fast_process_exist_1:
        return
    class_instance_path = os.path.join(opt.data_dir, f"{opt.task_type}_whole_preprocess_obj.pickle")
    if os.path.exists(class_instance_path):
        return
    print("save preprocesses ...")
    os.makedirs(opt.data_dir, exist_ok=True)
    with open(class_instance_path, "wb") as f:
        pickle.dump(pre_post_process, f)

def split_dataset_save_load_idx(dataset, opt, metadata):
    train_idx_path = os.path.join(opt.data_dir, f"train_dataloader_idx_{opt.task_type}.npy")
    val_idx_path = os.path.join(opt.data_dir, f"val_dataloader_idx_{opt.task_type}.npy")
    test_idx_path = os.path.join(opt.data_dir, f"test_dataloader_idx_{opt.task_type}.npy")

    if os.path.exists(train_idx_path) and os.path.exists(val_idx_path) and \
       os.path.exists(test_idx_path):
        train_idx = np.load(train_idx_path).tolist()
        val_idx = np.load(val_idx_path).tolist()
        test_idx = np.load(test_idx_path).tolist()
        print("train val test idx have been loaded")
        return train_idx, val_idx, test_idx
    
    base = "day7"
    if opt.problem_type == "unseen_donor":
        base = "donor32"
    test_idx = np.load(os.path.join(opt.data_dir, f"test_multi_{base}_inputs_idxcol.npz"), 
                                      allow_pickle=True)["index"]

    test_idx = metadata.index.get_indexer(test_idx)
    all_indices = np.arange(len(metadata))  # All row indices in metadata
    rest_indices = np.setdiff1d(all_indices, test_idx)  # Find indices not in row_indices

    assert len(test_idx) + len(rest_indices) == len(metadata), "unmatched length"
    
    np.random.shuffle(rest_indices)
    split_point = int(0.9 * len(rest_indices))
    train_idx = rest_indices[:split_point]
    val_idx = rest_indices[split_point:]
    # dataset_size = len(dataset)
    # train_size = int(dataset_size * 0.8)
    # val_size = int(dataset_size * 0.1)
    # indices = torch.randperm(dataset_size).tolist()
    # train_idx = indices[:train_size]
    # val_idx = indices[train_size:train_size + val_size]
    # test_idx = indices[train_size + val_size:]
    np.save(train_idx_path, train_idx)
    np.save(val_idx_path, val_idx)
    np.save(test_idx_path, test_idx)
    print("train val test idx have been splited and saved")
    return train_idx, val_idx, test_idx

#*--------------------------------------------------------------------------------------------------

class CrossValidation(object):
    def __init__(self):
        pass

    def compute_score(
        self,
        x,
        y,
        metadata,
        x_test,
        metadata_test,
        params,
        build_model=build_model,
        build_pre_post_process=build_pre_post_process,
        dump=False,
        dump_dir="./",
        n_splits=3,
        n_bagging=0,
        bagging_ratio=1.0,
        use_batch_group=True,
        model_class=None,
        pre_post_process_class=None,
    ):
        groups = None
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
        if use_batch_group:
            batch_groups = get_group_id(metadata)
            n_groups = len(np.unique(batch_groups))
            if n_groups > n_splits:
                print(f"use batch group. # groups is {n_groups}")
                groups = batch_groups
                kf = GroupKFold(n_splits=n_splits)
        pre_post_processes = []
        models = []
        cell_types = metadata["cell_type"].unique()
        scores_dict = {
            "mse_train": [],
            "corrscore_train": [],
            "mse_val": [],
            "corrscore_val": [],
        }

        for cell_type_name in cell_types:
            scores_dict[f"{cell_type_name}_mse_train"] = []
            scores_dict[f"{cell_type_name}_corrscore_train"] = []
            scores_dict[f"{cell_type_name}_mse_val"] = []
            scores_dict[f"{cell_type_name}_corrscore_val"] = []
        print("kfold type", type(kf), "group", groups)
        for fold, (idx_tr, idx_va) in enumerate(kf.split(x, y, groups=groups)):
            if groups is not None:
                print("train group:", np.unique(groups[idx_tr]))
                print("val group:", np.unique(groups[idx_va]))
            start_time = time.time()
            model = None
            score = {}
            gc.collect()
            _n_bagging = n_bagging
            if _n_bagging == 0:
                _n_bagging = 1
            y_train_pred = None
            y_val_pred = None
            metadata_train = metadata.iloc[idx_tr, :]
            if "selected_metadata" in params["model"]:
                selector = get_selector_with_metadata_pattern(
                    metadata=metadata_train, metadata_pattern=params["model"]["selected_metadata"]
                )
                if np.sum(selector) == 0:
                    print("skip!")
                    continue
            x_train = x[idx_tr]  # creates a copy, https://numpy.org/doc/stable/user/basics.copies.html
            y_train = y[idx_tr].toarray()
            # We validate the model
            x_val = x[idx_va]
            y_val = y[idx_va].toarray()
            metadata_val = metadata.iloc[idx_va, :]
            for bagging_i in range(_n_bagging):
                gc.collect()
                if n_bagging > 0:
                    raise "有问题"
                    print("bagging_i", bagging_i, flush=True)
                    n_bagging_size = int(x_train.shape[0] * bagging_ratio)
                    bagging_idx = np.random.permutation(x_train.shape[0])[:n_bagging_size]
                    x_train_bagging = x_train[bagging_idx]
                    y_train_bagging = y_train[bagging_idx]
                    metadata_train_bagging = metadata_train.iloc[bagging_idx, :]
                else:
                    x_train_bagging = x_train
                    y_train_bagging = y_train
                    metadata_train_bagging = metadata_train


                pre_post_process = build_pre_post_process(pre_post_process_class,
                                                          params=params["pre_post_process"])
                if not pre_post_process.is_fitting:
                    pre_post_process.fit_preprocess(
                        inputs_values=x_train_bagging,
                        targets_values=y_train_bagging,
                        metadata=metadata_train_bagging,
                        test_inputs_values=x_val,
                        test_metadata=metadata_val
                    )
                else:
                    print("skip pre_post_process fit")
                pre_post_processes.append(pre_post_process)
                preprocessed_x_train, preprocessed_y_train = pre_post_process.preprocess(
                    inputs_values=x_train_bagging,
                    targets_values=y_train_bagging,
                    metadata=metadata_train_bagging,
                )
                model = build_model(model_class=model_class, params=params["model"])
                # print("[]][][][][][]")
                # print(type(preprocessed_x_train))
                # print(type(preprocessed_y_train))
                print(f"model input shape X:{preprocessed_x_train.shape} Y:{preprocessed_y_train.shape}")

                model.fit(
                    x=x_train_bagging,
                    y=y_train_bagging,
                    preprocessed_x=preprocessed_x_train,
                    preprocessed_y=preprocessed_y_train,
                    metadata=metadata_train_bagging,
                    pre_post_process=pre_post_process,
                )
                preprocessed_y_train_pred = model.predict(x=x_train, preprocessed_x=preprocessed_x_train, metadata=metadata_train)
                new_y_train_pred = pre_post_process.postprocess(preprocessed_y_train_pred)
                preprocessed_x_val, _ = pre_post_process.preprocess(
                    inputs_values=x_val,
                    targets_values=None,
                    metadata=metadata_val,
                )
                preprocessed_y_val_pred = model.predict(x=x_val, preprocessed_x=preprocessed_x_val, metadata=metadata_val)
                new_y_val_pred = pre_post_process.postprocess(preprocessed_y_val_pred)
                models.append(model)

                mse_train = mean_squared_error(y_train, new_y_train_pred)
                corrscore_train = correlation_score(y_train, new_y_train_pred)
                mse_val = mean_squared_error(y_val, new_y_val_pred)
                corrscore_val = correlation_score(y_val, new_y_val_pred)
                print(
                    f"Fold {fold} bagging {bagging_i} "
                    f"mse_train: {mse_train: .5f} "
                    f"corrscore_train: {corrscore_train: .5f} "
                    f"mse_val: {mse_val: .5f} "
                    f"corrscore_val: {corrscore_val: .5f} "
                )

                if y_train_pred is None:
                    y_train_pred = new_y_train_pred
                else:
                    y_train_pred += new_y_train_pred
                if y_val_pred is None:
                    y_val_pred = new_y_val_pred
                else:
                    y_val_pred += new_y_val_pred

            y_train_pred /= _n_bagging
            y_val_pred /= _n_bagging

            mse_train = mean_squared_error(y_train, y_train_pred)
            corrscore_train = correlation_score(y_train, y_train_pred)
            score["mse_train"] = mse_train
            score["corrscore_train"] = corrscore_train
            cell_type_train = metadata_train["cell_type"].values
            for cell_type_name in cell_types:
                s = cell_type_train == cell_type_name
                if s.sum() > 10:
                    mse_train = mean_squared_error(y_train[s], y_train_pred[s])
                    corrscore_train = correlation_score(y_train[s], y_train_pred[s])
                    score[f"{cell_type_name}_mse_train"] = mse_train
                    score[f"{cell_type_name}_corrscore_train"] = corrscore_train
                else:
                    score[f"{cell_type_name}_mse_train"] = 0.0
                    score[f"{cell_type_name}_corrscore_train"] = 1.0
            gc.collect()

            mse_val = mean_squared_error(y_val, y_val_pred)
            corrscore_val = correlation_score(y_val, y_val_pred)
            score["mse_val"] = mse_val
            score["corrscore_val"] = corrscore_val
            cell_type_val = metadata_val["cell_type"].values
            for cell_type_name in cell_types:
                s = cell_type_val == cell_type_name
                if s.sum() > 10:
                    mse_val = mean_squared_error(y_val[s], y_val_pred[s])
                    corrscore_val = correlation_score(y_val[s], y_val_pred[s])
                    score[f"{cell_type_name}_mse_val"] = mse_val
                    score[f"{cell_type_name}_corrscore_val"] = corrscore_val
                else:
                    score[f"{cell_type_name}_mse_val"] = 0.0
                    score[f"{cell_type_name}_corrscore_val"] = 1.0
            if dump:
                np.save(os.path.join(dump_dir, f"k{fold}_y_val.npy"), y_val)
                np.save(os.path.join(dump_dir, f"k{fold}_y_val_pred.npy"), y_val_pred)

            del x_train, y_train, y_train_pred, x_val, y_val_pred, y_val
            print(f"Fold {fold}: score:{score} elapsed time = {time.time() - start_time: .3f}")
            for k, v in score.items():
                scores_dict[k].append(v)

        # Show overall score
        result_df = pd.DataFrame(scores_dict)
        return result_df, models, pre_post_processes
