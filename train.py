import gc, joblib, logging, os, pickle, pytz, time

import torch
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge

from utils.suzuki.model.commander.model_commander import ModelCommander
from utils.suzuki.pre_post_processing.pre_post_processing import PrePostProcessing
from utils import *

os.environ['TZ'] = 'America/New_York'
time.tzset()
MYLOGGER = logging.getLogger()


def main():
    assert torch.cuda.is_available(), "**** No available GPUs."
    opt = parse_opt()
    create_dirs_save_files(opt)
    set_seed(opt.seed)
    set_redirect_printing(opt)

    model_class = ModelCommander
    pre_post_process_class = PrePostProcessing


    params = get_params_core(model_class=model_class, 
                             pre_post_process_class=pre_post_process_class,
                             opt=opt)
    
    #! for debug ===================================================================================
    if opt.debug:
        print("\n==================================== DEBUGING mode ...")
    if opt.debug: #! 届时去掉
        print("using short data")
        if opt.task_type == "multi":
            import scipy.sparse
            train_inputs = scipy.sparse.load_npz("data/short/train_multi_inputs_values.sparse.npz")
            train_metadata = pd.read_parquet("data/short/train_multi_metadata.parquet")
            train_target = scipy.sparse.load_npz("data/short/train_multi_targets_values.sparse.npz")
            test_inputs = scipy.sparse.load_npz("data/short/test_multi_inputs_values.sparse.npz")
            test_metadata = pd.read_parquet("data/short/test_multi_metadata.parquet")
        elif opt.task_type == "cite":
            import scipy.sparse
            train_inputs = scipy.sparse.load_npz("data/short/train_cite_inputs_values.sparse.npz")
            train_metadata = pd.read_parquet("data/short/train_cite_metadata.parquet")
            train_target = scipy.sparse.load_npz("data/short/train_cite_targets_values.sparse.npz")
            test_inputs = scipy.sparse.load_npz("data/short/test_cite_inputs_values.sparse.npz")
            test_metadata = pd.read_parquet("data/short/test_cite_metadata.parquet")
    #! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    else:
        train_inputs, train_metadata, train_target = load_data(data_dir=opt.data_dir, 
                                                           task_type=opt.task_type, 
                                                           split="train", 
                                                           cell_type=CELL_TYPE_NAMES[opt.cell_type])
        test_inputs, test_metadata, _ = load_data(data_dir=opt.data_dir, 
                                                task_type=opt.task_type, 
                                                split="test")
    
    pre_post_process_default = pre_post_process_class(params["pre_post_process"])
    
    #! DEBUG =======================================================================================
    if opt.fast_process_exist_1 and not opt.debug:
        pre_post_process_default = load_pre_post_process_instance(opt)
    #! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    else:
        #! This takes long time
        print(f"start fit_preprocess {opt.task_type} ...")
        pre_post_process_default.fit_preprocess(
            inputs_values=train_inputs,
            targets_values=train_target,
            metadata=train_metadata,
            test_inputs_values=test_inputs,
            test_metadata=test_metadata,
        )
    
        if not opt.debug:
            save_pre_post_process_class_instance(pre_post_process_default, opt)

    print("\ntrain inputs shape  : ", train_inputs.shape)
    print("train metadata shape: ", train_metadata.shape)
    print("train targets shape : ", train_target.shape)
    print("test inputs shape   : ", test_inputs.shape)
    print("test metadata shape : ", test_metadata.shape, "\n")

    # cv = CrossValidation()
    # result_df, k_fold_models, k_fold_pre_post_processes = cv.compute_score(
    #     x=train_inputs,
    #     y=train_target,
    #     metadata=train_metadata,
    #     x_test=test_inputs,
    #     metadata_test=test_metadata,
    #     build_model=build_model,
    #     build_pre_post_process=build_pre_post_process,
    #     params=params,
    #     n_splits=3,
    #     dump=False,
    #     dump_dir=opt.save_dir,
    #     n_bagging=0,
    #     model_class=model_class,
    #     pre_post_process_class=pre_post_process_class,
    # )
    # print("Average:", result_df.mean(), flush=True)
    # del cv
    # gc.collect()

    print("train model to predict with test data", flush=True)
    start_time = time.time()
    # pre_post_process = build_pre_post_process(pre_post_process_class=pre_post_process_class, 
    #                                           params=params["pre_post_process"])
    pre_post_process = pre_post_process_default
    if not pre_post_process.is_fitting:
        pre_post_process.fit_preprocess(inputs_values=train_inputs, 
                                        targets_values=train_target, 
                                        metadata=train_metadata)
    else:
        print("skip pre_post_process fit")

    if opt.fast_process_exist_2 and not opt.debug:
        preprocessed_inputs_values, preprocessed_targets_values, preprocessed_test_inputs = \
                                                                load_processed_inputs_targets(opt)
    else:
        preprocessed_inputs_values, preprocessed_targets_values = pre_post_process.preprocess(
            inputs_values=train_inputs, targets_values=train_target, metadata=train_metadata
        )
        preprocessed_test_inputs, _ = pre_post_process.preprocess(
            inputs_values=test_inputs, targets_values=None, metadata=test_metadata
        )
        if not opt.debug:
            save_processed_inputs_targets(preprocessed_inputs_values, preprocessed_targets_values, 
                                          preprocessed_test_inputs, opt)


    print("preprocessed_inputs_values:", preprocessed_inputs_values.shape)
    # print(preprocessed_inputs_values)
    print("preprocessed_targets_values:", preprocessed_targets_values.shape)
    # print(preprocessed_targets_values)
    print("preprocessed_test_inputs:", preprocessed_test_inputs.shape)
    # print(preprocessed_test_inputs)

    model = build_model(model_class=model_class, params=params["model"])

    model.fit(
        x=train_inputs,
        y=train_target,
        preprocessed_x=preprocessed_inputs_values,
        preprocessed_y=preprocessed_targets_values,
        metadata=train_metadata,
        pre_post_process=pre_post_process,
    )

    print(f"elapsed time = {time.time() - start_time: .3f}")
    print("pridict with test data", flush=True)
    start_time = time.time()
    preprocessed_y_test_pred = model.predict(
        x=test_inputs, preprocessed_x=preprocessed_test_inputs, metadata=test_metadata
    )
    y_test_pred = pre_post_process.postprocess(preprocessed_y_test_pred)
    print(f"elapsed time = {time.time() - start_time: .3f}")
    print("dump preprocess and model")
    model_dir = opt.weights_dir
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "pre_post_process.pickle"), "wb") as f:
        pickle.dump(pre_post_process, f)
    model.save(model_dir)

    print("save results")
    if opt.task_type == "multi":
        pred_file_path = "multimodal_pred.pickle"
    elif opt.task_type == "cite":
        pred_file_path = "citeseq_pred.pickle"
    else:
        raise ValueError
    with open(os.path.join(opt.save_dir, pred_file_path), "wb") as f:
        pickle.dump(y_test_pred, f)
    print("completed !")


if __name__ == "__main__":
    main()
