import gc, joblib, logging, os, pickle, pytz, time

import torch
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from utils.suzuki.metric.correlation_score import correlation_score
from utils.suzuki.model.encoder_decoder.encoder_decoder import EncoderDecoder
from utils.suzuki.pre_post_processing.pre_post_processing import PrePostProcessing
from utils.suzuki.utility.get_selector_with_metadata_pattern import get_selector_with_metadata_pattern
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
    
    #! for debug ===================================================================================
    if opt.debug:
        print("\n==================================== DEBUGING mode ...")
        print("using short data:", opt.use_short_data, "\n")
        if opt.task_type == "multi":
            import scipy.sparse
            train_inputs = scipy.sparse.load_npz("data/fast/train_multi_inputs_values.sparse.npz")
            train_metadata = pd.read_parquet("data/fast/train_multi_metadata.parquet")
            train_target = scipy.sparse.load_npz("data/fast/train_multi_targets_values.sparse.npz")
            test_inputs = scipy.sparse.load_npz("data/fast/test_multi_inputs_values.sparse.npz")
            test_metadata = pd.read_parquet("data/fast/test_multi_metadata.parquet")
        elif opt.task_type == "cite":
            import scipy.sparse
            train_inputs = scipy.sparse.load_npz("data/fast/train_cite_inputs_values.sparse.npz")
            train_metadata = pd.read_parquet("data/fast/train_cite_metadata.parquet")
            train_target = scipy.sparse.load_npz("data/fast/train_cite_targets_values.sparse.npz")
            test_inputs = scipy.sparse.load_npz("data/fast/test_cite_inputs_values.sparse.npz")
            test_metadata = pd.read_parquet("data/fast/test_cite_metadata.parquet")
    #! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    else:
        train_inputs, train_metadata, train_target = load_data(data_dir=opt.data_dir, 
                                                           task_type=opt.task_type, 
                                                           split="train", 
                                                           cell_type=CELL_TYPE_NAMES[opt.cell_type])
        test_inputs, test_metadata, _ = load_data(data_dir=opt.data_dir, 
                                                task_type=opt.task_type, 
                                                split="test")

    if opt.model == "ead":
        model_class = EncoderDecoder
        pre_post_process_class = PrePostProcessing
    else:
        raise ValueError

    params = get_params_core(model_class=model_class, 
                             pre_post_process_class=pre_post_process_class,
                             opt=opt)
    
    pre_post_process_default = pre_post_process_class(params["pre_post_process"])
    
    #! DEBUG =======================================================================================
    if opt.debug and not opt.use_short_data: #! 届时去掉
        in_dcmp = joblib.load(f'data/fast_preprocess/{opt.task_type}_targets_decomposer.joblib')
        tar_dcmp = joblib.load(f'data/fast_preprocess/{opt.task_type}_targets_decomposer.joblib')
        tar_glb_med = np.load(f'data/fast_preprocess/{opt.task_type}_targets_global_median.npy')
        pre_post_process_default.preprocesses['targets_global_median'] = tar_glb_med
        pre_post_process_default.preprocesses['targets_decomposer'] = tar_dcmp
        pre_post_process_default.preprocesses['inputs_decomposer'] = in_dcmp
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
        print("save preprocesses ...")
        save_pre_post_process_default(pre_post_process_default.preprocesses, opt)

    #! DEBUG =======================================================================================
    # print("-----------keys")
    # print(pre_post_process_default.preprocesses.keys())
    # 'targets_imputator', 'targets_batch_medians', 'targets_global_median'有用, 
    # 'targets_decomposer'有用, 'binary_inputs_decomposer', 'inputs_decomposer'有用
    # print("targets_global_median", pre_post_process_default.preprocesses['targets_global_median'].shape) 
    # 23418
    # print(pre_post_process_default.preprocesses['targets_batch_medians'])
    # for key in pre_post_process_default.preprocesses.keys():
    #     print(key, type(pre_post_process_default.preprocesses[key]))
    #! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    print("\ntrain inputs shape  : ", train_inputs.shape)
    print("train metadata shape: ", train_metadata.shape)
    print("train targets shape : ", train_target.shape)
    print("test inputs shape   : ", test_inputs.shape)
    print("test metadata shape : ", test_metadata.shape, "\n")


    cv = CrossValidation()

    print("一切顺利")
    exit(0)
    result_df, k_fold_models, k_fold_pre_post_processes = cv.compute_score(
        x=train_inputs,
        y=train_target,
        metadata=train_metadata,
        x_test=test_inputs,
        metadata_test=test_metadata,
        build_model=build_model,
        build_pre_post_process=build_pre_post_process,
        params=params,
        n_splits=3,
        dump=False,
        dump_dir=opt.save_dir,
        n_bagging=0,
    )
    print("Average:", result_df.mean(), flush=True)
    del cv
    gc.collect()

    # if not args.skip_test_prediction:
    print("train model to predict with test data", flush=True)
    start_time = time.time()
    pre_post_process = build_pre_post_process(params["pre_post_process"])
    if not pre_post_process.is_fitting:
        pre_post_process.fit_preprocess(inputs_values=train_inputs, targets_values=train_target, metadata=train_metadata)
    else:
        print("skip pre_post_process fit")
    preprocessed_inputs_values, preprocessed_targets_values = pre_post_process.preprocess(
        inputs_values=train_inputs, targets_values=train_target, metadata=train_metadata
    )
    preprocessed_test_inputs, _ = pre_post_process.preprocess(
        inputs_values=test_inputs, targets_values=None, metadata=test_metadata
    )
    model = build_model(params=params["model"])
    
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
    model_dir = os.path.join(args.out_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "pre_post_process.pickle"), "wb") as f:
        pickle.dump(pre_post_process, f)
    model.save(model_dir)


    print("save results")
    if args.task_type == "multi":
        pred_file_path = "multimodal_pred.pickle"
    elif args.task_type == "cite":
        pred_file_path = "citeseq_pred.pickle"
    else:
        raise ValueError
    with open(os.path.join(args.weights, pred_file_path), "wb") as f:
        pickle.dump(y_test_pred, f)
    print("completed !")


if __name__ == "__main__":
    main()
