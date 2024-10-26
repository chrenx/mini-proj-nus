
python -m train.train \
    --wandb_pj_name         fog-challenge \
    --entity                chrenx \
    --save_best_model       \
    --seed                  19 \
    \
    --exp_name              unet_v4 \
    --cuda_id               3 \
    \
    --description           "daphnet" \
    --optimizer             "adam" \
    --train_num_steps       8000 \
    --batch_size            8 \
    --random_aug            \
    --max_grad_norm         1 \
    --weight_decay          1e-6 \
    --grad_accum_step       1 \
    --window                1024 \
    --learning_rate         26e-5 \
    --penalty_cost          0 \
    --lr_scheduler_warmup_steps  8 \
    --lr_scheduler          ReduceLROnPlateau \
    --lr_scheduler_factor   0.2 \
    --lr_scheduler_patience 20 \
    \
    --train_datasets        daphnet \
    \
    --fog_model_feat_dim    512 \
    --fog_model_nheads      8 \
    --fog_model_nlayers     6 \
    --fog_model_encoder_dropout  0.5 \
    --clip_dim              512 \
    --activation            'gelu' \
    \
    --preload_gpu   \
    #    --sgd_momentum          0.9 \
    #    --sgd_enable_nesterov   \
    #    --disable_wandb \
    #    --disable_scheduler \
    #    --txt_cond               \
    #    --train_datasets        daphnet kaggle_pd_data_defog kaggle_pd_data_tdcsfog turn_in_place \
    #    --penalty_cost          2.5 \
    #    --disable_scheduler
    #    --window                -1 \
    #    --disable_wandb
    #    --preload_gpu \
    #    --disable_wandb         #!!!
    #    --preload_gpu           \      
    #    --weight_decay                   1e-6 \

