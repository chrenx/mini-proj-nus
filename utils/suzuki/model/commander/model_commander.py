import gc
import json
import os
import pickle
import time

import numpy as np
import torch, wandb

from utils.suzuki.model.commander.cite_encoder_decoder_module import CiteEncoderDecoderModule
from utils.suzuki.model.commander.mlp_module import HierarchicalMLPBModule, MLPBModule
from utils.suzuki.model.commander.multi_encoder_decoder_module import MultiEncoderDecoderModule
from utils.suzuki.model.torch_dataset.citeseq_dataset import CITEseqDataset
from utils.suzuki.model.torch_dataset.multiome_dataset import MultiomeDataset
from utils.suzuki.model.torch_helper.set_weight_decay import set_weight_decay
from utils.suzuki.utility.summeary_torch_model_parameters import summeary_torch_model_parameters


class ModelCommander(object):
    @staticmethod
    def get_params(opt, snapshot=None):
        params = {
            "device": opt.device,
            "snapshot": snapshot,
            "train_batch_size": opt.train_batch_size,
            "test_batch_size": opt.test_batch_size,
            "task_type": opt.task_type,
            "lr": opt.lr,
            "eps": opt.eps,
            "weight_decay": opt.weight_decay,
            "epoch": opt.epoch,
            "pct_start": opt.pct_start,
            "burnin_length_epoch": opt.burnin_length_epoch,
            "backbone": opt.backbone,
            "max_inputs_values_noisze_sigma": opt.max_inputs_values_noisze_sigma,
            "max_cutout_p": opt.max_cutout_p,
            "opt": opt,
        }
        if params["backbone"] == "mlp":
            backbone_params = {
                "encoder_h_dim": opt.encoder_h_dim,  # 128,
                "decoder_h_dim": opt.decoder_h_dim,  # 128,
                "encoder_dropout_p": opt.encoder_dropout_p,
                "decoder_dropout_p": opt.decoder_dropout_p,
                "n_encoder_block": opt.n_encoder_block,
                "n_decoder_block": opt.n_decoder_block,
                "norm": opt.norm,
                "activation": opt.activation,  # relu, "gelu"
                "skip": opt.skip,
            }
        else:
            raise RuntimeError
        params.update(backbone_params)

        task_specific_params = {}
        if opt.task_type == "multi":
            task_specific_params["lr"] = 9.97545796487608e-05
            task_specific_params["eps"] = 1.8042413185663546e-09
            task_specific_params["weight_decay"] = 1.7173609280566294e-07
            task_specific_params["encoder_dropout_p"] = 0.4195254015709299
            task_specific_params["decoder_dropout_p"] = 0.30449413021670935
        elif opt.task_type == "cite":
            task_specific_params["lr"] = 0.00012520653814999459
            task_specific_params["eps"] = 7.257005721594269e-08
            task_specific_params["weight_decay"] = 2.576638574613591e-06
            task_specific_params["encoder_dropout_p"] = 0.5952997562668841
            task_specific_params["decoder_dropout_p"] = 0.31846059114042935
        params.update(task_specific_params)

        return params

    def __init__(self, params):
        self.params = params
        self.inputs_info = {}
        self.model = None
        self.opt = params['opt']
        self.params.pop('opt', None)

    def _build_model(self):
        if self.params["snapshot"] is not None:
            print(f"load model from {self.params['snapshot']}")
            model = torch.load(os.path.join(self.params["snapshot"], "model.pt"))
            return model
        x_dim = self.inputs_info["x_dim"]
        y_dim = self.inputs_info["y_dim"]
        inputs_decomposer_components = torch.tensor(self.inputs_info["inputs_decomposer_components"])
        targets_decomposer_components = torch.tensor(self.inputs_info["targets_decomposer_components"])
        y_statistic = {}
        for k, v in self.inputs_info["y_statistic"].items():
            y_statistic[k] = torch.tensor(v)
        if self.params["backbone"] == "mlp":
            encoder = MLPBModule(
                # input_dim=x_dim,
                input_dim=None,
                output_dim=self.params["encoder_h_dim"],
                n_block=self.params["n_encoder_block"],
                h_dim=self.params["encoder_h_dim"],
                skip=self.params["skip"],
                dropout_p=self.params["encoder_dropout_p"],
                activation=self.params["activation"],
                norm=self.params["norm"],
            )

            decoder = HierarchicalMLPBModule(
                input_dim=self.params["encoder_h_dim"],
                # output_dim=y_dim,
                # output_dim=y_dim,
                output_dim=None,
                n_block=self.params["n_decoder_block"],
                h_dim=self.params["decoder_h_dim"],
                skip=self.params["skip"],
                dropout_p=self.params["decoder_dropout_p"],
                activation=self.params["activation"],
                norm=self.params["norm"],
            )
        else:
            raise RuntimeError

        if self.params["task_type"] == "multi":
            model = MultiEncoderDecoderModule(
                x_dim=x_dim,
                y_dim=y_dim,
                y_statistic=y_statistic,
                encoder_h_dim=self.params["encoder_h_dim"],
                decoder_h_dim=self.params["decoder_h_dim"],
                n_decoder_block=self.params["n_decoder_block"],
                encoder=encoder,
                decoder=decoder,
                inputs_decomposer_components=inputs_decomposer_components,
                targets_decomposer_components=targets_decomposer_components,
            )
        elif self.params["task_type"] == "cite":
            model = CiteEncoderDecoderModule(
                x_dim=x_dim,
                y_dim=y_dim,
                y_statistic=y_statistic,
                encoder_h_dim=self.params["encoder_h_dim"],
                decoder_h_dim=self.params["decoder_h_dim"],
                n_decoder_block=self.params["n_decoder_block"],
                encoder=encoder,
                decoder=decoder,
                inputs_decomposer_components=inputs_decomposer_components,
                targets_decomposer_components=targets_decomposer_components,
            )
        else:
            raise ValueError
        return model

    def _batch_to_device(self, batch):
        return tuple(batch[i].to(self.params["device"]) for i in range(len(batch)))

    def _train_step_forward(self, batch, training_length_ratio):
        loss = self.model.loss(*batch, training_length_ratio=training_length_ratio)
        return loss

    def fit(self, x, preprocessed_x, y, preprocessed_y, metadata, pre_post_process):
        # print("x             :", x.shape)               # (105942, 228942)
        # print("preprocessed_x:", preprocessed_x.shape)  # (105942, 256)
        # print("y             :", y.shape)               # (105942, 23418)
        # print("preprocessed_y:", preprocessed_y.shape)  # (105942, 128)
        # print("metadata      :", metadata.shape)        # (105942, 57)

        if self.params["device"] != "cpu":
            gc.collect()
            torch.cuda.empty_cache()

        self.inputs_info["x_dim"] = preprocessed_x.shape[1]
        self.inputs_info["y_dim"] = preprocessed_y.shape[1]

        dataset = self._build_dataset(
            x=x, preprocessed_x=preprocessed_x, metadata=metadata, y=y, preprocessed_y=preprocessed_y, eval=False
        )
        print("dataset size", len(dataset))
        assert len(dataset) > 0

        self.inputs_info["inputs_decomposer_components"] = pre_post_process.preprocesses["inputs_decomposer"].components_
        self.inputs_info["targets_decomposer_components"] = pre_post_process.preprocesses["targets_decomposer"].components_

        y_statistic = {
            "y_loc": np.mean(preprocessed_y, axis=0),
            "y_scale": np.std(preprocessed_y, axis=0),
        }

        if "targets_global_median" in pre_post_process.preprocesses:
            y_statistic["targets_global_median"] = pre_post_process.preprocesses["targets_global_median"]
        self.inputs_info["y_statistic"] = y_statistic
        batch_size = self.params["train_batch_size"]
        if batch_size > len(dataset):
            batch_size = len(dataset)
        num_workers = int(os.getenv("OMP_NUM_THREADS", 1))

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        self.model = self._build_model()

        self.model.to(device=self.params["device"])

        #! 很奇怪 -----------------------------------------------------------------------------------
        dummy_batch = next(iter(data_loader))
        dummy_batch = self._batch_to_device(dummy_batch)
        self._train_step_forward(dummy_batch, 1.0)

        lr = self.params["lr"]
        eps = self.params["eps"]
        weight_decay = self.params["weight_decay"]
        model_parameters = set_weight_decay(module=self.model, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(model_parameters, lr=lr, eps=eps, weight_decay=weight_decay)
        n_epochs = self.params["epoch"]

        pct_start = self.params["pct_start"]
        total_steps = n_epochs * (len(dataset) // batch_size)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=lr, total_steps=total_steps, pct_start=pct_start
        )

        print("start to train\n")

        if not self.opt['disable_wandb']:
            merged_dict = self.opt
            merged_dict.update(self.params)
            wandb.init(config=merged_dict, project=self.opt.wandb_pj_name, 
                       entity=self.opt.wandb_entity, name=self.opt.exp_name, dir=self.opt.save_dir)
        start_time = time.time()
        self.model.train()
        step_counter = 1
        for epoch in range(n_epochs):
            gc.collect()
            epoch_start_time = time.time()
            if epoch < self.params["burnin_length_epoch"]:
                training_length_ratio = 0.0
            else:
                training_length_ratio = (epoch - self.params["burnin_length_epoch"]) / (
                    n_epochs - self.params["burnin_length_epoch"]
                )
            for _, batch in enumerate(data_loader):
                batch = self._batch_to_device(batch)
                optimizer.zero_grad()
                losses = self._train_step_forward(batch, training_length_ratio)
                losses["loss"].backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clipping)
                optimizer.step()
                scheduler.step()

                if not self.opt.disable_wandb:
                    log_dict = {
                        "Train/loss_step": losses["loss"].item(),
                    }
                    wandb.log(log_dict, step=step_counter)
                step_counter += 1

            end_time = time.time()

            if not self.opt.disable_wandb:
                wandb.log({'learning_rate': scheduler.get_last_lr()[0]}, 
                            step=step_counter)

            if self.params["task_type"] == "multi":
                loss = losses["loss"]
                loss_corr = losses["loss_corr"]
                loss_mse = losses["loss_mse"]
                loss_res_mse = losses["loss_res_mse"]
                loss_total_corr = losses["loss_total_corr"]
        
                print(
                    f"epoch: {epoch} total time: {end_time - start_time:.1f}, "
                    f"epoch time: {end_time - epoch_start_time:.1f}, loss:{loss: .3f} "
                    f"loss_corr:{loss_corr: .3f} "
                    f"loss_mse:{loss_mse: .3f} "
                    f"loss_res_mse:{loss_res_mse: .3f} "
                    f"loss_total_corr:{loss_total_corr: .3f} ",
                    flush=True,
                )
                if not self.opt.disable_wandb:
                    log_dict = {
                        "Train/loss_epoch": loss.item(),
                        "Train/loss_corr": loss_corr.item(), 
                        "Train/loss_mse": loss_mse.item(),
                        "Train/loss_res_mse": loss_res_mse.item(),
                        "Train/loss_total_corr": loss_total_corr.item(),
                    }
                    wandb.log(log_dict, step=step_counter)
            elif self.params["task_type"] == "cite":
                loss = losses["loss"]
                loss_corr = losses["loss_corr"]
                loss_mae = losses["loss_mae"]
                print(
                    f"epoch: {epoch} total time: {end_time - start_time:.1f}, epoch time: {end_time - epoch_start_time:.1f}, loss:{loss: .3f} "
                    f"loss_corr:{loss_corr: .3f} "
                    f"loss_mse:{loss_mae: .3f} ",
                    flush=True,
                )
                if not self.opt.disable_wandb:
                    log_dict = {
                        "Train/loss_epoch": loss.item(),
                        "Train/loss_corr": loss_corr.item(), 
                        "Train/loss_mse": loss_mae.item(),
                    }
                    wandb.log(log_dict, step=step_counter)
            else:
                raise RuntimeError
            #! break  for debug

        print("completed training", flush=True)
        # summeary_torch_model_parameters(self.model)
        self.model.to("cpu")
        return self

    def _build_dataset(self, x, preprocessed_x, metadata, y, preprocessed_y, eval=True):
        selected_metadata = None
        if not eval:
            if "selected_metadata" in self.params:
                selected_metadata = self.params["selected_metadata"]
        if self.params["task_type"] == "multi":
            dataset = MultiomeDataset(
                inputs_values=x,
                preprocessed_inputs_values=preprocessed_x,
                metadata=metadata,
                targets_values=y,
                preprocessed_targets_values=preprocessed_y,
                selected_metadata=selected_metadata,
            )
        elif self.params["task_type"] == "cite":
            dataset = CITEseqDataset(
                inputs_values=x,
                preprocessed_inputs_values=preprocessed_x,
                metadata=metadata,
                targets_values=y,
                preprocessed_targets_values=preprocessed_y,
                selected_metadata=selected_metadata,
            )
        else:
            raise ValueError

        return dataset

    def predict(self, x, preprocessed_x, metadata):
        if self.params["device"] != "cpu":
            gc.collect()
            torch.cuda.empty_cache()
        self.model = self.model.to(self.params["device"])
        self.model.eval()
        dataset = self._build_dataset(
            x=x, preprocessed_x=preprocessed_x, metadata=metadata, y=None, preprocessed_y=None, eval=True
        )
        test_batch_size = self.params["test_batch_size"]
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, num_workers=0)
        y_pred = []
        with torch.no_grad():
            for batch in data_loader:
                batch = self._batch_to_device(batch)
                y_batch_pred = self.model.predict(*batch[0:3])
                y_batch_pred = y_batch_pred.to("cpu").detach().numpy()
                y_pred.append(y_batch_pred)
        y_pred = np.vstack(y_pred)
        self.model.to("cpu")
        return y_pred

    def save(self, model_dir):
        with open(os.path.join(model_dir, "params.json"), "w") as f:
            json.dump(self.params, f, indent=2)
        with open(os.path.join(model_dir, "inputs_info.pickle"), "wb") as f:
            pickle.dump(self.inputs_info, f)
        self.model.to(device="cpu")
        saved_info = {
            'model': self.model.state_dict(),
        }
        torch.save(saved_info, os.path.join(model_dir, f"{self.opt.task_type}_model.pt"))

    def load(self, model_dir):
        with open(os.path.join(model_dir, "params.json")) as f:
            self.params = json.load(f)
        with open(os.path.join(model_dir, "inputs_info.pickle"), "rb") as f:
            self.inputs_info = pickle.load(f)
        self.model = torch.load(os.path.join(model_dir, "model.pt"))
