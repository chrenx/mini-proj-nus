import gc, glob, json, os, pickle, time

import numpy as np
import torch, wandb
from torch.utils.data import Subset
from tqdm import tqdm

from utils import split_dataset_save_load_idx
from utils.suzuki.model.commander.cite_encoder_decoder_module import CiteEncoderDecoderModule
from utils.suzuki.model.commander.mlp_module import HierarchicalMLPBModule, MLPBModule
from utils.suzuki.model.commander.multi_encoder_decoder_module import MultiEncoderDecoderModule
from utils.suzuki.model.commander.multi_unet import MultiUnet
from utils.suzuki.model.torch_dataset.citeseq_dataset import CITEseqDataset
from utils.suzuki.model.torch_dataset.multiome_dataset import MultiomeDataset
from utils.suzuki.model.torch_helper.set_weight_decay import set_weight_decay
# from utils.suzuki.utility.summeary_torch_model_parameters import summeary_torch_model_parameters


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
                "encoder_h_dim": opt.encoder_h_dim,  # 2048,
                "decoder_h_dim": opt.decoder_h_dim,  # 128,
                "encoder_dropout_p": opt.encoder_dropout_p,
                "decoder_dropout_p": opt.decoder_dropout_p,
                "n_encoder_block": opt.n_encoder_block,
                "n_decoder_block": opt.n_decoder_block,
                "norm": opt.norm,
                "activation": opt.activation,  # relu, "gelu"
                "skip": opt.skip,
            }
        elif params["backbone"] == "unet":
            # TODO
            backbone_params = {
                "encoder_h_dim": 512,
                "decoder_h_dim": 512,  # 128,
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
        elif self.params["backbone"] == "unet":
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
            if self.params["backbone"] == "mlp":
                model = MultiEncoderDecoderModule(
                    x_dim=x_dim, # 256
                    y_dim=y_dim, # 128
                    y_statistic=y_statistic,
                    encoder_h_dim=self.params["encoder_h_dim"],
                    decoder_h_dim=self.params["decoder_h_dim"],
                    n_decoder_block=self.params["n_decoder_block"],
                    encoder=encoder,
                    decoder=decoder,
                    inputs_decomposer_components=inputs_decomposer_components,
                    targets_decomposer_components=targets_decomposer_components,
                )
            elif self.params["backbone"] == "unet":
                model = MultiUnet(
                    x_dim=x_dim, # 256
                    y_dim=y_dim, # 128
                    y_statistic=y_statistic,
                    encoder_h_dim=self.params["encoder_h_dim"],
                    decoder_h_dim=self.params["decoder_h_dim"],
                    n_decoder_block=self.params["n_decoder_block"],
                    channel=self.opt.channel,
                    encoder=None,
                    decoder=decoder,
                    inputs_decomposer_components=inputs_decomposer_components,
                    targets_decomposer_components=targets_decomposer_components,
                )
            else:
                raise ValueError
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
            x=x, preprocessed_x=preprocessed_x, metadata=metadata, 
            y=y, preprocessed_y=preprocessed_y, eval=False
        )
        print("dataset size", len(dataset))
        assert len(dataset) > 0

        train_idx, val_idx, test_idx = split_dataset_save_load_idx(dataset, self.opt, metadata)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

        print("len train dataset:", len(train_dataset))
        print("len val dataset:", len(val_dataset))
        print("len test dataset:", len(test_dataset))

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

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # drop_last=True,
            num_workers=num_workers,
        )

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            # drop_last=True,
            num_workers=num_workers,
        )

        test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            # drop_last=True,
            num_workers=num_workers,
        )

        self.model = self._build_model()

        self.model.to(device=self.params["device"])

        #! 为了一些var的初始化  -----------------------------------------------------------------------
        # dummy_batch = next(iter(train_data_loader))
        # dummy_batch = self._batch_to_device(dummy_batch)
        # self._train_step_forward(dummy_batch, 1.0)

        lr = self.params["lr"]
        eps = self.params["eps"]
        weight_decay = self.params["weight_decay"]
        if self.opt.backbone == "mlp":
            model_parameters = set_weight_decay(module=self.model, weight_decay=weight_decay, 
                                                opt=self.opt)

            optimizer = torch.optim.Adam(model_parameters, lr=lr, eps=eps, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
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

        step_counter = 1
        best_val_avg_loss = float('inf')
        best_val_avg_pcc = 0
        for epoch in tqdm(range(n_epochs), desc="Train"):
            gc.collect()
            epoch_start_time = time.time()
            if epoch < self.params["burnin_length_epoch"]:
                training_length_ratio = 0.0
            else:
                training_length_ratio = (epoch - self.params["burnin_length_epoch"]) / (
                    n_epochs - self.params["burnin_length_epoch"]
                )
            #! Training part =======================================================================    
            self.model.train()
            avg_train_loss = 0
            avg_train_pcc = 0
            for _, batch in enumerate(train_data_loader):
                batch = self._batch_to_device(batch)
                optimizer.zero_grad()
                losses = self._train_step_forward(batch, training_length_ratio)
                losses["loss"].backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clipping)
                optimizer.step()
                scheduler.step()

                avg_train_loss += losses["loss"].item()
                avg_train_pcc += losses["pcc"].item()

                if not self.opt.disable_wandb:
                    log_dict = {
                        "Train/loss_step": losses["loss"].item(),
                        "Train/pcc_step": losses["pcc"].item(),
                    }
                    wandb.log(log_dict, step=step_counter)
                step_counter += 1

            end_time = time.time()

            avg_train_loss /= len(train_data_loader)
            avg_train_pcc /= len(train_data_loader)

            if self.params["task_type"] == "multi":
                loss = losses["loss"]
                loss_corr = losses["loss_corr"]
                loss_mse = losses["loss_mse"]
                loss_res_mse = losses["loss_res_mse"]
                loss_total_corr = losses["loss_total_corr"]
                pcc_epoch = losses['pcc']

                print(
                    f"epoch: {epoch} total time: {end_time - start_time:.1f}, "
                    f"epoch time: {end_time - epoch_start_time:.1f}, "
                    f"avg_train_loss: {avg_train_loss}, "
                    f"avg_train_pcc: {avg_train_pcc}",
                    flush=True,
                )
        
                if not self.opt.disable_wandb:
                    log_dict = {
                        "Train/avg_train_loss": avg_train_loss,
                        "Train/avg_train_pcc": avg_train_pcc,
                        "Train/loss_epoch": loss.item(),
                        "Train/loss_corr": loss_corr.item(), 
                        "Train/loss_mse": loss_mse.item(),
                        "Train/loss_res_mse": loss_res_mse.item(),
                        "Train/loss_total_corr": loss_total_corr.item(),
                    }
                    wandb.log(log_dict, step=step_counter)
            else:
                raise RuntimeError
            
            if not self.opt.disable_wandb:
                wandb.log({'learning_rate': scheduler.get_last_lr()[0]}, 
                            step=step_counter)
                
            #! For validation and testing ==========================================================
            self.model.eval()
            with torch.no_grad():
                val_avg_loss = 0.0
                val_avg_pcc = 0.0
                for _, batch in enumerate(val_data_loader):
                    batch = self._batch_to_device(batch)
                    losses = self._train_step_forward(batch, training_length_ratio=0)
                    val_avg_loss += losses['loss'].item()
                    val_avg_pcc += losses['pcc'].item()

                val_avg_loss /= len(val_data_loader)
                val_avg_pcc /= len(val_data_loader)
    
                if not self.opt.disable_wandb:
                    log_dict = {
                        "Val/avg_loss_epoch": val_avg_loss,
                        "Val/avg_pcc_epoch": val_avg_pcc,
                    }
                    wandb.log(log_dict, step=step_counter)
                
                print(f"average VAL loss at epoch {epoch} ---> {val_avg_loss}", flush=True)
                print(f"average VAL pcc at epoch {epoch} ---> {val_avg_pcc}", flush=True)
                
                if val_avg_loss < best_val_avg_loss:
                    best_val_avg_loss = val_avg_loss
                    if not self.opt.disable_wandb:
                        wandb.run.summary['best_val_loss/step/epoch'] = [best_val_avg_loss, 
                                                                         step_counter, epoch]
                    self.run_test(test_data_loader, len(test_data_loader), 
                                  step=step_counter, epoch=epoch, base="loss")
                    self.save_best_model(step_counter, epoch, base="loss")

                if abs(val_avg_pcc) > abs(best_val_avg_pcc):
                    best_val_avg_pcc = val_avg_pcc
                    if not self.opt.disable_wandb:
                        wandb.run.summary['best_val_pcc/step/epoch'] = [best_val_avg_pcc, 
                                                                        step_counter, epoch]
                    self.run_test(test_data_loader, len(test_data_loader), 
                                  step=step_counter, epoch=epoch, base="pcc")
                    self.save_best_model(step_counter, epoch, base="pcc")
                
                print()


        print("completed training", flush=True)
        self.model.to("cpu")
        if not self.opt.disable_wandb:
            wandb.run.finish()
        return self
    
    def run_test(self, test_data_loader, len_test_data_loader, step, epoch, base):
        self.model.eval()
        with torch.no_grad():
            test_avg_loss = 0.0
            test_avg_pcc = 0.0
            for _, batch in enumerate(test_data_loader):
                batch = self._batch_to_device(batch)
                losses = self._train_step_forward(batch, training_length_ratio=0)
                test_avg_loss += losses['loss'].item()
                test_avg_pcc += losses['pcc'].item()


            test_avg_loss /= len_test_data_loader
            test_avg_pcc /= len_test_data_loader
    
            if not self.opt.disable_wandb:
                wandb.run.summary[f'test_loss_from_best_val_{base}/step/epoch'] = [test_avg_loss, 
                                                                                   step, epoch]
                wandb.run.summary[f'test_pcc_from_best_val_{base}/step/epoch'] = [test_avg_pcc, 
                                                                                  step, epoch]
            
            print(f"average TEST loss at epoch {epoch} ---> {test_avg_loss}", flush=True)
            print(f"average TEST pcc at epoch {epoch} ---> {test_avg_pcc}", flush=True)
        
    def _build_dataset(self, x, preprocessed_x, metadata, y, preprocessed_y, eval=True):
        selected_metadata = None
        if not eval:
            if "selected_metadata" in self.params:
                print("!有 selected_metadata")
                selected_metadata = self.params["selected_metadata"]
        if self.params["task_type"] == "multi":
            dataset = MultiomeDataset(
                inputs_values=x,
                preprocessed_inputs_values=preprocessed_x,
                metadata=metadata,
                targets_values=y,
                preprocessed_targets_values=preprocessed_y,
                selected_metadata=selected_metadata,
                opt=self.opt
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

        search_pattern = os.path.join(self.opt.weights_dir, f"best_model_loss*")
        files_to_search = glob.glob(search_pattern)
        assert len(files_to_search) == 1, "more than one best model"
        best_model_path = files_to_search[0]
        weights = torch.load(best_model_path, map_location=self.opt.device, 
                             weights_only=True)['model']
        predict_model = self._build_model()
        predict_model.to(self.params["device"])
        predict_model.load_state_dict(weights)
        
        dataset = self._build_dataset(
            x=x, preprocessed_x=preprocessed_x, metadata=metadata, y=None, preprocessed_y=None, eval=True
        )
        test_batch_size = self.params["test_batch_size"]
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, num_workers=0)
        y_pred = []
        predict_model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = self._batch_to_device(batch)
                y_batch_pred = predict_model.predict(*batch[0:3])
                y_batch_pred = y_batch_pred.to("cpu").detach().numpy()
                y_pred.append(y_batch_pred)
        y_pred = np.vstack(y_pred)

        return y_pred

    def save_best_model(self, step_counter, epoch, base):
        model_info = {
            'step': step_counter,
            'epoch': epoch,
            'model': self.model.state_dict(),
        }
        # delete previous best* or model*
        search_pattern = os.path.join(self.opt.weights_dir, f"best_model_{base}*")
        files_to_delete = glob.glob(search_pattern)
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
        filename = f"best_model_{base}_s{step_counter}_e{epoch}.pt"
        torch.save(model_info, os.path.join(self.opt.weights_dir, filename))     
        print(f"         saved best model at step {step_counter}, epoch {epoch}")     

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
