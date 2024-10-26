import argparse, glob, logging, os, yaml

import torch, wandb
import numpy as np
from torch.optim import Adam, AdamW
from torch.utils import data

from data.custom_dataset import CMI_PB_Dataset
from tqdm import tqdm
from utils import *

MYLOGGER = logging.getLogger()


class Trainer(object):
    def __init__(self, opt):
        super().__init__()
        
        # self.model = ModelLoader(opt.exp_name, opt).model
        # self.model.to(opt.device)
        
        # self.use_wandb = not opt.disable_wandb
        # self.save_best_model = opt.save_best_model
        # self.weights_dir = opt.weights_dir
        # self.warmup_steps = opt.lr_scheduler_warmup_steps
        # self.penalty_cost = opt.penalty_cost
        
        # if self.use_wandb:
        #     MYLOGGER.info("Initialize W&B")
        #     wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, 
        #                name=opt.exp_name, dir=opt.save_dir)
        #     opt.wandb = wandb
        
        #* NEW: self.train_dl, self.train_n_batch, self.val_dl, self.val_n_batch
        self._prepare_dataloader(opt) 

        # self.optimizer = None
        # self.cur_step = 0
        # self.train_num_steps = opt.train_num_steps
        # self.device = opt.device
        # self.bce_loss = torch.nn.BCELoss(reduction='none')
        # self.max_grad_norm = opt.max_grad_norm
        # self.preload_gpu = opt.preload_gpu
        # self.grad_accum_step = opt.grad_accum_step
        # self.disable_scheduler = opt.disable_scheduler
        # self.opt = opt

        # cond = {
        #     'learning_rate': opt.learning_rate,
        #     'adam_betas': opt.adam_betas,
        #     'adam_eps': opt.adam_eps,
        #     'weight_decay': opt.weight_decay,
        #     'sgd_momentum': opt.sgd_momentum,
        #     'sgd_enable_nesterov': opt.sgd_enable_nesterov,
        # }

        # self.optimizer = get_optimizer(opt.optimizer, self.model.parameters(), cond)
        
        # cond = {
        #     'lr_scheduler_factor': opt.lr_scheduler_factor, 
        #     'lr_scheduler_patience': opt.lr_scheduler_patience
        # }
        # self.scheduler_optim = get_lr_scheduler(opt.lr_scheduler, self.optimizer, cond)
        
        # opt.model_num_params = count_model_parameters(self.model)
        
    def _prepare_dataloader(self, opt):
        MYLOGGER.info("Loading training data ...")
        
        # self.train_ds = FoGDataset(opt, mode='train')
        
        # self.val_ds = FoGDataset(opt, mode='val')
        
        # self.test_ds = FoGDataset(opt, mode='test')
        
        # dl = data.DataLoader(self.train_ds, 
        #                     batch_size=opt.batch_size, 
        #                     shuffle=True, 
        #                     pin_memory=False, 
        #                     num_workers=0)
        # self.train_n_batch = len(dl) 
        # self.train_dl = cycle_dataloader(dl)
        # opt.train_n_batch = self.train_n_batch
        
        # dl = data.DataLoader(self.val_ds, 
        #                     batch_size=opt.batch_size, 
        #                     shuffle=False, 
        #                     pin_memory=False, 
        #                     num_workers=0)
        # self.val_n_batch = len(dl)
        # self.val_dl = cycle_dataloader(dl)
        # opt.val_n_batch = self.val_n_batch
        
        # dl = data.DataLoader(self.test_ds, 
        #                     batch_size=opt.batch_size, 
        #                     shuffle=False, 
        #                     pin_memory=False, 
        #                     num_workers=0)
        # self.test_n_batch = len(dl)
        # self.test_dl = cycle_dataloader(dl)
        # opt.test_n_batch = self.test_n_batch
        
    def _save_model(self, step, base, best=False):
        
        data = {
            'step': step,
            'model': self.model.state_dict(),
        }
        # delete previous best* or model*
        if best: 
            search_pattern = os.path.join(self.weights_dir, f"best_model_{base}*")
        else:
            search_pattern = os.path.join(self.weights_dir, "model*")
        files_to_delete = glob.glob(search_pattern)

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
            except OSError as e:
                MYLOGGER.error(f"Error deleting file {file_path}: {e}")
            
        filename = f"best_model_{base}_{step}.pt" if best else f"model_{base}_{step}.pt"
        torch.save(data, os.path.join(self.weights_dir, filename))      
        
    def _loss_func(self, pred, gt, train_mode=False):
        """Compute the Binary Cross-Entropy loss for each class and sum over the class dimension

        Args:
            pred: (B, window, 1) prob
            gt: (B, window, 3) one hot
        """
        penalty_cost = self.penalty_cost if train_mode else 0
        max_indices = torch.argmax(gt, dim=2, keepdim=True) # (B, window, 1)
        tmp_gt_mask = (max_indices != 2).float() # (B, window, 1)
        tmp_gt = max_indices * tmp_gt_mask # (B, window, 1)
        
        loss = self.bce_loss(pred, tmp_gt) # (B, window, 1)

        mask = (gt[:,:,2] != 1).float() # (B, window)
        mask = mask.unsqueeze(-1) # (B, window, 1)
        
        # Additional cost for misclassifying the minority class
        minority_mask = (gt[:,:,1] == 1).float() # (B, window)
        minority_mask = minority_mask.unsqueeze(-1) # (B, window, 1)
        loss = loss * (mask + penalty_cost * minority_mask)
   
        return loss.sum() / mask.sum()

    def _evaluation_metrics(self, output, gt):
        """Generate precision, recall, and f1 score.

        Args:
            output: (B, window, 1)   # prob class
            gt (inference):   (B, window, 3)   # one hot
        """
        # Convert the model output probabilities to class predictions
        pred = torch.round(output)  # (B, window, 1)

        # Extract the first two classes from the ground truth
        real = torch.argmax(gt[:, :, :2], dim=-1, keepdim=True)  # (B, window, 1)

        # Create a mask to ignore the positions where the ground truth class is 2
        mask = (gt[:, :, 2] != 1).unsqueeze(-1)  # (B, window, 1)

        # Apply the mask to the predictions and ground truth
        pred = (pred * mask.float()).squeeze() # (B, window)
        real = (real * mask.float()).squeeze() # (B, window)
        

        # Calculate true positives, false positives, and false negatives
        tp = ((pred == 1) & (real == 1)).float().sum()
        fp = ((pred == 1) & (real == 0)).float().sum()
        fn = ((pred == 0) & (real == 1)).float().sum()

        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return precision, recall, f1

    def _eval_test_data(self, step, base):
        avg_test_f1, avg_test_loss, avg_test_prec, avg_test_recall = 0.0, 0.0, 0.0, 0.0      
        self.model.eval()
        with torch.no_grad():
            for _ in tqdm(range(self.test_n_batch), desc=f"test data"):
                test_data = next(self.test_dl) 
                test_gt = test_data['gt'] # (B, window, 3)
                
                test_input = {}    
                for idx, body_name in test_data['idx_feats'].items():
                    # (BS, window, 3)
                    test_input[body_name[0]] = test_data[body_name[0]]
                    
                    if not self.preload_gpu:
                        test_input[body_name[0]] = test_input[body_name[0]].to(self.device)

                test_input['event'] = test_data['event'] # (bs)
                
                test_pred = self.model(test_input) # (B, window, 1)
                
                prec, recall, f1 = self._evaluation_metrics(test_pred, 
                                                        test_gt.to(self.device))
                test_loss = self._loss_func(test_pred, test_gt.to(self.device))
                
                avg_test_f1 += f1
                avg_test_loss += test_loss
                avg_test_prec += prec
                avg_test_recall += recall
                
            avg_test_f1 /= self.test_n_batch
            avg_test_loss /= self.test_n_batch
            avg_test_prec /= self.test_n_batch
            avg_test_recall /= self.test_n_batch
            
            avg_test_f1 = round(avg_test_f1.item(), 4)
            avg_test_prec = round(avg_test_prec.item(), 4)
            avg_test_recall = round(avg_test_recall.item(), 4)
            avg_test_loss = avg_test_loss.item()
            
            MYLOGGER.info(f"avg_test_loss: {avg_test_loss}")
            MYLOGGER.info(f"avg_test_f1: {avg_test_f1}")
            MYLOGGER.info(f"avg_test_prec: {avg_test_prec}")
            MYLOGGER.info(f"avg_test_recall: {avg_test_recall}")
            
            if self.use_wandb:
                wandb.run.summary[f'best_test_f1_from_val_{base} / step'] = [avg_test_f1, step]
                wandb.run.summary[f'best_test_prec_from_val_{base} / step'] = [avg_test_prec, step]
                wandb.run.summary[f'best_test_recall_from_val_{base} / step'] = [avg_test_recall, step]
                wandb.run.summary[f'best_test_loss_from_val_{base} / step'] = [avg_test_loss, step]

    def train(self):
        best_val_f1 = 0
        best_val_prec = 0
        best_val_recall = 0
        best_val_loss = float('inf')

        for step_idx in tqdm(range(0, self.train_num_steps), desc="Train"):
            self.model.train()
            
            #* training part -----------------------------------------------------------------------
            train_data = next(self.train_dl)
            
            # print(train_data.keys())
            # print(train_data['idx_feats'].keys())
            # print(len(train_data['idx_feats']))
            # print(train_data['lowerback_acc'].shape)
            # print(len(train_data['event']))
            # print(len(train_data['event'][0]))
            # exit(0)
            
            train_gt = train_data['gt'] # (B, window, 3) one-hot        
            train_input = {}    
            for idx, body_name in train_data['idx_feats'].items():
                # (BS, window, 3)
                train_input[body_name[0]] = train_data[body_name[0]]
                
                if not self.preload_gpu:
                    train_input[body_name[0]] = train_input[body_name[0]].to(self.device)

            # train_input['event'] = [list(i) for i in zip(*train_data['event'])]
            # train_input['event'] = [list(tup) for tup in train_data['event']] # (window, bs)
            
            train_input['event'] = train_data['event']
            
            train_pred = self.model(train_input) # (B,window,1)
            
            
            train_loss = self._loss_func(train_pred, train_gt.to(self.device), train_mode=True)
            train_loss /= self.grad_accum_step
            train_loss.backward()
            
            # check gradients
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).\
                                        to(self.device) for p in parameters]), 2.0)
            nan_exist = False
            if torch.isnan(total_norm):
                MYLOGGER.warning('NaN gradients. Skipping to next data...')
                torch.cuda.empty_cache()
                nan_exist = True
            
            if (step_idx + 1) % self.grad_accum_step == 0:
                if nan_exist:
                    continue
                if self.max_grad_norm is not None:
                    # parameters = [p for p in self.model.parameters() if p.grad is not None]
                    # print("before clipping")
                    # print(parameters[0].shape)
                    # print(parameters[0])
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    # parameters = [p for p in self.model.parameters() if p.grad is not None]
  
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if self.use_wandb:
                log_dict = {
                    "Train/loss": train_loss.item(),
                }
                wandb.log(log_dict, step=step_idx+1)
            
            if not self.preload_gpu:
                torch.cuda.empty_cache()
                
            #* validation part ---------------------------------------------------------------------
            cur_epoch = (step_idx + 1) // self.train_n_batch
            if not self.use_wandb or (step_idx + 1) % self.train_n_batch == 0: #* an epoch
            # if True:
                avg_val_f1, avg_val_loss, avg_val_prec, avg_val_recall = 0.0, 0.0, 0.0, 0.0
                
                self.model.eval()
                with torch.no_grad():
                    for _ in tqdm(range(self.val_n_batch), desc=f"Validation at epoch {cur_epoch}"):
                        
                        val_data = next(self.val_dl)

                        val_gt = val_data['gt'] # (B, window, 3)
                        
                        val_input = {}    
                        for idx, body_name in val_data['idx_feats'].items():
                            # (BS, window, 3)
                            val_input[body_name[0]] = val_data[body_name[0]]
                            
                            if not self.preload_gpu:
                                val_input[body_name[0]] = val_input[body_name[0]].to(self.device)

                        val_input['event'] = val_data['event'] # (bs)
                        
                        val_pred = self.model(val_input) # (B, window, 1)
                        
                        prec, recall, f1 = self._evaluation_metrics(val_pred, 
                                                                    val_gt.to(self.device))
                        val_loss = self._loss_func(val_pred, val_gt.to(self.device))
                        
                        avg_val_f1 += f1
                        avg_val_loss += val_loss
                        avg_val_prec += prec
                        avg_val_recall += recall
                        
                    avg_val_f1 /= self.val_n_batch
                    avg_val_loss /= self.val_n_batch
                    avg_val_prec /= self.val_n_batch
                    avg_val_recall /= self.val_n_batch
                    
                    if self.use_wandb:
                        log_dict = {
                            "Val/avg_val_loss": avg_val_loss.item(),
                            "Val/avg_val_f1": avg_val_f1.item(),
                            "Val/avg_val_prec": avg_val_prec.item(),
                            "Val/avg_val_recall": avg_val_recall.item(),
                            # "Val/pr_auc": pr_auc,
                        }
                        wandb.log(log_dict, step=step_idx+1)
        
                    MYLOGGER.info(f"avg_val_loss: {avg_val_loss.item():4f}")
                    MYLOGGER.info(f"avg_val_f1: {avg_val_f1.item():4f}")
                    MYLOGGER.info(f"avg_val_prec: {avg_val_prec.item():4f}")
                    MYLOGGER.info(f"avg_val_recall: {avg_val_recall.item():4f}")

                # Log learning rate
                if self.use_wandb:
                    wandb.log({'learning_rate': self.scheduler_optim.get_last_lr()[0]}, 
                              step=step_idx+1)

                if self.save_best_model and avg_val_f1 > best_val_f1:
                    best_val_f1 = avg_val_f1
                    tmp = f"{avg_val_f1.item():4f}"
                    if self.use_wandb:
                        wandb.run.summary['best_val_f1 / step'] = [tmp, step_idx+1]
                    self._save_model(step_idx + 1, base='f1', best=True)
                    self._eval_test_data(step=step_idx + 1, base='f1')

                if self.save_best_model and avg_val_prec > best_val_prec:
                    best_val_prec = avg_val_prec
                    tmp = f"{avg_val_prec.item():4f}"
                    if self.use_wandb:
                        wandb.run.summary['best_val_prec / step'] = [tmp, step_idx+1]
                    self._save_model(step_idx + 1, base='prec', best=True)
                    self._eval_test_data(step=step_idx + 1, base='prec')
                    
                if self.save_best_model and avg_val_recall > best_val_recall:
                    best_val_recall = avg_val_recall
                    tmp = f"{avg_val_recall.item():4f}"
                    if self.use_wandb:
                        wandb.run.summary['best_val_recall / step'] = [tmp, step_idx+1]
                    self._save_model(step_idx + 1, base='recall', best=True)
                    self._eval_test_data(step=step_idx + 1, base='recall')
                    
                if self.save_best_model and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if self.use_wandb:
                        wandb.run.summary['best_val_loss / step'] = [avg_val_loss.item(), 
                                                                     step_idx+1]
                    self._save_model(step_idx + 1, base='loss', best=True)
                    self._eval_test_data(step=step_idx + 1, base='loss')
                    
                self._save_model(step_idx + 1, base='regular', best=False)
            
                #* learning rate scheduler ---------------------------------------------------------
                if cur_epoch > self.warmup_steps and not self.disable_scheduler:    
                    self.scheduler_optim.step(avg_val_loss)
                    
                if not self.preload_gpu:
                    torch.cuda.empty_cache()

        if self.use_wandb:
            wandb.run.finish()

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # project information: names ===============================================
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--exp_name', type=str, required=True, 
                                      help='save to project/name')
    parser.add_argument('--cur_time', default=None, help='Time running this program')
    parser.add_argument('--description', type=str, default=None, help='important notes')

    # wandb setup ==============================================================
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--wandb_pj_name', type=str, default='fog-challenge', 
                                           help='wandb project name')
    parser.add_argument('--entity', default='', help='W&B entity or username')

    # data path
    parser.add_argument('--root_dpath', default='data/rectified_data', 
                                        help='directory that contains different processed datasets')
    parser.add_argument('--train_datasets', type=str, required=True, 
                                       help='provided dataset_name, e.g. kaggle, ...')
    parser.add_argument('--lab_home', type=str, default="lab", 
                                       help='lab, lab_home')
    
    # GPU ======================================================================
    parser.add_argument('--cuda_id', default='0', help='assign gpu')
    parser.add_argument('--device_info', type=str, default='')
    
    # training monitor =========================================================
    parser.add_argument('--save_best_model', action='store_true', 
                                                  help='save best model during training')
    # parser.add_argument('--save_every_n_epoch', type=int, default=50, 
    #                                               help='save model during training')

    # hyperparameters ==========================================================
    parser.add_argument('--seed', type=int, default=42, 
                                      help='set up seed for torch, numpy, random')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--optimizer', type=str, default="adam", 
                                       help="Choice includes [adam, adamw]")
    parser.add_argument('--learning_rate', type=float, default=26e-5, # 0.00026 
                                           help='generator_learning_rate')
    parser.add_argument('--adam_betas', default=(0.9, 0.98), help='betas for Adam optimizer')
    parser.add_argument('--adam_eps', default=1e-9, help='epsilon for Adam optimizer')
    parser.add_argument('--sgd_momentum', type=float, default=0)
    parser.add_argument('--sgd_enable_nesterov', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0, help='for adam optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='lr scheduler')
    parser.add_argument('--lr_scheduler_patience', type=int, default=10, help='for adam optimizer')
    parser.add_argument('--lr_scheduler_warmup_steps', type=int, default=64, help='lr scheduler')

    parser.add_argument('--train_num_steps', type=int, default=20000, 
                                                 help='number of training steps')
    parser.add_argument('--penalty_cost', type=float, default=0, 
                                          help='penalize when misclassifying the minor class(fog)')
    
    parser.add_argument('--random_aug', action='store_true', help="randomly augment data")
    parser.add_argument('--feats', type=str, nargs='+', default=FEATURES_LIST, 
                                                 help='number of features in raw data')
    
    parser.add_argument('--window', type=int, default=1024, help="-1 means using full trial") 

    parser.add_argument('--max_grad_norm', type=float, default=None, 
                                           help="prevent gradient explosion")

    parser.add_argument('--grad_accum_step', type=int, default=1)
    
    parser.add_argument('--preload_gpu', action='store_true', help="preload all data to gpu")
    parser.add_argument('--disable_scheduler', action='store_true', help="no adaptive lr")
    
    parser.add_argument('--fog_model_input_dim', type=int, default=3)
    parser.add_argument('--fog_model_feat_dim', type=int, default=250)
    parser.add_argument('--fog_model_nheads', type=int, default=10)
    parser.add_argument('--fog_model_nlayers', type=int, default=5)
    parser.add_argument('--fog_model_lstm_nlayers', type=int, default=2)
    parser.add_argument('--fog_model_first_dropout', type=float, default=0.1)
    parser.add_argument('--fog_model_encoder_dropout', type=float, default=0.1)
    parser.add_argument('--fog_model_mha_dropout', type=float, default=0.0)
    
    parser.add_argument('--clip_dim', type=int, default=512)
    parser.add_argument('--clip_version', type=str, default='ViT-B/32')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--txt_cond', action='store_true', help="give text as condition")
    
    #! may need to change if embed annotation
    # parser.add_argument('--fog_model_input_dim', type=int, default=18*(len(FEATURES_LIST)-1))

    # parser.add_argument('--feats_list', type=str, nargs='+', default=FEATURES_LIST)
    
    # file tracker =============================================================
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--weights_dir', type=str, default='')

    opt = parser.parse_args()
    
    cur_time = get_cur_time()
    opt.save_dir = os.path.join(opt.project, opt.exp_name, cur_time)
    opt.cur_time = cur_time
    opt.weights_dir = os.path.join(opt.save_dir, 'weights')
    opt.codes_dir = os.path.join(opt.save_dir, 'codes')
    opt.device_info = torch.cuda.get_device_name(int(opt.cuda_id)) 
    opt.device = f"cuda:{opt.cuda_id}"
    opt.feats = DATASETS_FEATS[opt.train_datasets]
        
    return opt

if __name__ == "__main__":
    assert torch.cuda.is_available(), "**** No available GPUs."
    
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # opt = parse_opt()

    # create_training_folders(opt)
    # save_codes(opt)
    # set_seed(opt)
    
    # trainer = Trainer(opt)
    
    # print_initial_info(opt, model=trainer.model)
    # print_initial_info(opt, redirect_file="model_info.log", model=trainer.model)
    
    # set_redirect_printing(opt)
    
    # save_group_args(opt)
    
    # trainer.train()
    # torch.cuda.empty_cache()
