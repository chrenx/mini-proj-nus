import torch
from torch import nn

from scipy.stats import pearsonr

from utils.suzuki.model.torch_dataset.multiome_dataset import METADATA_KEYS
from utils.suzuki.model.torch_helper.correlation_loss import correlation_loss, correlation_score
from utils.suzuki.model.torch_helper.row_normalize import row_normalize


class MultiUnet(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        y_statistic,
        encoder_h_dim,
        decoder_h_dim,
        n_decoder_block,
        inputs_decomposer_components,
        targets_decomposer_components,
        channel=10,
        encoder=None,
        decoder=None,
    ):
        super(MultiUnet, self).__init__()
        self.x_dim = x_dim # 256
        self.y_dim = y_dim # 128
        self.info_dim = len(METADATA_KEYS)
        # self.encoder = encoder


        self.y_loc = torch.nn.Parameter(y_statistic["y_loc"], requires_grad=False)
        self.y_scale = torch.nn.Parameter(y_statistic["y_scale"], requires_grad=False)
        # print("y_loc: ", self.y_loc.shape)     # (128,)
        # print("y_scale: ", self.y_scale.shape) # (128,)

        # print("inputs_decomposer_components: ", inputs_decomposer_components.shape)
        # print("targets_decomposer_components: ", targets_decomposer_components.shape)

        # inputs_decomposer_components:  torch.Size([128, 228942])
        # targets_decomposer_components:  torch.Size([128, 23418])
        # targest_global_median             (128, 23418)


        self.inputs_decomposer_components = torch.nn.Parameter(inputs_decomposer_components, requires_grad=False)
        self.targets_decomposer_components = torch.nn.Parameter(targets_decomposer_components, requires_grad=False)
        self.targets_global_median = torch.nn.Parameter(y_statistic["targets_global_median"], requires_grad=False)
        self.correlation_loss_func = correlation_loss
        self.mse_loss_func = nn.MSELoss()


        self.decoder = decoder
        decoder_out_fcs = []
        decoder_out_res_fcs = []
        for _ in range(n_decoder_block + 1):
            decoder_out_fcs.append(nn.Linear(decoder_h_dim, y_dim))
            decoder_out_res_fcs.append(nn.Linear(decoder_h_dim, self.targets_decomposer_components.shape[1]))
        self.decoder_out_fcs = nn.ModuleList(decoder_out_fcs)
        self.decoder_out_res_fcs = nn.ModuleList(decoder_out_res_fcs)

        self.layer_y_preds = nn.Linear(self.x_dim, self.y_dim) # 256 -> 128
        self.layer_y_res_preds = nn.Linear(self.x_dim, self.targets_decomposer_components.shape[1])



        self.activ = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        #* Encoding layers *************************************************************************
        self.e11 = nn.Conv1d(channel, 32, kernel_size=3, padding='same')
        self.e12 = nn.Conv1d(32, 32, kernel_size=3, padding='same')
        self.maxpool1 = nn.MaxPool1d(kernel_size=2) # 1024
        
        self.e21 = nn.Conv1d(32, 64, kernel_size=3, padding='same')
        self.e22 = nn.Conv1d(64, 64, kernel_size=3, padding='same')
        self.maxpool2 = nn.MaxPool1d(kernel_size=2) # 512
        
        self.e31 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.e32 = nn.Conv1d(128, 128, kernel_size=3, padding='same')
        self.maxpool3 = nn.MaxPool1d(kernel_size=2) # 256
        
        self.e41 = nn.Conv1d(128, 256, kernel_size=3, padding='same')
        self.e42 = nn.Conv1d(256, 256, kernel_size=3, padding='same')
        self.maxpool4 = nn.MaxPool1d(kernel_size=2) # 128
        
        self.e51 = nn.Conv1d(256, 512, kernel_size=3, padding='same')
        self.e52 = nn.Conv1d(512, 512, kernel_size=3, padding='same') #128*512
        self.maxpool5 = nn.MaxPool1d(kernel_size=2) # 64

        #* Decoding layers *************************************************************************
        self.d51 = nn.Conv1d(512, 512, kernel_size=3, padding='same')
        self.d52 = nn.Conv1d(512, 512, kernel_size=3, padding='same')
        self.up5 = nn.ConvTranspose1d(512, 512, kernel_size=2, stride=2, padding=0)
        # cat: up5, e52: 512 + 512 = 1024
        
        self.d41 = nn.Conv1d(1024, 256, kernel_size=3, padding='same')
        self.d42 = nn.Conv1d(256, 256, kernel_size=3, padding='same')
        self.up4 = nn.ConvTranspose1d(256, 512, kernel_size=2, stride=2, padding=0)
        # cat: up4, e42: 512 + 256 = 768
        
        self.d31 = nn.Conv1d(768, 128, kernel_size=3, padding='same')
        self.d32 = nn.Conv1d(128, 128, kernel_size=3, padding='same')
        self.up3 = nn.ConvTranspose1d(128, 512, kernel_size=2, stride=2, padding=0)
        # cat: up3, e32: 512 + 128 = 640
        
        self.d21 = nn.Conv1d(640, 64, kernel_size=3, padding='same')
        self.d22 = nn.Conv1d(64, 64, kernel_size=3, padding='same')
        self.up2 = nn.ConvTranspose1d(64, 512, kernel_size=2, stride=2, padding=0)
        # cat: up2, e22: 512 + 64 = 576
        
        self.d11 = nn.Conv1d(576, 64, kernel_size=3, padding='same')
        self.d12 = nn.Conv1d(64, 64, kernel_size=3, padding='same')
        self.up1 = nn.ConvTranspose1d(64, 512, kernel_size=2, stride=2, padding=0)
        # cat: up1, e12: 512 + 32 = 544
        
        # self.outconv = nn.Conv1d(544, 1, kernel_size=3, padding='same')
        self.outconv = nn.Conv1d(544, n_decoder_block + 1, kernel_size=3, padding='same')


    def _decode(self, z):
        # z: (B, 512)
        h = z
        _, hs = self.decoder(h)
        ys = []
        y_reses = []
        for i in range(len(hs)):
            new_h = hs[i]
            y_base = self.decoder_out_fcs[i](new_h)
            y = y_base * self.y_scale[None, :] + self.y_loc[None, :]
            ys.append(y)
            y_res = self.decoder_out_res_fcs[i](new_h)
            y_reses.append(y_res)
        return ys, y_reses

    def forward(self, x):
        # x: (B,C,D)
        #* Encoding layers *************************************************************************
        xe11 = self.activ(self.e11(x))
        xe12 = self.activ(self.e12(xe11))
        xp1 = self.maxpool1(xe12)  # 128
        
        xe21 = self.activ(self.e21(xp1))
        xe22 = self.activ(self.e22(xe21))
        xp2 = self.maxpool2(xe22)  # 64
        
        xe31 = self.activ(self.e31(xp2))
        xe32 = self.activ(self.e32(xe31))
        xp3 = self.maxpool3(xe32)  # 32
        
        xe41 = self.activ(self.e41(xp3))
        xe42 = self.activ(self.e42(xe41))
        xp4 = self.maxpool4(xe42) # 16
        
        xe51 = self.activ(self.e51(xp4))
        xe52 = self.activ(self.e52(xe51))
        xp5 = self.maxpool5(xe52)  # (64, 512, 8)

        # split into 8 pieces
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        p = xp5.shape[2]
        all_y_preds, all_y_res_preds = [], []
        for i in range(p):
            ys, y_reses = self._decode(xp5[:,:,i]) # [(B, 128), ...] [(B,23418), ...]
            all_y_preds.append(ys)
            all_y_res_preds.append(y_reses)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        
        #* Decoding layers *************************************************************************
        xd51 = self.activ(self.d51(xp5))
        xd52 = self.activ(self.d52(xd51))
        xup5 = self.activ(self.up5(xd52))
        xup5 = torch.cat([xup5, xe52], dim=-2)
        
        xd41 = self.activ(self.d41(xup5))
        xd42 = self.activ(self.d42(xd41))
        xup4 = self.activ(self.up4(xd42))
        xup4 = torch.cat([xup4, xe42], dim=-2)
        
        xd31 = self.activ(self.d31(xup4))
        xd32 = self.activ(self.d32(xd31))
        xup3 = self.activ(self.up3(xd32))
        xup3 = torch.cat([xup3, xe32], dim=-2)
        
        xd21 = self.activ(self.d21(xup3))
        xd22 = self.activ(self.d22(xd21))
        xup2 = self.activ(self.up2(xd22))
        xup2 = torch.cat([xup2, xe22], dim=-2)
        
        xd11 = self.activ(self.d11(xup2))
        xd12 = self.activ(self.d12(xd11))
        xup1 = self.activ(self.up1(xd12))
        xup1 = torch.cat([xup1, xe12], dim=-2)
        
        decoding = self.outconv(xup1) # torch.Size([64, 1, 256]) (B, 1, D)
        decoding = decoding.squeeze()
     
        y_preds_2 = self.layer_y_preds(self.activ(decoding)) # (B, 6, 128)
        y_res_preds_2 = self.layer_y_res_preds(self.activ(decoding)) # (B, 6, 23418)

        y_preds_2_list = []    # len: n_decoder_block + 1
        y_res_preds_2_list = []   # len: n_decoder_block + 1
        for i in range(y_preds_2.shape[1]):
            y_preds_2_list.append(y_preds_2[:,i,:])
            y_res_preds_2_list.append(y_res_preds_2[:,i,:])

        all_y_preds.append(y_preds_2_list)
        all_y_res_preds.append(y_res_preds_2_list)

        avg_y_preds = [torch.stack(tensors).mean(dim=0) for tensors in zip(*all_y_preds)]
        avg_y_res_preds = [torch.stack(tensors).mean(dim=0) for tensors in zip(*all_y_res_preds)]


        # x shape:  torch.Size([64, 256])
        # z shape: torch.Size([64, 2048])
        # y_preds:6, (64,128), y_res_preds:6, (64, 23418) target

        # z = self._encode(x, gender_id, nonzero_ratio)
        # y_preds, y_res_preds = self._decode(z, None, None)

        return avg_y_preds, avg_y_res_preds

    def loss(self, x, gender_id, info, day_id, donor_id, y, preprocessed_y, training_length_ratio):
        # x:            (B, 256)
        # gender_id:    (B,)
        # info:         (B, 7)
        # day_id:       (B,)
        # donor_id:     (B,)
        B = x.shape[0]
        D = x.shape[1]
        C = 3 + info.shape[1] # 10
        new_gender_id = gender_id.view(B, 1, 1).repeat(1, 1, D) # (B,1,256)
        new_day_id = day_id.view(B, 1, 1).repeat(1, 1, D)
        new_donor_id = donor_id.view(B, 1, 1).repeat(1, 1, D)
        new_info = info.unsqueeze(2).repeat(1, 1, D)            # (B,7,256)
        new_x = x.unsqueeze(1).repeat(1,C,1)

        # print(new_x[0,0,:5])
        # print("new_gender_id: ", new_gender_id.shape)
        # print("new_day_id: ", new_day_id.shape)
        # print("new_donor_id: ", new_donor_id.shape)
        # print("new_info: ", new_info.shape)

        cond = torch.cat([new_day_id, new_donor_id, new_gender_id, new_info], dim=1) # (B,C,D)
        
        # print("cond: ", cond.shape)
        # print("new_x1: ", new_x.shape) # (B,C,D)

        new_x += cond # (B,C,D)

        # print("new_x", new_x.shape)
        # print(new_x[0,0,:5])

        ret = {
            "loss": 0,
            "loss_corr": 0,
            "loss_mse": 0,
            "loss_res_mse": 0,
            "loss_total_corr": 0,
            "pcc": 0,
        }

        y_preds, y_res_preds = self(new_x)
    
        normalized_y = row_normalize(y)
        for i in range(len(y_preds)):
            y_pred = y_preds[i]
            y_res_pred = y_res_preds[i]
                                            #  (B, 128)  (128, 23418)
            postprocessed_y_pred = torch.matmul(y_pred, self.targets_decomposer_components) + \
                                                        self.targets_global_median[None, :]
            normalized_postprocessed_y_pred_detached = row_normalize(postprocessed_y_pred.detach())
            y_res = normalized_y - normalized_postprocessed_y_pred_detached
            y_total_pred = normalized_postprocessed_y_pred_detached + y_res_pred
            ret["loss_corr"] = ret["loss_corr"] + self.correlation_loss_func(postprocessed_y_pred, y)
            ret["loss_mse"] = ret["loss_mse"] + self.mse_loss_func(y_pred, preprocessed_y)
            ret["loss_res_mse"] = ret["loss_res_mse"] + self.mse_loss_func(y_res, y_res_pred)
            ret["loss_total_corr"] = ret["loss_total_corr"] + self.correlation_loss_func(y_total_pred, y)
            ret["pcc"] += correlation_score(y, y_total_pred)

        w = (1 - training_length_ratio) ** 2
        ret["loss_corr"] /= len(y_preds)
        ret["loss"] = ret["loss"] + ret["loss_corr"]
        ret["loss_mse"] /= len(y_preds)
        ret["loss"] = ret["loss"] + w * ret["loss_mse"]
        ret["loss_res_mse"] /= len(y_preds)
        ret["loss"] = ret["loss"] + w * ret["loss_res_mse"]
        ret["loss_total_corr"] /= len(y_preds)
        ret["loss"] = ret["loss"] + ret["loss_total_corr"]
        ret["pcc"] /= len(y_preds)
        ret["pcc"] = torch.mean(ret['pcc'])
        return ret

    def predict(self, x, gender_id, info):
        raise RuntimeError
        y_preds, y_res_preds = self(x, gender_id, info)
        postprocessed_y_pred = None
        for i in range(len(y_preds)):
            new_postprocessed_y_pred = row_normalize(
                torch.matmul(y_preds[i], self.targets_decomposer_components) + self.targets_global_median[None, :]
            )
            new_postprocessed_y_pred += y_res_preds[i]
            new_postprocessed_y_pred = row_normalize(new_postprocessed_y_pred)
            if postprocessed_y_pred is None:
                postprocessed_y_pred = new_postprocessed_y_pred
            else:
                postprocessed_y_pred += new_postprocessed_y_pred
        postprocessed_y_pred /= len(y_preds)
        return postprocessed_y_pred
