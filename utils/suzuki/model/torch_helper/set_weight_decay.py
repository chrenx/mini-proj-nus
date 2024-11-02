from torch import nn

from utils.suzuki.model.commander.cite_encoder_decoder_module import CiteEncoderDecoderModule
from utils.suzuki.model.commander.multi_encoder_decoder_module import MultiEncoderDecoderModule
from utils.suzuki.model.commander.multi_unet import MultiUnet


def set_weight_decay(module, weight_decay, opt):
    params_decay = []
    params_no_decay = []
    ignoring_params = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, nn.Conv1d):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            if m.bias is not None:
                params_no_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, (CiteEncoderDecoderModule, MultiEncoderDecoderModule, MultiUnet)):
            ignoring_params.append(m.inputs_decomposer_components)
            ignoring_params.append(m.targets_decomposer_components)
            if hasattr(m, "targets_global_median"):
                ignoring_params.append(m.targets_global_median)
            ignoring_params.append(m.y_loc)
            ignoring_params.append(m.y_scale)
            
            if opt.backbone == "mlp":
                params_no_decay.append(m.gender_embedding)
                params_no_decay.append(m.day_embedding)
                params_no_decay.append(m.donor_embedding)
            elif opt.backbone == "unet":
                pass
            else: 
                raise RuntimeError
            
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay) + len(ignoring_params)
    params = [dict(params=params_decay, weight_decay=weight_decay), dict(params=params_no_decay, weight_decay=0.0)]

    return params
