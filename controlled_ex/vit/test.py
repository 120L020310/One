import timm
import torch.nn as nn
model = timm.create_model('vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k', pretrained=True, 
                          pretrained_cfg_overlay=dict(file='/home/zyz/data/vit/pytorch_model.bin'))
model.patch_embed.proj = nn.Conv2d(1,512,kernel_size=(16,16),stride=(16,16))
print(model)