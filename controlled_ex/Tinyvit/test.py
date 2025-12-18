from tiny_vit import tiny_vit_21m_224
import torch.nn as nn
model = tiny_vit_21m_224(pretrained=True)
# model.patch_embed.seq[0].c = nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
print(model)