# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
from numpy import prod

from torchaudio.transforms import MFCC

from torchvision.models import vgg19

from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification

try:
    from capsules import RoutingCapsules
except ImportError:
    from .capsules import RoutingCapsules


# # Mel

# :::{note}
# 需要使用Mel Spectrogram将输入的音频转换为 $224\times 224$的谱图，然后在通道上叠加三次，得到$3\times 224 \times 224$的输入。　　
# :::

# + editable=true slideshow={"slide_type": ""}
class Melspec(nn.Module):
    def __init__(self):
        super().__init__()

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            hop_length=215,
            n_fft=512,
            n_mels=224,
        )
        # self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=16000,
        #     hop_length=512,
        #     n_fft=2048,
        #     n_mels=224,
        # ) # (48000 -> 224,94)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          x:torch.Tensor: (B, 1, 48000)

        Returns:
            a tensor: (B, 3, 224, 224)
        """
        mel = self.mel_spectrogram(x)
        mel = torch.concat([mel, mel, mel], dim=1)
        return mel


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# ## 测试
# x = torch.randn(2, 1, 16000 * 3)
# m = Melspec()
# m(x).shape

# + [markdown] editable=true slideshow={"slide_type": ""}
# ---
# -

# # VGG18

# + [markdown] editable=true slideshow={"slide_type": ""}
#
# <center><img src="https://cdn.jsdelivr.net/gh/RedamancyAY/CloudImage@main/img/202408031530037.png" width="700" alt="model structrue"/></center>

# + [markdown] editable=true slideshow={"slide_type": ""}
#
# 模型代码见：[torchvision.models.vgg — Torchvision main documentation](https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg19)
#
# 原始模型的forward：
# ```python
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
# ```

# + [markdown] editable=true slideshow={"slide_type": ""}
# :::{warning}
# 不知道是什么VGG18提取出来的是什么格式，
# - 如果是二维特征，那么形状为$512 \times 7 \times 7$
# - 如果是一维特征，就是分类前的特征，那么是$B \times 512$
# :::
#

# + editable=true slideshow={"slide_type": ""}
class VGG18(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = vgg19(weights=torchvision.models.VGG19_Weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          x:torch.Tensor: (B, 3, 224, 224)

        Returns:
            a tensor: (B, 512, 7, 7)
        """
        x = self.model.features(x)  # (B, 512, 7, 7)
        # x = self.model.avgpool(x) # (B, 512, 7, 7)
        # x = torch.flatten(x, 1) # (B, 512)
        return x


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# x = torch.randn(2, 3, 224, 94)
# model = VGG18()
# y = model(x)
# print(y.shape)

# + [markdown] editable=true slideshow={"slide_type": ""}
# # Attention
# -

# :::{warning}
# 不知道是什么VGG18提取出来的是什么格式，这里的attention也知道该怎么实现
# :::
#

# + [markdown] editable=true slideshow={"slide_type": ""}
# Let $F$ denote the set of features extracted by VGG18, where $F=\left\{f_1, f_2, \ldots, f_n\right\}$ and each $f_i$ is a feature vector. The attention mechanism assigns a weight $w_i$ to each feature vector $f_i$, with the weights being determined by a trainable attention layer. The output of the attention mechanism, $F^{\prime}$, is a weighted sum of the feature vectors, given by:
# $$
# F^{\prime}=\sum_{i=1}^n w_i \cdot f_i
# $$
#
# The weights $w_i$ are computed using a softmax function over the scores assigned to each feature vector by the attention layer, as follows:
# $$
# w_i=\frac{e^{s\left(f_i\right)}}{\sum_{j=1}^n e^{s\left(f_j\right)}}
# $$
# where $s\left(f_i\right)$ is the score assigned to feature vector $f_i$ by the attention layer, which is typically implemented as a fully connected layer with a single output unit. The softmax function ensures that the weights sum up to 1 , allowing them to be interpreted as probabilities that indicate the importance of each feature vector in the context of the detection task.
# -

# 看起来像，VGG提取出来的特征是($B \times T\times C$), 在$T$维度上进行加权和相加，得到$B\times C$。　
#

# + editable=true slideshow={"slide_type": ""}
from einops import rearrange


# + tags=["style-activity"] editable=true slideshow={"slide_type": ""}
class Attention_to_1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          x:torch.Tensor: (B, 512, 7, 7)

        Returns:
            a tensor: (B, 512, 7, 7)
        """
        x = rearrange(x, "b c h w -> b (h w ) c")
        attn_weight = self.attn(x)  # (b, hw, 1)

        attn_weight = attn_weight.softmax(1)  # (b, hw, 1)
        x = x * attn_weight  # (b, hw, c)
        x = x.sum(1)  # (b, c)
        return x


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# m = Attention_to_1D()
# x = torch.randn(2, 512, 7, 7)
# m(x).shape

# + editable=true slideshow={"slide_type": ""}
class Attention_to_2D(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = nn.Linear(49, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          x:torch.Tensor: (B, 512, 7, 7)

        Returns:
            a tensor: (B, 512, 7, 7)
        """
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b c (h w)")
        attn_weight = self.attn(x)  # (b, c, 1)
        attn_weight = attn_weight.softmax(1)  # (b, hw, 1)

        x = x * attn_weight  # (b, c, (hw))
        x = x.sum((1), keepdims=True)  # (b, 1, hw)
        x = rearrange(x, "b 1 (h w) -> b 1 h w", h=h)
        return x


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# m = Attention_to_2D()
# x = torch.randn(2, 512, 7, 7)
# m(x).shape
# -

# # Capsule Network

# ## conv layer

# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# conv_layer = nn.Sequential(
#     nn.Conv2d(1, 512, 3, stride=2, bias=True, padding=0),
#     nn.ReLU(),
#     nn.Conv2d(512, 512, 3, stride=2, bias=True, padding=0),
# )
# x = torch.randn(2, 1, 7, 7)
# conv_layer(x).shape
# -

# ## PrimaryCaps

# + editable=true slideshow={"slide_type": ""}
class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2):
        """Constructs a list of convolutional layers to be used in
        creating capsule output vectors.
        param num_capsules: number of capsules to create
        param in_channels: input depth of features, default value = 256
        param out_channels: output depth of the convolutional layers, default value = 32
        """
        super(PrimaryCaps, self).__init__()

        # creating a list of convolutional layers for each capsule I want to create
        # all capsules have a conv layer with the same parameters
        self.capsules = nn.ModuleList(
            [nn.Linear(in_features=in_channels, out_features=out_channels) for _ in range(num_capsules)]
        )

    def forward(self, x):
        """Defines the feedforward behavior.
        param x: the input; features from a convolutional layer
        return: a set of normalized, capsule output vectors
        """
        # get batch size of inputs
        batch_size = x.size(0)
        # reshape convolutional layer outputs to be (batch_size, vector_dim=1152, 1)
        # print(self.capsules[0](x).shape)

        u = [capsule(x).view(batch_size, -1, 1) for capsule in self.capsules]

        # stack up output vectors, u, one for each capsule
        u = torch.cat(u, dim=-1)
        # squashing the stack of vectors
        u_squash = self.squash(u)
        return u_squash

    def squash(self, input_tensor):
        """Squashes an input Tensor so it has a magnitude between 0-1.
        param input_tensor: a stack of capsule inputs, s_j
        return: a stack of normalized, capsule output vectors, v_j
        """
        squared_norm = (input_tensor**2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)  # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# x = torch.randn(2, 512)
# m = PrimaryCaps(kernel_size=3, in_channels=512, out_channels=512, stride=1, num_capsules=32)
# m(x).shape
# + editable=true slideshow={"slide_type": ""}



# + editable=true slideshow={"slide_type": ""}



# + editable=true slideshow={"slide_type": ""}
def softmax(input_tensor, dim=1):
    # transpose input
    transposed_input = input_tensor.transpose(dim, len(input_tensor.size()) - 1)
    # calculate softmax
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    # un-transpose result
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input_tensor.size()) - 1)


# dynamic routing
def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    """Performs dynamic routing between two capsule layers.
    param b_ij: initial log probabilities that capsule i should be coupled to capsule j
    param u_hat: input, weighted capsule vectors, W u
    param squash: given, normalizing squash function
    param routing_iterations: number of times to update coupling coefficients
    return: v_j, output capsule vectors
    """
    # update b_ij, c_ij for number of routing iterations
    for iteration in range(routing_iterations):
        # softmax calculation of coupling coefficients, c_ij
        c_ij = softmax(b_ij, dim=2)

        # calculating total capsule inputs, s_j = sum(c_ij*u_hat)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)

        # squashing to get a normalized vector output, v_j
        v_j = squash(s_j)

        # if not on the last iteration, calculate agreement and new b_ij
        if iteration < routing_iterations - 1:
            # agreement
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)

            # new b_ij
            b_ij = b_ij + a_ij

    return v_j  # return latest v_j


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, previous_layer_nodes=32 * 6 * 6, in_channels=8, out_channels=16):
        """Constructs an initial weight matrix, W, and sets class variables.
        param num_capsules: number of capsules to create
        param previous_layer_nodes: dimension of input capsule vector, default value = 1152
        param in_channels: number of capsules in previous layer, default value = 8
        param out_channels: dimensions of output capsule vector, default value = 16
        """
        super(DigitCaps, self).__init__()

        # setting class variables
        self.num_capsules = num_capsules
        self.previous_layer_nodes = previous_layer_nodes  # vector input (dim=1152)
        self.in_channels = in_channels  # previous layer's number of capsules

        # starting out with a randomly initialized weight matrix, W
        # these will be the weights connecting the PrimaryCaps and DigitCaps layers
        self.W = nn.Parameter(torch.randn(num_capsules, previous_layer_nodes, in_channels, out_channels))

    def forward(self, u):
        """Defines the feedforward behavior.
        param u: the input; vectors from the previous PrimaryCaps layer
        return: a set of normalized, capsule output vectors
        """

        # adding batch_size dims and stacking all u vectors
        u = u[None, :, :, None, :]
        # 4D weight matrix
        W = self.W[:, None, :, :, :]

        # calculating u_hat = W*u
        u_hat = torch.matmul(u, W)

        # getting the correct size of b_ij
        # setting them all to 0, initially
        b_ij = torch.zeros(*u_hat.size(), device=u.device)

        # update coupling coefficients and calculate v_j
        v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=1)

        return v_j.transpose(0, 1).squeeze((2, 3))  # return final vector outputs

    def squash(self, input_tensor):
        """Squashes an input Tensor so it has a magnitude between 0-1.
        param input_tensor: a stack of capsule inputs, s_j
        return: a stack of normalized, capsule output vectors, v_j
        """
        # same squash function as before
        squared_norm = (input_tensor**2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)  # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor


# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# d = DigitCaps(in_channels=32, num_capsules=1, out_channels=512, previous_layer_nodes=512).cuda()
# x = torch.randn(2, 512, 32).cuda()
# d(x).shape

# + tags=["style-student", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# d = RoutingCapsules(in_caps=32, in_dim=512, num_caps=2, num_routing=3, dim_caps=512)
#
# x = torch.randn(2, 32, 512)
# d(x).shape

# + editable=true slideshow={"slide_type": ""}
class CN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, bias=True, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=2, bias=True, padding=0),
        )

        num_capsules = 10

        self.primary1 = PrimaryCaps(
            kernel_size=3, in_channels=512, out_channels=512, stride=1, num_capsules=num_capsules
        )
        # self.digits1 = DigitCaps(in_channels=num_capsules, num_capsules=2, out_channels=512, previous_layer_nodes=512)
        self.digits1 = RoutingCapsules(in_caps=num_capsules, in_dim=512, num_caps=2, num_routing=3, dim_caps=512)

        self.primary2 = PrimaryCaps(
            kernel_size=3, in_channels=512, out_channels=512, stride=1, num_capsules=num_capsules
        )
        # self.digits2 = DigitCaps(in_channels=num_capsules, num_capsules=2, out_channels=512, previous_layer_nodes=512)
        self.digits2 = RoutingCapsules(in_caps=num_capsules, in_dim=512, num_caps=2, num_routing=3, dim_caps=512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:torch.Tensor: (B, 512, 7, 7)

        Returns:
            a tensor: (B, 512, 7, 7)
        """
        if x.ndim == 4:
            x = self.conv_layer(x).squeeze(-1).squeeze(-1)  # (B, 512)
        x = self.primary1(x)  # (B, 512, num_capsules)
        # x = self.digits1(x)  # (B, num_capsules, 512)
        x = self.digits1(x.transpose(1, 2))  # (B, num_capsules, 512)

        # return x.transpose(1,2 )

        x = x.mean(1)  # (B, num_capsules, 512)-> (B, 512)
        x = self.primary2(x)  # (B, 512, num_capsules)
        x = self.digits2(x.transpose(1, 2))  # (B, 2, 512)
        return x


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# x = torch.randn(2, 512)
# m1 = CN()
# m1(x).shape
# -

# # Model

# + editable=true slideshow={"slide_type": ""}
class ABCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = Melspec()
        self.model = VGG18()
        self.attention = Attention_to_1D()
        self.capsule = CN()

        self.cls_head = nn.Linear(512, 1)

    def extract_2D_feature(self, x):
        x = self.mel(x)  # (B, 3, 224, 224)
        x = self.model(x)  # (B, 512, 7, 7)
        return x

    def forward(self, x):
        capsule_output = None

        x = self.extract_2D_feature(x)
        x = self.attention(x)  # (B, 512)

        # logit = self.cls_head(x)

        capsule_output = self.capsule(x)  # (B, 2, 512)
        # logit = capsule_output.mean(-1)
        # logit = (capsule_output**2).sum(dim=-1) ** 0.5

        logit = self.cls_head(capsule_output).squeeze(-1)
        
        return logit, capsule_output


# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# x = torch.randn(2, 1, 48000)
# model = ABCNet()
# model(x)

# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# x = torch.randn(2, 1, 48000).cuda()
# model = ABCNet().cuda()
# model(x)

# + [markdown] editable=true slideshow={"slide_type": ""}
# # Loss

# + editable=true slideshow={"slide_type": ""}
class CapsuleLoss(nn.Module):
    def __init__(self):
        """Constructs a CapsuleLoss module."""
        super(CapsuleLoss, self).__init__()

    def forward(self, x, labels):
        """Defines how the loss compares inputs.
        param x: digit capsule outputs
        param labels:
        param images: the original MNIST image input data
        param reconstructions: reconstructed MNIST image data
        return: weighted margin and reconstruction loss, averaged over a batch
        """
        batch_size = x.size(0)

        ##  calculate the margin loss   ##

        # get magnitude of digit capsule vectors, v_c
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        # calculate "correct" and incorrect loss
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        # sum the losses, with a lambda = 0.5
        # print(labels.shape, left.shape)
        margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        margin_loss = margin_loss.mean()
        return margin_loss


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# loss = CapsuleLoss()
# x = torch.randn(32, 2, 512)
# labels = torch.randint(0, 1, (32, 1))
# loss(x, labels)
# -

# # Lit Model



# + editable=true slideshow={"slide_type": ""}
class ABCNet_lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        self.model = ABCNet()
        self.configure_loss_fn()

        if args is not None and hasattr(args, "profiler"):
            self.profiler = args.profiler
        else:
            self.profiler = None

        self.lr = 1e-4
        self.save_hyperparameters()

    def configure_loss_fn(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.cap_loss = CapsuleLoss()

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        batch_size = len(label)
        ce_loss = self.ce_loss(batch_res["ce_logit"], label.long())
        cls_loss = self.bce_loss(batch_res["logit"], label.type(torch.float32))
        cap_loss = (
            self.cap_loss(batch_res["capsule_output"], label[:, None]) if batch_res["capsule_output"] is not None else 0
        )
        loss = 1.0 * ce_loss + 0 * cls_loss + 1.0 * cap_loss
        # loss = cap_loss
        return {"loss": loss, "ce_loss": ce_loss, "cls_loss": cls_loss, "cap_loss": cap_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.num_training_batches = self.trainer.num_training_batches
        return [optimizer]

    def _shared_pred(self, batch, batch_idx, stage="train"):
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        B = len(audio)
        logit, capsule_output = self.model(audio)
        binary_logit = F.softmax(logit, dim=-1)[:, 1]

        return {"ce_logit": logit, "logit": binary_logit, "capsule_output": capsule_output}

    def _shared_eval_step(self, batch, batch_idx, stage="train", dataloader_idx=0, *args, **kwargs):
        try:
            batch_res = self._shared_pred(batch, batch_idx, stage=stage)
        except TypeError:
            batch_res = self._shared_pred(batch, batch_idx)

        label = batch["label"]
        loss = self.calcuate_loss(batch_res, batch)

        if not isinstance(loss, dict):
            loss = {"loss": loss}

        suffix = "" if dataloader_idx == 0 else f"-dl{dataloader_idx}"
        self.log_dict(
            {f"{stage}-{key}{suffix}": loss[key] for key in loss},
            on_step=True if stage == "train" else False,
            # on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=batch["label"].shape[0],
        )
        batch_res.update(loss)
        return batch_res
