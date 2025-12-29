# MVCL-ADD
Code for "Multi-View Collaborative Learning Network for Speech Deepfake Detection"

This repository contains the code for the paper titled "Multi-View Collaborative Learning Network for Speech Deepfake Detection" published in AAAI 2025.

## Abstract

As deep learning techniques advance rapidly, deepfake speech synthesized through text-to-speech or voice conversion networks is becoming increasingly realistic, posing significant challenges for detection and raising potential threats to social security. This growing realism has prompted extensive research in speech deepfake detection. However, current detection methods primarily focus on extracting features from either the raw waveform or the spectrogram, often overlooking the valuable correspondences between these two modalities that could enhance the detection of previously unseen types of deepfakes. In this work, we propose a multi-view collaborative learning network for speech deepfake detection, which jointly learns robust speech representations from both raw waveforms and spectrograms. 
Specifically, we first design a \textbf{D}ual-\textbf{B}ranch \textbf{C}ontrastive \textbf{L}earning (DBCL) framework for learning different view features. DBCL consists of two branches that learn representations from the raw waveform or the spectrogram and utilizes contrastive learning to enhance inter- and inner-view correlations. Additionally, we introduce a \textbf{W}aveform-\textbf{S}pectrogram \textbf{F}usion \textbf{M}odule (WSFM) to exchange multi-view information for collaborative learning. In the feature learning process, WSFM converts features between views and merges them adaptively using waveform-spectrogram cross-attention. The final detection is conducted based on the concatenation of the waveform and spectrogram features. We conduct extensive experiments on four benchmark deepfake speech detection datasets, and the experimental results demonstrate that our method can achieve better detection performance than current state-of-the-art detection methods.

## Requirements

```bash
pip install -r requirements.txt
```
Actually, the package versions are not strict. Maybe the latest versions of torch and pytorch_lightning can still work.


## Usage


One can run the following commands to train or test our multiview model.
```bash


python train_controlled_XLSR_oneclass.py --gpu 0 --cfg 'MultiView/ASV2021_LA'  -v 0;\
stage1：得到一个最佳ckpt：oneclass.ckpt,将该ckpt作为下一步的初始权重，同时确保加载了centroid参数。
python train_controlled_XLSR_oneclass_data_aug_detach_noisy.py --gpu 0 --cfg 'MultiView/ASV2021_LA'  -v 0;\
stage2：得到一个最佳ckpt：data_aug_bestmodel.ckpt,作为knn_bank.py中的模型来提取特征memorybank（该文件夹中已保存至"memory_bank_xlsr_best_0890.pt"，直接运行。
ps:所有代码已经适配5090计算方式，但是3090环境可能无法正常运行
```

## Acknowledgements

Please feel free to contact me (zkyhitsz@gmail.com) if you have any questions.

