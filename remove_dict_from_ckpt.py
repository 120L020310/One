import torch

# 1. 加载现有的 checkpoint 文件
ckpt_path = "/home/zyz/data/dim=128_test/data_aug_SafeRawAugmentor(noise_intensity=0.1, mask_ratio=0.1)_5:1/1227_baseline_detach_noisy/MultiView/ASV2021_LA/version_0/checkpoints/epoch=7-val-auc=0.9602-val-eer=0.0890.ckpt"  # 你的原始文件路径
save_path = "/home/zyz/data/dim=128_test/data_aug_SafeRawAugmentor(noise_intensity=0.1, mask_ratio=0.1)_5:1/1227_baseline_detach_noisy/MultiView/ASV2021_LA/version_0/checkpoints/epoch=7-val-auc=0.9602-val-eer=0.0890_no_post_std.ckpt" # 处理后的保存路径

checkpoint = torch.load(ckpt_path, map_location="cpu")

# 2. 确定 state_dict 的位置
# 提示：如果是 PyTorch Lightning 生成的，通常在 checkpoint['state_dict'] 里
# 如果是普通 PyTorch 保存的，可能直接就是字典，或者在 checkpoint['model'] 里
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# 3. 定义要删除的键名
keys_to_remove = ["post_std.n", "post_std.mean", "post_std.M2"]

# 4. 执行删除
for key in keys_to_remove:
    if key in state_dict:
        del state_dict[key]
        print(f"成功移除键: {key}")
    else:
        print(f"未找到键: {key}，跳过")

# 5. 写回并保存
# 注意：如果原来有其他信息（如 epoch, optimizer），也要保留
if "state_dict" in checkpoint:
    checkpoint["state_dict"] = state_dict

torch.save(checkpoint, save_path)
print(f"清理后的模型已保存至: {save_path}")