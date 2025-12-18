import timm

model = timm.create_model('davit_base.msft_in1k', pretrained=True, 
                          pretrained_cfg_overlay=dict(file='/home/zyz/.cache/huggingface/hub/davit_base/pytorch_model.bin'))
print(model)