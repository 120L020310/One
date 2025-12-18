

原始模型代码为：https://github.com/QiShanZhang/SLSforASVspoof-2021-DF/blob/main/model.py


## 修改1

原始模型代码使用旧版本的fairseq来加载xlsr2-300m，容易和现有package起冲突。
因此，使用`transformers` package来加载模型：

```python
    def __init__(self, pretrained_path="facebook/wav2vec2-xls-r-300m", audio_length=48000):
        super(XLS_R_SLS, self).__init__()

                
        # model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([pretrained_path])
        # self.ssl_model = model[0]
        self.ssl_model = AutoModelForPreTraining.from_pretrained(pretrained_path)
```
你可以指定"facebook/wav2vec2-xls-r-300m"下载的位置，并传入到`pretrained_path`。




## 修改2
```python
x = self.first_bn(fullfeature)
x = self.selu(x)
x = F.max_pool2d(x, (3, 3))
x = torch.flatten(x, 1)
x = self.fc1(x)
```
原始模型使用maxpool处理xls-r的hidden-states,形状大小为（batch, 1, time, channels），然后拉伸为1维向量，并使用fc1映射到1024维度。
如果time不一样，那么fc1的形状也不一样。因此，使用DynamicSliceLinear，动态处理输入，最大支持 64000 长度的音频输入：
```python
    def __init__(self, pretrained_path="facebook/wav2vec2-xls-r-300m", audio_length=48000):

        ...

        self.fc1 = DynamicSliceLinear(1024, 22847) ## 最大支持 64000 长度的音频输入
```

