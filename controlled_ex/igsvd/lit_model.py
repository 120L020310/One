import os
import yaml
import torch
import torch.optim as optim
from typing import Optional, List, Dict, Any

from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification

try:
    from .org_code.training.detectors import AudioFakeDetector
    from .org_code.training.trainer.sam import SAM
except ImportError:
    from org_code.training.detectors import AudioFakeDetector
    from org_code.training.trainer.sam import SAM


current_dir = os.path.dirname(os.path.abspath(__file__))

yaml_path = os.path.join(current_dir, "org_code/training/config/detector/audiofakedetection_audio.yaml")



def load_config(yaml_path):

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config['model_name'] = 'AudioFakeDetector'
    config['cuda'] = False


    return config

### see org_code/training/train.py#L98
def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, 
                            rho=config['sam_param']['rho'], adaptive=config['sam_param']['use_adaptive_sam'], 
                            lr=float(config['optimizer'][opt_name]['lr']), 
                            momentum=config['optimizer'][opt_name]['momentum'], 
                            weight_decay=config['optimizer'][opt_name]['weight_decay'])
    elif opt_name == 'adam':
        base_optimizer = torch.optim.Adam
        optimizer = SAM(model.parameters(), base_optimizer, 
                            rho=config['sam_param']['rho'], adaptive=config['sam_param']['use_adaptive_sam'], 
                            lr=float(config['optimizer'][opt_name]['lr']), 
                            weight_decay=config['optimizer'][opt_name]['weight_decay'],
                            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
                            eps=config['optimizer'][opt_name]['eps'],
                            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
    return optimizer
    
### see org_code/training/train.py#L122
def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config['lr_step'], 
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['lr_T_max'], 
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))





class IG_SVD_lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        
        
        self.config = load_config(yaml_path)
        
        try:
            vocoder_num_class = cfg.method_classes
            print(f"!!!!!!vocoder_num_class: {vocoder_num_class}, must same as the dataset, otherwise, error will occur!!!!!!")
        except:
            vocoder_num_class = 50
            print(f"!!!!!! No method_classes in the model cfg, Using default vocoder_num_class: {vocoder_num_class}, must same as the dataset, otherwise, error will occur!!!!!!")
        
        self.model = AudioFakeDetector(config=self.config, vocoder_num_class=vocoder_num_class)
        for para in self.model.parameters():
            para = para.to(torch.device('cpu'))
        self.model = self.model.to("cpu")


        # self.manual_backward = True
        self.automatic_optimization = False

        self.save_hyperparameters()

    
    def forward(self, x, stage='train'):
        inference = 1 if stage != 'train' else 0
        
        
        ### see org_code/training/detectors/audio_detector.py#L419
        pred_dict = self.model(x, inference=inference)
        return pred_dict
    
    
    def configure_optimizers(self):
        
            # 检查模型参数是否需要梯度
        params_with_grad = [p for p in self.model.parameters() if p.requires_grad]
        print(f"参数总数: {len(list(self.model.parameters()))}")
        print(f"需要梯度的参数数: {len(params_with_grad)}")
    
    
        optimizer = choose_optimizer(self.model, self.config)
        scheduler = choose_scheduler(self.config, optimizer)
        if scheduler is not None:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        else:
            return [optimizer]

    
    
    def proprecess_batch(self, batch):
        """
        预处理batch数据
        """
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        if len(audio.shape) == 3:
            batch["audio"] = batch["audio"][:, 0, :]
        
        # print(audio.shape)
        #### note, IG_SVD use 0 for real audio and 1 for fake audio, thus we need to flip the label
        batch['label'] = 1 - batch["label"]
        batch['label_spe'] = batch['vocoder_label']
        
        
        
    def _shared_pred(self, batch, batch_idx, stage='train', proprecess_batch=True, **kwargs):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        if proprecess_batch:
            self.proprecess_batch(batch)


        pred_dict = self.forward(batch,  stage=stage)
        
        batch_res = pred_dict
        batch_res['logit'] = pred_dict['cls'][:, 1] - pred_dict['cls'][:, 0]
        # batch_res['logit'] = pred_dict['cls'][:, 1]
    
        return batch_res
    
    
    def calcuate_loss(self, batch_res, batch, stage='train'):
        inference = True if stage != 'train' else False
        # print(batch_res, batch)
        losses = self.model.get_losses(batch, batch_res, inference=inference)
        if 'overall' in losses:
            losses['loss'] = losses['overall']
        else:
            losses['loss'] = losses['common']

        return losses
    
    
    def training_step(self, batch, batch_idx):
        """
        元学习训练步骤
        """

        # print(self.trainer.callbacks)

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        
        enable_running_stats(self.model)
        
        batch_res = self._shared_pred(batch, batch_idx, stage='train')
        losses = self.calcuate_loss(batch_res, batch, stage='train')
        # losses['overall'].backward()
        self.manual_backward(losses['overall'])
        
        optimizer.first_step(zero_grad=True)
        if scheduler is not None:
            scheduler.step()
        
        disable_running_stats(self.model)
        batch_res = self._shared_pred(batch, batch_idx, stage='train', proprecess_batch=False)
        losses = self.calcuate_loss(batch_res, batch, stage='train')
        # losses['overall'].backward()
        self.manual_backward(losses['overall'])
        optimizer.second_step(zero_grad=True)
        if scheduler is not None:
            scheduler.step()
        
        
        ##### 只有这个才能出发trainer.global_step += 1，进而才能使用modelcheckpoint保存模型
        ##### 所以，在optimizer那里，把step()改成了空操作
        optimizer.step()
        
        # self.trainer.fit_loop.epoch_loop.global_step += 1
                
        batch_res.update(losses)
        return batch_res
    
    # def on_train_start(self) -> None:
    #     print(self.trainer.logger)
    #     print(self.trainer.callbacks)
        
        
    # def on_validation_epoch_end(self, *args, **kwargs):
    #     """
    #     在验证周期结束后被调用
    #     """
    #     print("--- 进入调试模式 ---")
    #     print("当前所有可用的回调指标:")
    #     print(self.trainer.callback_metrics)
        
    # def configure_custom_callbacks(self, current_callbacks: Optional[List] = None) -> List:
    #     from pytorch_lightning.callbacks import ModelCheckpoint
        
    #     for i, cb in enumerate(current_callbacks):
    #         if isinstance(cb, ModelCheckpoint):
    #             print(f"移除现有的ModelCheckpoint回调: {cb}")
    #             current_callbacks.pop(i)

        
    #     # checkpoint_callback_last = ModelCheckpoint(
    #     #     dirpath=None,  # 使用默认目录
    #     #     save_top_k=0,  # 不保存最佳模型
    #     #     save_last=True,  # 保存最后一个模型
    #     #     save_weights_only=False,  # 保存整个模型，而不仅仅是权重
    #     # )
    #     checkpoint_callback_best = ModelCheckpoint(
    #         dirpath=None,  # 使用默认目录
    #         monitor='val-auc',  # 监控验证集上的AUC指标
    #         mode='max',  # AUC指标越大越好
    #         save_top_k=1,  # 只保存最佳模型
    #         save_last=True,  # 不保存最后一个模型
    #         save_weights_only=False,  # 保存整个模型，而不仅仅是权重
    #         save_on_train_epoch_end=False,  # 在验证周期结束时保存模型
    #     )
    #     return [checkpoint_callback_best]
        
    #     checkpoint_callback = [checkpoint_callback_last, checkpoint_callback_best]
        
    #     return checkpoint_callback

    
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
