from myutils.torch.lightning.callbacks import (
    Color_progress_bar,
    EER_Callback,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from myutils.torch.lightning.callbacks.metrics import (
    BinaryACC_Callback,
    BinaryAUC_Callback,
)
from myutils.torch.lightning.callbacks import Collect_Callback


def common_callbacks():
    callbacks = [
        Color_progress_bar(),
        BinaryACC_Callback(batch_key="label", output_key="logit"),
        BinaryAUC_Callback(batch_key="label", output_key="logit"),
        EER_Callback(batch_key="label", output_key="logit"),
        # BinaryACC_Callback(batch_key="label", output_key="logit1D",theme="1D"),
        # BinaryAUC_Callback(batch_key="label", output_key="logit1D",theme="1D"),
        # EER_Callback(batch_key="label", output_key="logit1D",theme="1D"),
        # BinaryACC_Callback(batch_key="label", output_key="logit2D",theme="2D"),
        # BinaryAUC_Callback(batch_key="label", output_key="logit2D",theme="2D"),
        # EER_Callback(batch_key="label", output_key="logit2D",theme="2D"),
        # BinaryACC_Callback(batch_key="label", output_key="logitASR",theme="ASR"),
        # BinaryAUC_Callback(batch_key="label", output_key="logitASR",theme="ASR"),
        # EER_Callback(batch_key="label", output_key="logitASR",theme="ASR"),
    ]

    return callbacks


def custom_callbacks(args, cfg):
    callbacks = []
    return callbacks

from pytorch_lightning.callbacks import ModelCheckpoint

class UniqueCheckpoint(ModelCheckpoint):
    """
    为了解决 'Found more than one stateful callback' 错误，
    我们需要一个自定义的 state_key。
    """
    def __init__(self, state_key_id, **kwargs):
        super().__init__(**kwargs)
        self.state_key_id = state_key_id

    @property
    def state_key(self):
        return f"ModelCheckpoint_{self.state_key_id}"
def training_callbacks(args):

    monitor = "val-eer"
    es = EarlyStopping

    callbacks = [
        # save last ckpt
        ModelCheckpoint(
            dirpath=None, save_top_k=0, save_last=True, save_weights_only=False
        ),
        # save best ckpt
        # UniqueCheckpoint(
        #     state_key_id="save_all_epochs", # 【关键】唯一的 ID
        #     dirpath=None,
        #     save_top_k=-1,           # 无限保留
        #     every_n_epochs=1,        # 每个 epoch 保存
        #     monitor=None,            # 不监控
        #     filename="{epoch}-{val-eer:.4f}",# 文件名 epoch_0.ckpt
        #     save_weights_only=True,
        #     verbose=True,
        # ),
        ModelCheckpoint(
            dirpath=None,
            save_top_k=1,
            monitor=monitor,
            mode="min",
            save_last=True,
            filename="best-{epoch}-{val-eer:.4f}",
            save_weights_only=True,
            verbose=True,
        ),
        # --- 2. 辅助 Checkpoint (负责保存每个 Epoch) ---
        # 使用自定义的 UniqueCheckpoint 类，避免冲突
    ]

    if args.earlystop:
        callbacks.append(
            # es(
            #     monitor=monitor,
            #     min_delta=0.001,
            #     patience=args.earlystop if args.earlystop > 1 else 3,
            #     mode="max",
            #     stopping_threshold=0.998 if monitor == "val-auc+++val-acc" else 0.999,
            #     verbose=True,
            # )
            es(
                monitor=monitor,
                min_delta=0.001,
                patience=args.earlystop if args.earlystop > 1 else 3,
                mode="min",
                stopping_threshold=0.05,
                verbose=True,
            )
        )
    return callbacks


def make_collect_callbacks(args, cfg):
    name = args.cfg.replace("/", "-")
    callbacks = [
        Collect_Callback(
            batch_keys=["label", "vocoder_label"],
            output_keys=["feature"],
            save_path=f"./0-实验结果/npz/{name}",
        )
    ]
    return callbacks


def make_callbacks(args, cfg):
    callbacks = common_callbacks()
    callbacks += custom_callbacks(args, cfg)
    if not args.test:
        callbacks += training_callbacks(args)

    if args.collect and args.test:
        callbacks += make_collect_callbacks(args, cfg)

    return callbacks
