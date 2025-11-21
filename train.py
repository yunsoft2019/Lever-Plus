import json
import os
from functools import partial

import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from torch import optim
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from lever_lm.utils import data_split, collate_fn
from utils import load_ds


# 简洁表格输出回调
class SimpleTableLogger(Callback):
    def __init__(self):
        self.metrics_history = []
        self.header_printed = False
        self.val_count = 0
        
    def on_validation_end(self, trainer, pl_module):
        # 获取指标（每次验证后都打印）
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get('train_loss', float('nan'))
        val_loss = trainer.callback_metrics.get('val_loss', float('nan'))
        
        # 跳过 sanity check（训练前的验证，train_loss 为 nan）
        if trainer.sanity_checking:
            return
        
        # 打印表头（只打印一次）
        if not self.header_printed:
            print("\n" + "="*50)
            print(f"{'Epoch':>6} {'Step':>6} {'Train Loss':>12} {'Val Loss':>12}")
            print("="*50)
            self.header_printed = True
        
        # 计算当前步数（基于验证次数）
        self.val_count += 1
        step_indicator = f"{self.val_count % 4 if self.val_count % 4 != 0 else 4}/4"
        
        # 打印数据行（只在有有效 train_loss 时打印）
        if not float('nan') == train_loss and not float('inf') == train_loss:
            print(f"{epoch:6d} {step_indicator:>6} {train_loss:12.6f} {val_loss:12.6f}")
        
        # 保存历史记录
        self.metrics_history.append({
            'epoch': epoch,
            'step': self.val_count,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss)
        })
    
    def on_train_end(self, trainer, pl_module):
        import math
        print("="*50)
        print("训练完成！")
        # 过滤出有效的验证损失值（排除 NaN 和 Inf）
        valid_val_losses = [
            m['val_loss'] for m in self.metrics_history 
            if not (math.isnan(m['val_loss']) or math.isinf(m['val_loss']))
        ]
        if valid_val_losses:
            print(f"最佳 Val Loss: {min(valid_val_losses):.6f}")
        else:
            print("警告: 没有有效的验证损失记录（可能是验证数据为空）")
        print("="*50)


# define the LightningModule
class LeverLM(pl.LightningModule):
    def __init__(self, lever_lm, lr, weight_decay=1e-2, warm_steps=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["lever_lm"])
        self.lever_lm = lever_lm

    def training_step(self, batch, batch_idx):
        output = self.lever_lm(**batch)
        loss = output["loss"]
        self.log(
            "train_loss", loss, batch_size=len(batch["icd_seq_idx"]), sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.lever_lm(**batch)
        loss = output["loss"]
        self.log("val_loss", loss, batch_size=len(batch["icd_seq_idx"]), sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.lever_lm.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        step_batches = self.trainer.estimated_stepping_batches
        if isinstance(self.hparams.warm_steps, float):
            warm_steps = self.hparams.warm_steps * step_batches
        elif isinstance(self.hparams.warm_steps, int):
            warm_steps = self.hparams.warm_steps
        else:
            raise ValueError(
                f"the warm_steps should be int or float, but got {type(self.hparams.warm_steps)}"
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_steps, num_training_steps=step_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class ICDSeqDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
    ):
        """
        dataset_para: The dataset parameters
        dataset: The *.py file name of the dataset class
        dataset_name: The dataset Class name
        """
        super().__init__()
        # 数据文件路径应该包含数据集子目录
        # 从 cfg.dirpath 或 data_files 中提取数据集名称
        dataset_name = cfg.get('dataset', {}).get('name', 'okvqa')
        data_files_path = os.path.join(cfg.result_dir, dataset_name, "generated_data", cfg.data_files)
        with open(data_files_path, "r") as f:
            data = json.load(f)
        self.train_data_list, self.val_data_list = data_split(data, cfg.train_ratio)
        self.ds_factory = hydra.utils.instantiate(cfg.train.lever_lm_ds, _partial_=True)
        self.index_ds = load_ds(cfg, "train")
        self.processor = CLIPProcessor.from_pretrained(cfg.train.lever_lm.clip_name)

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.trainset = self.ds_factory(
                data_list=self.train_data_list, index_ds=self.index_ds
            )
            self.valset = self.ds_factory(
                data_list=self.val_data_list, index_ds=self.index_ds
            )
            # 检查验证数据集是否为空
            if len(self.valset) == 0:
                val_icd_seq_len = len(self.val_data_list.get('icd_seq', []))
                val_icd_score_len = len(self.val_data_list.get('icd_score', []))
                val_scores = self.val_data_list.get('icd_score', [])
                if val_scores:
                    min_score = min(val_scores)
                    max_score = max(val_scores)
                    print(f"警告: 验证数据集为空 (len={len(self.valset)})")
                    print(f"  验证数据列表长度: icd_seq={val_icd_seq_len}, icd_score={val_icd_score_len}")
                    print(f"  验证数据分数范围: min={min_score:.6f}, max={max_score:.6f}")
                    print(f"  可能原因: 验证数据格式不正确或数据集创建时出错")
                else:
                    print(f"警告: 验证数据集为空，且验证数据列表也为空")

    def train_dataloader(self):
        global collate_fn
        return DataLoader(
            self.trainset,
            batch_size=self.hparams.cfg.batch_size,
            num_workers=self.hparams.cfg.num_workers,
            shuffle=True,
            collate_fn=partial(collate_fn, processor=self.processor),
            pin_memory=True,
        )

    def val_dataloader(self):
        global collate_fn
        return DataLoader(
            self.valset,
            batch_size=self.hparams.cfg.batch_size,
            num_workers=self.hparams.cfg.num_workers,
            collate_fn=partial(collate_fn, processor=self.processor),
            shuffle=False,
        )


@hydra.main(version_base=None, config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    # 不使用 wandb，禁用 logger
    use_simple_logger = cfg.get('use_simple_logger', False)
    logger = False  # 禁用 logger
    
    # 只保存 val_loss 最优的模型（不保存 train_loss 最优的）
    # 使用自定义文件名前缀（如果提供）或默认格式
    checkpoint_filename_prefix = cfg.get('checkpoint_filename', 'best_model')
    
    vl_model_cpk_callback = ModelCheckpoint(
        filename=f"{checkpoint_filename_prefix}_epoch={{epoch}}_train={{train_loss:.5f}}_val={{val_loss:.5f}}",
        monitor="val_loss",
        save_last=False,  # 不保存最后一个 epoch（通常不是最优的）
        save_top_k=1,     # 只保留最优的 1 个
        mode="min",
        dirpath=cfg.dirpath,
        auto_insert_metric_name=False,
    )
    
    # 准备回调列表
    callbacks = [
        vl_model_cpk_callback,  # 只保存 val_loss 最优的模型
    ]
    
    # 根据配置选择进度条和日志方式
    if use_simple_logger:
        callbacks.append(SimpleTableLogger())
    else:
        callbacks.extend([
            LearningRateMonitor(),
            RichModelSummary(max_depth=2),
            RichProgressBar(),
        ])
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=(not use_simple_logger),
        **cfg.trainer_args,
    )
    lever_lm = hydra.utils.instantiate(cfg.train.lever_lm)
    model = LeverLM(lever_lm, cfg.lr, cfg.weight_decay, cfg.warm_steps)
    data_module = ICDSeqDataModule(cfg)
    
    # 支持从检查点恢复训练
    # 优先从环境变量读取（避免 Hydra 解析路径中的特殊字符）
    ckpt_path = os.environ.get('RESUME_CKPT_PATH', None)
    if not ckpt_path:
        ckpt_path = cfg.get('ckpt_path', None)
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"从检查点恢复训练: {ckpt_path}")
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, data_module)


if __name__ == "__main__":
    load_dotenv()
    main()
