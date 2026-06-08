import gc
import logging
import warnings
from pathlib import Path
from typing import Annotated, Any

import lightning.pytorch as pl
import nemo.collections.asr as nemo_asr
import torch
import typer
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging as nemo_logging
from numba.core.errors import NumbaPerformanceWarning
from omegaconf import DictConfig, open_dict
from torch import nn

from csrc_parakeet.config import (
    AdapterConfig,
    LhotseConfig,
    PartialFreezeConfig,
    TrainConfig,
    load_config,
    save_config,
)

app = typer.Typer()

nemo_logging.setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def load_model(
    model_name_or_path: str,
    adapter_cfg: AdapterConfig | None = None,
) -> nemo_asr.models.ASRModel:
    """HF名 or .nemoパスからモデルをロードする。

    adapter_cfg が有効な場合、encoderクラスをAdapter互換クラスに差し替えてロードする。
    """
    use_adapter = adapter_cfg is not None and adapter_cfg.enabled
    p = Path(model_name_or_path)

    if use_adapter:
        # configを取得してencoder _target_ をAdapter互換クラスに差し替え
        if p.is_file() and p.suffix == ".nemo":
            model_cfg = nemo_asr.models.ASRModel.restore_from(str(p), return_config=True)
        else:
            model_cfg = nemo_asr.models.ASRModel.from_pretrained(model_name_or_path, return_config=True)

        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
        if adapter_metadata is None:
            raise ValueError(f"No adapter registered for: {model_cfg.encoder._target_}")

        with open_dict(model_cfg):
            model_cfg.encoder._target_ = adapter_metadata.adapter_class_path

        if p.is_file() and p.suffix == ".nemo":
            logger.info(f"Restoring model (adapter mode) from: {p}")
            return nemo_asr.models.ASRModel.restore_from(str(p), override_config_path=model_cfg)
        logger.info(f"Loading pretrained model (adapter mode): {model_name_or_path}")
        return nemo_asr.models.ASRModel.from_pretrained(
            model_name_or_path, override_config_path=model_cfg,
        )

    # 通常ロード
    if p.is_file() and p.suffix == ".nemo":
        logger.info(f"Restoring model from: {p}")
        return nemo_asr.models.ASRModel.restore_from(str(p))
    logger.info(f"Loading pretrained model: {model_name_or_path}")
    return nemo_asr.models.ASRModel.from_pretrained(model_name_or_path)


def configure_data(  # noqa: PLR0913
    model: nemo_asr.models.ASRModel,
    train_manifest: Path,
    val_manifest: Path,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    *,
    use_lhotse: bool = True,
    lhotse_cfg: LhotseConfig | None = None,
) -> None:
    """train_ds / validation_ds を設定する。"""
    with open_dict(model.cfg):
        model.cfg.train_ds.manifest_filepath = str(train_manifest)
        model.cfg.train_ds.num_workers = num_workers
        model.cfg.train_ds.is_tarred = False
        model.cfg.train_ds.use_lhotse = use_lhotse
        model.cfg.train_ds.shuffle = True
        model.cfg.train_ds.text_field = "text"

        # モデルのデフォルト(min=1.0, max=10.0)だと大半の音声が除外されるため制限を緩和
        # 注意: float("inf") は Lhotse バケッティングで NaN を引き起こすため有限値を使用
        model.cfg.train_ds.min_duration = 0.0
        model.cfg.train_ds.max_duration = 10000.0

        # min_tps=Noneだと NeMo の TokenPerSecondFilter で TypeError になるため明示指定
        # -1 / inf はフィルタを実質無効化する（0以上にするとtokenizer必須のTPSフィルタが起動する）
        model.cfg.train_ds.min_tps = -1
        model.cfg.train_ds.max_tps = float("inf")

        if use_lhotse and lhotse_cfg is not None:
            model.cfg.train_ds.batch_size = None
            model.cfg.train_ds.batch_duration = lhotse_cfg.batch_duration
            model.cfg.train_ds.quadratic_duration = lhotse_cfg.quadratic_duration
            model.cfg.train_ds.use_bucketing = lhotse_cfg.use_bucketing
            model.cfg.train_ds.num_buckets = lhotse_cfg.num_buckets
            if lhotse_cfg.noise_path is not None:
                model.cfg.train_ds.noise_path = lhotse_cfg.noise_path
                model.cfg.train_ds.noise_snr = list(lhotse_cfg.noise_snr)
                model.cfg.train_ds.noise_mix_prob = lhotse_cfg.noise_mix_prob
        else:
            model.cfg.train_ds.batch_size = batch_size

        model.cfg.validation_ds.manifest_filepath = str(val_manifest)
        model.cfg.validation_ds.batch_size = val_batch_size
        model.cfg.validation_ds.num_workers = num_workers
        model.cfg.validation_ds.is_tarred = False
        model.cfg.validation_ds.use_lhotse = False
        model.cfg.validation_ds.text_field = "text"
        model.cfg.validation_ds.min_duration = 0.0
        model.cfg.validation_ds.max_duration = 10000.0

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)


def _count_manifest_lines(manifest_path: Path) -> int:
    """マニフェスト(JSONL)の行数を数えてサンプル数を返す。"""
    with manifest_path.open() as f:
        return sum(1 for _ in f)


def configure_optimizer(  # noqa: PLR0913
    model: nemo_asr.models.ASRModel,
    lr: float,
    weight_decay: float,
    sched_name: str,
    warmup_steps: int,
    min_lr: float,
    max_steps: int = -1,
) -> None:
    """Optimizer + LR Schedulerを設定する。"""
    with open_dict(model.cfg):
        model.cfg.optim.name = "adamw"
        model.cfg.optim.lr = lr
        model.cfg.optim.weight_decay = weight_decay

        model.cfg.optim.sched.name = sched_name
        model.cfg.optim.sched.warmup_steps = warmup_steps
        model.cfg.optim.sched.min_lr = min_lr
        # Lhotse IterableDataset は len() 未対応のため max_steps を明示指定する
        model.cfg.optim.sched.max_steps = max_steps

    model.setup_optimization(model.cfg.optim)


class TrainLossLogCallback(pl.Callback):
    """train_lossをプログレスバーに表示するコールバック。"""

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # noqa
        outputs: Any,  # noqa
        batch: Any,  # noqa
        batch_idx: int,  # noqa
    ) -> None:
        loss = trainer.callback_metrics.get("train_loss")
        if loss is None:
            return
        loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        bar = getattr(trainer.progress_bar_callback, "main_progress_bar", None)
        if bar is not None:
            bar.set_postfix(loss=f"{loss_val:.6f}", lr=f"{lr:.2e}")


class CudaCacheFlushCallback(pl.Callback):
    """Validation終了後にGPUメモリキャッシュを解放するコールバック。"""

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        gc.collect()
        torch.cuda.empty_cache()


class NemoModelCheckpoint(pl.Callback):
    """val_wer を監視して .nemo 形式でベスト/ラストモデルを保存するコールバック。"""

    def __init__(self, dirpath: Path, save_last: bool = True) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.save_last = save_last
        self.best_wer: float = float("inf")
        self.best_model_path: Path | None = None

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        val_wer = trainer.callback_metrics.get("val_wer")
        if val_wer is None:
            return

        self.dirpath.mkdir(parents=True, exist_ok=True)
        current_wer = val_wer.item() if hasattr(val_wer, "item") else float(val_wer)
        epoch = trainer.current_epoch

        if self.save_last:
            last_path = self.dirpath / "last.nemo"
            pl_module.save_to(str(last_path))  # type: ignore
            logger.info(f"Last model saved: {last_path}")

        if current_wer < self.best_wer:
            if self.best_model_path and self.best_model_path.exists():
                self.best_model_path.unlink()
            self.best_wer = current_wer
            self.best_model_path = self.dirpath / f"best-epoch{epoch:02d}-val_wer{current_wer:.4f}.nemo"
            pl_module.save_to(str(self.best_model_path))  # type: ignore
            logger.info(f"New best model (val_wer={current_wer:.4f}): {self.best_model_path}")


def setup_adapter(model: nemo_asr.models.ASRModel, adapter_cfg: AdapterConfig) -> None:
    """Adapterを追加し、adapter以外のパラメータを凍結する。"""
    module_name = adapter_cfg.adapter_module_name
    adapter_full_name = f"{module_name}:{adapter_cfg.adapter_name}"

    # in_features: encoder/decoderで異なる次元を使用
    # encoder+decoder の場合はencoderの次元を使用（decoderは_update_adapter_cfg_input_dimで自動補正）
    if module_name == "decoder":
        in_features = model.decoder.pred_hidden
    elif module_name == "joint":
        in_features = model.joint.pred_hidden
    else:
        in_features = model.encoder.d_model

    adapter_type_cfg = DictConfig({
        "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
        "in_features": in_features,
        "dim": adapter_cfg.dim,
        "activation": adapter_cfg.activation,
        "norm_position": adapter_cfg.norm_position,
        "dropout": adapter_cfg.dropout,
        "adapter_strategy": {
            "_target_": "nemo.core.classes.mixins.adapter_mixin_strategies.ResidualAddAdapterStrategy",
            "stochastic_depth": adapter_cfg.stochastic_depth,
            "l2_lambda": adapter_cfg.l2_lambda,
        },
    })

    model.add_adapter(adapter_full_name, cfg=adapter_type_cfg)
    model.freeze()
    model.set_enabled_adapters(adapter_full_name, enabled=True)
    model.unfreeze_enabled_adapters()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Adapter: {adapter_full_name} (dim={adapter_cfg.dim})")
    logger.info(f"  Total: {total:,}, Trainable: {trainable:,} ({trainable / total * 100:.2f}%)")


def _unfreeze_batchnorm_in_layers(layers: list[nn.Module], freeze_n: int) -> int:
    """凍結層内のBatchNormモジュールを学習可能にする。解凍した数を返す。"""
    bn_count = 0
    for i, layer in enumerate(layers):
        if i >= freeze_n:
            break
        for module in layer.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train()
                for param in module.parameters():
                    param.requires_grad_(True)
                bn_count += 1
    return bn_count


def apply_partial_encoder_freeze(
    model: nemo_asr.models.ASRModel,
    partial_freeze_cfg: PartialFreezeConfig,
) -> None:
    """エンコーダの下位N層を凍結する（optimizer設定前に呼ぶ）。"""
    encoder = model.encoder
    num_layers = len(encoder.layers)
    freeze_n = partial_freeze_cfg.freeze_n_layers

    if freeze_n is None or freeze_n <= 0:
        return

    # 下位N層を凍結
    for i, layer in enumerate(encoder.layers):
        if i < freeze_n:
            for param in layer.parameters():
                param.requires_grad_(False)

    # pre_encode (conv subsampling) も凍結
    if hasattr(encoder, "pre_encode"):
        for param in encoder.pre_encode.parameters():
            param.requires_grad_(False)

    logger.info(f"Partial freeze: {min(freeze_n, num_layers)}/{num_layers} bottom layers frozen")

    # 凍結層のBatchNormを学習可能にする
    if partial_freeze_cfg.unfreeze_bn:
        bn_count = _unfreeze_batchnorm_in_layers(list(encoder.layers), freeze_n)
        if bn_count > 0:
            logger.info(f"Unfroze {bn_count} BatchNorm modules in frozen layers")


class EncoderUnfreezeCallback(pl.Callback):
    """エンコーダの凍結/解凍を制御するコールバック。

    freeze_encoder_epochs (ステップ制御が無効の場合):
        -1: 永久凍結（解凍しない）
         0: 凍結しない
        >0: Nエポック後に解凍

    freeze_encoder_steps (優先):
        None: エポック制御を使用
        -1: 永久凍結（解凍しない）
         0: 凍結しない
        >0: Nステップ後に解凍
    """

    def __init__(self, freeze_encoder_epochs: int, freeze_encoder_steps: int | None = None) -> None:
        super().__init__()
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.freeze_encoder_steps = freeze_encoder_steps
        self._unfrozen = False

    @property
    def _use_step_control(self) -> bool:
        return self.freeze_encoder_steps is not None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._use_step_control:
            # ステップ制御時はエポック0で凍結のみ行う
            if trainer.current_epoch == 0 and self.freeze_encoder_steps != 0:
                logger.info(f"Freezing encoder (will unfreeze at step {self.freeze_encoder_steps})")
                pl_module.encoder.freeze()  # type: ignore[attr-defined]
            return

        # エポック制御
        if self.freeze_encoder_epochs == -1:
            if trainer.current_epoch == 0:
                logger.info("Freezing encoder permanently")
                pl_module.encoder.freeze()  # type: ignore[attr-defined]
        elif trainer.current_epoch < self.freeze_encoder_epochs:
            if trainer.current_epoch == 0:
                logger.info(f"Freezing encoder until epoch {self.freeze_encoder_epochs}")
                pl_module.encoder.freeze()  # type: ignore[attr-defined]
        elif trainer.current_epoch == self.freeze_encoder_epochs:
            logger.info(f"Unfreezing encoder at epoch {self.freeze_encoder_epochs}")
            pl_module.encoder.unfreeze()  # type: ignore[attr-defined]

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,  # noqa: ARG002, ANN401
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        if not self._use_step_control or self._unfrozen:
            return
        if self.freeze_encoder_steps == -1:
            return  # 永久凍結
        if trainer.global_step >= self.freeze_encoder_steps:  # type: ignore
            logger.info(f"Unfreezing encoder at step {trainer.global_step}")
            pl_module.encoder.unfreeze()  # type: ignore[attr-defined]
            self._unfrozen = True


def run_finetuning(cfg: TrainConfig) -> Path:  # noqa: C901
    """ファインチューニングのフルパイプラインを実行する。"""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # 設定をYAMLダンプ（再現性）
    config_dump_path = cfg.output_dir / "config.yaml"
    save_config(cfg, config_dump_path)
    logger.info(f"Config saved to {config_dump_path}")

    # モデルロード（adapter有効時はadapter対応クラスでロード）
    model = load_model(
        cfg.model_name_or_path,
        adapter_cfg=cfg.adapter if cfg.adapter.enabled else None,
    )

    # Adapterセットアップ
    if cfg.adapter.enabled:
        setup_adapter(model, cfg.adapter)

    # SpecAugment無効化（adapter訓練時など）
    if cfg.disable_spec_augment:
        model.spec_augmentation = None
        logger.info("SpecAugment disabled")

    # NeMo モデルレベル設定（RNNT loss安定化・ログ制御）
    with open_dict(model.cfg):
        model.cfg.skip_nan_grad = cfg.skip_nan_grad
        model.cfg.rnnt_reduction = cfg.rnnt_reduction
        model.cfg.log_prediction = cfg.log_prediction
        model.cfg.compute_eval_loss = cfg.compute_eval_loss

    # データ設定
    val_batch_size = cfg.val_batch_size or cfg.batch_size
    configure_data(
        model,
        cfg.data.train_manifest,
        cfg.data.val_manifest,
        cfg.batch_size,
        val_batch_size,
        cfg.num_workers,
        use_lhotse=cfg.use_lhotse,
        lhotse_cfg=cfg.lhotse if cfg.use_lhotse else None,
    )

    # max_steps の計算（スケジューラ用）
    # Lhotse 使用時は動的バッチングのため実際のバッチ数と一致しないが、
    # CosineAnnealing 等のスケジューラに減衰目標として近似値を渡す。
    num_samples = _count_manifest_lines(cfg.data.train_manifest)
    steps_per_epoch = num_samples // cfg.batch_size // cfg.accumulate_grad_batches
    max_steps = steps_per_epoch * cfg.max_epochs
    logger.info(
        f"max_steps={max_steps} ({num_samples} samples, {steps_per_epoch} steps/epoch)"
        + (" [approximate: Lhotse dynamic batching]" if cfg.use_lhotse else ""),
    )
    if cfg.max_steps is not None:
        logger.info(f"Training will stop early at {cfg.max_steps} steps (scheduler uses {max_steps})")

    # 部分Freeze（adapter無効時のみ、optimizer設定前）
    if not cfg.adapter.enabled and cfg.partial_freeze.freeze_n_layers is not None:
        apply_partial_encoder_freeze(model, cfg.partial_freeze)

    # Optimizer設定
    configure_optimizer(
        model,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        sched_name=cfg.scheduler.name,
        warmup_steps=cfg.scheduler.warmup_steps,
        min_lr=cfg.scheduler.min_lr,
        max_steps=max_steps,
    )

    # Callbacks
    callbacks: list[pl.Callback] = [TrainLossLogCallback(), CudaCacheFlushCallback()]

    # EncoderUnfreezeCallback: adapter modeでは不要（adapter側で凍結管理）
    if not cfg.adapter.enabled:
        if cfg.freeze_encoder_steps is not None:
            if cfg.freeze_encoder_steps != 0:
                callbacks.append(EncoderUnfreezeCallback(cfg.freeze_encoder_epochs, cfg.freeze_encoder_steps))
        elif cfg.freeze_encoder_epochs != 0:
            callbacks.append(EncoderUnfreezeCallback(cfg.freeze_encoder_epochs))

    nemo_checkpoint = NemoModelCheckpoint(
        dirpath=cfg.output_dir / "checkpoints",
        save_last=True,
    )
    callbacks.append(nemo_checkpoint)

    # Logger
    trainer_logger = None
    if cfg.wandb.enabled:
        trainer_logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name or cfg.output_dir.name,
            save_dir=str(cfg.output_dir),
        )

    # Trainer
    trainer_kwargs: dict[str, Any] = {
        "devices": 1,
        "accelerator": "auto",
        "max_epochs": cfg.max_epochs if cfg.max_steps is None else -1,
        "max_steps": cfg.max_steps if cfg.max_steps is not None else -1,
        "precision": cfg.precision,
        "accumulate_grad_batches": cfg.accumulate_grad_batches,
        "gradient_clip_val": cfg.gradient_clip_val,
        "callbacks": callbacks,
        "default_root_dir": str(cfg.output_dir),
        "enable_progress_bar": True,
        "log_every_n_steps": 10,
        "logger": trainer_logger,
        "use_distributed_sampler": False,  # Lhotse は独自の分散サンプラーを使用
    }
    if cfg.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = cfg.limit_train_batches
    if cfg.val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = cfg.val_check_interval

    trainer = pl.Trainer(**trainer_kwargs)

    # 学習実行
    trainer.fit(model)

    # 最終モデルを保存
    nemo_path = cfg.output_dir / "finetuned_model.nemo"
    model.save_to(str(nemo_path))
    logger.info(f"Model saved to {nemo_path}")
    return nemo_path


@app.command()
def main(
    config: Annotated[Path, typer.Argument(help="YAML設定ファイルパス")],
) -> None:
    """Parakeetモデルのファインチューニングを実行する。

    設定はすべてYAMLファイルで指定する。
    サンプル: csrc_parakeet/configs/finetune_sample.yaml
    """
    cfg = load_config(config)
    run_finetuning(cfg)


if __name__ == "__main__":
    app()
