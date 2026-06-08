from pathlib import Path

import yaml
from pydantic import BaseModel


class IpaNormalizeConfig(BaseModel):
    enabled: bool = False
    reverse_dict_path: Path | None = None


class AdapterConfig(BaseModel):
    enabled: bool = False
    adapter_name: str = "child_speech"
    adapter_module_name: str = "encoder"
    dim: int = 32
    activation: str = "swish"
    norm_position: str = "pre"
    dropout: float = 0.0
    stochastic_depth: float = 0.0
    l2_lambda: float = 0.0


class PartialFreezeConfig(BaseModel):
    freeze_n_layers: int | None = None  # 下位N層を凍結 (None=無効)
    unfreeze_bn: bool = True  # 凍結層のBatchNormは学習可能にする



class TrainSource(BaseModel):
    csv: Path
    audio_dir: Path
    cer_threshold: dict[str, float | None] | None = None
    wer_threshold: dict[str, float | None] | None = None
    max_text_len: int | None = None  # テキスト文字数上限（超過サンプルを除外）
    max_duration_sec: float | None = None  # 音声長上限（超過サンプルを除外）


class DataConfig(BaseModel):
    train_sources: list[TrainSource]
    train_manifest: Path
    val_manifest: Path


class OptimConfig(BaseModel):
    lr: float = 1e-5
    weight_decay: float = 1e-3


class SchedulerConfig(BaseModel):
    name: str = "CosineAnnealing"
    warmup_steps: int = 100
    min_lr: float = 5e-6


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "csrc-parakeet"
    name: str | None = None


class LhotseConfig(BaseModel):
    batch_duration: float = 80.0
    quadratic_duration: float | None = 15.0
    use_bucketing: bool = True
    num_buckets: int = 30
    # ノイズ拡張（Lhotseネイティブ機能）
    noise_path: str | None = None
    noise_snr: tuple[float, float] = (10.0, 20.0)
    noise_mix_prob: float = 0.5


class TrainConfig(BaseModel):
    model_name_or_path: str = "nvidia/parakeet-tdt-0.6b-v3"
    output_dir: Path = Path("output/finetune")
    max_epochs: int = 5
    max_steps: int | None = None  # 指定時はステップ数で学習を打ち切る（max_epochsより優先）
    val_check_interval: int | None = None  # 指定時はNステップごとにvalidation実行
    batch_size: int = 4
    val_batch_size: int | None = None  # 未指定時はbatch_sizeを使用
    precision: str = "bf16-mixed"
    freeze_encoder_epochs: int = -1
    freeze_encoder_steps: int | None = None  # 指定時はステップ数でエンコーダ解凍（epochs より優先）
    gradient_clip_val: float = 1.0
    num_workers: int = 4
    accumulate_grad_batches: int = 1
    limit_train_batches: int | None = None  # 訓練バッチ数を制限（Lhotse使用時はintのみ）
    use_lhotse: bool = True

    # NeMo モデルレベル設定
    skip_nan_grad: bool = True  # NaN勾配をスキップして学習崩壊を防止
    rnnt_reduction: str = "mean_volume"  # RNNT loss の正規化方式
    log_prediction: bool = True  # サンプル予測をログ出力
    compute_eval_loss: bool = False  # 長い評価サンプルでのOOM防止
    disable_spec_augment: bool = False  # SpecAugment無効化（adapter訓練時に推奨）

    adapter: AdapterConfig = AdapterConfig()
    partial_freeze: PartialFreezeConfig = PartialFreezeConfig()

    data: DataConfig
    optim: OptimConfig = OptimConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    wandb: WandbConfig = WandbConfig()
    lhotse: LhotseConfig = LhotseConfig()
    ipa_normalize: IpaNormalizeConfig = IpaNormalizeConfig()


def load_config(config_path: Path) -> TrainConfig:
    """YAMLファイルからTrainConfigを読み込む。"""
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


def save_config(cfg: TrainConfig, path: Path) -> None:
    """TrainConfigをYAMLファイルに保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(cfg.model_dump(mode="json"), f, default_flow_style=False, allow_unicode=True)
