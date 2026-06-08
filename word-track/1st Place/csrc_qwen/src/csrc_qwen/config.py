from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LoraConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_qlora: bool = False
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


class SpecAugmentConfig(BaseModel):
    freq_masks: int = 0
    freq_width: int = 15
    time_masks: int = 0
    time_width: float = 0.05
    prob: dict[str, float] = {}


class SpeedPerturbConfig(BaseModel):
    rates: list[float] = [0.9, 1.1]
    prob: dict[str, float] = {}


class AddNoiseConfig(BaseModel):
    noise_dir: Path
    min_snr_db: float = 5.0
    max_snr_db: float = 20.0
    prob: dict[str, float] = {}


class AugmentConfig(BaseModel):
    spec_augment: SpecAugmentConfig | None = None
    speed_perturb: SpeedPerturbConfig | None = None
    add_noise: AddNoiseConfig | None = None


class TrainSource(BaseModel):
    csv: Path
    audio_dir: Path
    source_name: str = "unknown"
    cer_threshold: dict[str, float | None] | None = None
    wer_threshold: dict[str, float | None] | None = None
    filter_mode: str = "and"


class DataConfig(BaseModel):
    train_sources: list[TrainSource]
    train_manifest: Path
    val_manifest: Path
    max_duration_sec: float = 30.0


class OptimConfig(BaseModel):
    lr: float = 2e-5
    weight_decay: float = 0.01
    optim_name: str = "adamw_torch"


class SchedulerConfig(BaseModel):
    name: str = "linear"
    warmup_ratio: float = Field(default=0.02, ge=0.0, le=1.0)


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "csrc"
    name: str | None = None


class TrainConfig(BaseModel):
    model_name_or_path: str = "input/Qwen3-ASR-0.6B"
    output_dir: Path = Path("output/qwen_finetune")
    train_audio_encoder: bool = False
    train_projector: bool = False
    prompt: str = ""

    max_epochs: int = 1
    per_device_train_batch_size: int = 4
    eval_batch_size: int | None = None
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    precision: str = "bf16"

    eval_steps: int = 5000
    save_steps: int = 5000
    save_total_limit: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    dataloader_num_workers: int = 4

    lora: LoraConfig = LoraConfig()
    data: DataConfig
    optim: OptimConfig = OptimConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    augment: AugmentConfig = AugmentConfig()
    wandb: WandbConfig = WandbConfig()


def load_config(config_path: Path) -> TrainConfig:
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


def save_config(cfg: TrainConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(cfg.model_dump(mode="json"), f, default_flow_style=False, allow_unicode=True)
