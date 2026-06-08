"""Fine-tune a word track ASR model. Model type is determined by config."""
import dirtygit
import pytorch_lightning as pl
from loguru import logger
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

from src.config import CONFIG_DIR, load_config, setup_logging
from src.data.dataset import load_and_split
from src.data.sampler import DurationBucketSampler
from src.data.utils import load_jsonl
from src.models.word.factory import create_model, create_dataset, get_collate_fn
from src.paths import DATA_DIR, MODELS_DIR


DEFAULT_CONFIG = CONFIG_DIR / "training" / "word" / "qwen_distill_augment.yaml"


def make_loader(ds, cfg, collate_fn, train=True):
    sampler_type = cfg.training.get("sampler", "default")

    if sampler_type == "duration_bucket" and train:
        max_batch_duration = cfg.training.max_batch_duration
        durations = [e["audio_duration_sec"] for e in ds.entries]
        sampler = DurationBucketSampler(
            durations, max_batch_duration=max_batch_duration,
            shuffle=True, seed=cfg.training.seed,
        )
        logger.info(f"{'Train' if train else 'Val'}: duration-bucketed batching "
                     f"(max_batch_duration={max_batch_duration}, {len(sampler)} batches)")
        return DataLoader(
            ds, batch_sampler=sampler,
            num_workers=cfg.training.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    return DataLoader(
        ds, batch_size=cfg.training.batch_size, shuffle=train,
        num_workers=cfg.training.num_workers, collate_fn=collate_fn, pin_memory=True,
    )


def main(cfg):
    logger.info(f"githash={cfg.githash}")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg)}")

    pl.seed_everything(cfg.training.seed)

    model = create_model(cfg)
    train_entries, val_entries = load_and_split()

    extra_manifest = cfg.get("data", {}).get("extra_manifest", None)
    if extra_manifest:
        manifests = [extra_manifest] if isinstance(extra_manifest, str) else list(extra_manifest)
        for m in manifests:
            extra = load_jsonl(DATA_DIR / m)
            for e in extra:
                e["audio_path"] = str(DATA_DIR / e["audio_path"])
            logger.info(f"Extra data: {len(extra)} entries from {m}")
            train_entries.extend(extra)

    dur_filter = cfg.get("data", {}).get("duration_filter", None)
    if dur_filter:
        min_dur = dur_filter.get("min", 0)
        max_dur = dur_filter.get("max", float("inf"))
        before = len(train_entries)
        train_entries = [e for e in train_entries if min_dur <= e["audio_duration_sec"] < max_dur]
        logger.info(f"Duration filter [{min_dur}, {max_dur}): {before} -> {len(train_entries)} train entries")

    logger.info(f"Train: {len(train_entries)}, Val: {len(val_entries)}")

    train_ds = create_dataset(train_entries, model, cfg, train=True)
    val_ds = create_dataset(val_entries, model, cfg, train=False)
    collate_fn = get_collate_fn(model, cfg)

    train_loader = make_loader(train_ds, cfg, collate_fn, train=True)
    val_loader = make_loader(val_ds, cfg, collate_fn, train=False)

    checkpoint_dir = MODELS_DIR / f"{cfg.model.type}_word"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=f"{cfg.model.type}-word-{{epoch:02d}}-{{{cfg.checkpoint.monitor}:.4f}}",
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=cfg.checkpoint.save_top_k,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        max_steps=cfg.training.max_steps,
        accumulate_grad_batches=cfg.training.gradient_accumulations,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        val_check_interval=cfg.training.val_check_interval,
        callbacks=callbacks,
        default_root_dir=str(checkpoint_dir),
    )
    if cfg.training.get("validate_first", True):
        trainer.validate(model, val_loader)
    model.train()
    trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.checkpoint.resume_from)


if __name__ == "__main__":
    githash = None #dirtygit.check()
    cfg = load_config(DEFAULT_CONFIG)
    OmegaConf.update(cfg, "githash", githash)
    setup_logging()
    main(cfg)
