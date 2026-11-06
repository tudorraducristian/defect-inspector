"""EfficientAD training, resumable after a stopped Colab session.

Follows train, teacher_normalization and map_normalization from EfficientAD, see NOTICE.
The saving and resuming are new.
"""
import hashlib
import itertools
import os
import random
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from defect_train.architecture import get_autoencoder, get_pdn_small
from defect_train.data import PenaltyImages, TrainImages, load_splits

OUT_CHANNELS = 384
BASE_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
HARD_LOSS_QUANTILE = 0.999
MAP_QUANTILE_START = 0.9
MAP_QUANTILE_END = 0.995
CHECKPOINT_NAME = "checkpoint.pt"


class CheckpointMismatch(Exception):
    """The checkpoint in the output folder belongs to a different training."""


class TrainConfig(NamedTuple):
    dataset_root: Path
    teacher_weights: Path
    penalty_root: Optional[Path]
    out_dir: Path
    image_size: int = 256
    use_autoencoder: bool = True
    train_steps: int = 70000
    checkpoint_every: int = 2000
    seed: int = 42
    num_workers: int = 4


class TrainedParts(NamedTuple):
    teacher: torch.nn.Module
    student: torch.nn.Module
    autoencoder: Optional[torch.nn.Module]
    teacher_mean: torch.Tensor
    teacher_std: torch.Tensor
    quantiles: dict


def learning_rate(step_index, train_steps):
    """Return the rate for a 0-based step: ten times lower from 95% of the steps on."""
    return BASE_LEARNING_RATE * (0.1 if step_index >= int(0.95 * train_steps) else 1.0)


def _fingerprint(root, paths):
    """Hash the names and sizes of the files, to recognize the dataset on resume."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        entry = "{}:{}\n".format(path.relative_to(root).as_posix(), path.stat().st_size)
        digest.update(entry.encode("utf-8"))
    return digest.hexdigest()


def _save_checkpoint(path, state):
    """Write beside the target and swap, so a stopped session cannot leave half a file."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _forever(loader):
    while True:
        yield from loader


@torch.no_grad()
def _teacher_normalization(teacher, loader, device):
    means = []
    for image, _ in loader:
        means.append(torch.mean(teacher(image.to(device)), dim=[0, 2, 3]))
    channel_mean = torch.mean(torch.stack(means), dim=0)[None, :, None, None]

    variances = []
    for image, _ in loader:
        distance = (teacher(image.to(device)) - channel_mean) ** 2
        variances.append(torch.mean(distance, dim=[0, 2, 3]))
    channel_var = torch.mean(torch.stack(variances), dim=0)[None, :, None, None]
    return channel_mean, torch.sqrt(channel_var)


@torch.no_grad()
def _map_quantiles(parts, loader, device):
    maps_st, maps_ae = [], []
    for image, _ in loader:
        image = image.to(device)
        teacher_output = (parts.teacher(image) - parts.teacher_mean) / parts.teacher_std
        student_output = parts.student(image)
        maps_st.append(torch.mean((teacher_output - student_output[:, :OUT_CHANNELS]) ** 2,
                                  dim=1, keepdim=True))
        if parts.autoencoder is not None:
            ae_output = parts.autoencoder(image)
            maps_ae.append(torch.mean((ae_output - student_output[:, OUT_CHANNELS:]) ** 2,
                                      dim=1, keepdim=True))
    maps_st = torch.cat(maps_st)
    quantiles = {
        "st_start": torch.quantile(maps_st, MAP_QUANTILE_START).item(),
        "st_end": torch.quantile(maps_st, MAP_QUANTILE_END).item(),
        "ae_start": 0.0,
        "ae_end": 1.0,
    }
    if maps_ae:
        maps_ae = torch.cat(maps_ae)
        quantiles["ae_start"] = torch.quantile(maps_ae, MAP_QUANTILE_START).item()
        quantiles["ae_end"] = torch.quantile(maps_ae, MAP_QUANTILE_END).item()
    return quantiles


def _loss(teacher, student, autoencoder, mean, std, image_st, image_ae, image_penalty):
    with torch.no_grad():
        teacher_st = (teacher(image_st) - mean) / std
    distance_st = (teacher_st - student(image_st)[:, :OUT_CHANNELS]) ** 2
    d_hard = torch.quantile(distance_st, q=HARD_LOSS_QUANTILE)
    loss = torch.mean(distance_st[distance_st >= d_hard])

    if image_penalty is not None:
        loss = loss + torch.mean(student(image_penalty)[:, :OUT_CHANNELS] ** 2)

    if autoencoder is not None:
        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_ae = (teacher(image_ae) - mean) / std
        student_ae = student(image_ae)[:, OUT_CHANNELS:]
        loss = loss + torch.mean((teacher_ae - ae_output) ** 2)
        loss = loss + torch.mean((ae_output - student_ae) ** 2)
    return loss


def _load_checkpoint(path, identity, device):
    """Return the saved state, or None; refuse one that belongs to another training."""
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=device)
    for key, value in identity.items():
        if checkpoint[key] != value:
            raise CheckpointMismatch(
                "{} belongs to another training ({} differs); use a new output folder".format(
                    path, key))
    return checkpoint


def train(config, progress=print):
    """Train, or continue from the checkpoint in out_dir, and return the trained parts."""
    if config.use_autoencoder and config.image_size != 256:
        raise ValueError("the autoencoder has fixed sizes and needs image_size 256")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME

    root = Path(config.dataset_root)
    splits = load_splits(root)
    identity = {
        "fingerprint": _fingerprint(root, splits.train + splits.val),
        "image_size": config.image_size,
        "use_autoencoder": config.use_autoencoder,
    }
    checkpoint = _load_checkpoint(checkpoint_path, identity, device)
    start_step = checkpoint["step"] if checkpoint else 0

    # A different seed after a resume, so the session does not repeat the same images.
    torch.manual_seed(config.seed + start_step)
    np.random.seed(config.seed + start_step)
    random.seed(config.seed + start_step)

    teacher = get_pdn_small(OUT_CHANNELS)
    teacher.load_state_dict(torch.load(config.teacher_weights, map_location="cpu"))
    teacher = teacher.to(device).eval()
    teacher.requires_grad_(False)

    student_channels = 2 * OUT_CHANNELS if config.use_autoencoder else OUT_CHANNELS
    student = get_pdn_small(student_channels).to(device)
    autoencoder = get_autoencoder(OUT_CHANNELS).to(device) if config.use_autoencoder else None

    loader_options = {"batch_size": 1, "num_workers": config.num_workers,
                      "pin_memory": device.type == "cuda"}
    train_images = TrainImages(splits.train, config.image_size)
    trained_parameters = itertools.chain(
        student.parameters(), autoencoder.parameters() if autoencoder is not None else [])
    optimizer = torch.optim.Adam(trained_parameters, lr=BASE_LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    if checkpoint:
        student.load_state_dict(checkpoint["student"])
        if autoencoder is not None:
            autoencoder.load_state_dict(checkpoint["autoencoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        teacher_mean = checkpoint["teacher_mean"]
        teacher_std = checkpoint["teacher_std"]
    else:
        teacher_mean, teacher_std = _teacher_normalization(
            teacher, DataLoader(train_images, **loader_options), device)

    if start_step < config.train_steps:
        endless = dict(loader_options, shuffle=True, persistent_workers=config.num_workers > 0)
        images = _forever(DataLoader(train_images, **endless))
        penalties = itertools.repeat(None)
        if config.penalty_root is not None:
            penalty_images = PenaltyImages(config.penalty_root, config.image_size)
            penalties = _forever(DataLoader(penalty_images, **endless))

        student.train()
        if autoencoder is not None:
            autoencoder.train()

        for step in range(start_step + 1, config.train_steps + 1):
            image_st, image_ae = next(images)
            image_penalty = next(penalties)
            if image_penalty is not None:
                image_penalty = image_penalty.to(device)
            loss = _loss(teacher, student, autoencoder, teacher_mean, teacher_std,
                         image_st.to(device), image_ae.to(device), image_penalty)

            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step - 1, config.train_steps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress({"step": step, "loss": loss.item()})

            if step % config.checkpoint_every == 0 or step == config.train_steps:
                _save_checkpoint(checkpoint_path, dict(
                    identity,
                    step=step,
                    student=student.state_dict(),
                    autoencoder=autoencoder.state_dict() if autoencoder is not None else None,
                    optimizer=optimizer.state_dict(),
                    teacher_mean=teacher_mean,
                    teacher_std=teacher_std,
                ))

    student.eval()
    if autoencoder is not None:
        autoencoder.eval()
    parts = TrainedParts(teacher, student, autoencoder, teacher_mean, teacher_std, {})
    val_loader = DataLoader(TrainImages(splits.val, config.image_size), **loader_options)
    return parts._replace(quantiles=_map_quantiles(parts, val_loader, device))
