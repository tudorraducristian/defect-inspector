import pytest
import torch

from defect_train import train_loop
from defect_train.train_loop import CheckpointMismatch, TrainConfig, learning_rate, train


@pytest.fixture
def tiny_config(tiny_dataset, random_teacher, tmp_path):
    return TrainConfig(dataset_root=tiny_dataset, teacher_weights=random_teacher, penalty_root=None,
                       out_dir=tmp_path / "out", num_workers=0)


def load_teacher_state(config):
    return torch.load(config.teacher_weights)


def raise_oserror(*args):
    raise OSError("disk went away")


def test_short_training_writes_checkpoint(tiny_config):
    train(tiny_config._replace(train_steps=2, checkpoint_every=1), progress=lambda s: None)
    assert torch.load(tiny_config.out_dir / "checkpoint.pt")["step"] == 2


def test_resume_continues_from_saved_step(tiny_config):
    train(tiny_config._replace(train_steps=2, checkpoint_every=1), progress=lambda s: None)
    steps = []
    train(tiny_config._replace(train_steps=4, checkpoint_every=1), progress=steps.append)
    assert [s["step"] for s in steps] == [3, 4]


def test_resume_restores_student_weights(tiny_config):
    first = train(tiny_config._replace(train_steps=2, checkpoint_every=1), progress=lambda s: None)
    saved = {k: v.clone() for k, v in first.student.state_dict().items()}
    finished = train(tiny_config._replace(train_steps=2), progress=lambda s: None)
    assert all(torch.equal(saved[k], v) for k, v in finished.student.state_dict().items())


def test_finished_training_runs_no_more_steps(tiny_config):
    train(tiny_config._replace(train_steps=2), progress=lambda s: None)
    steps = []
    train(tiny_config._replace(train_steps=2), progress=steps.append)
    assert steps == []


def test_final_step_is_saved_between_checkpoint_intervals(tiny_config):
    train(tiny_config._replace(train_steps=3, checkpoint_every=2), progress=lambda s: None)
    assert torch.load(tiny_config.out_dir / "checkpoint.pt")["step"] == 3


def test_interrupted_save_leaves_previous_checkpoint(tiny_config, monkeypatch):
    train(tiny_config._replace(train_steps=2, checkpoint_every=1), progress=lambda s: None)
    monkeypatch.setattr(train_loop.os, "replace", raise_oserror)
    with pytest.raises(OSError):
        train(tiny_config._replace(train_steps=3, checkpoint_every=1), progress=lambda s: None)
    assert torch.load(tiny_config.out_dir / "checkpoint.pt")["step"] == 2


def test_resume_on_other_dataset_is_refused(tiny_config, other_dataset):
    train(tiny_config._replace(train_steps=1, checkpoint_every=1), progress=lambda s: None)
    with pytest.raises(CheckpointMismatch):
        train(tiny_config._replace(dataset_root=other_dataset, train_steps=2), progress=lambda s: None)


def test_resume_with_other_model_shape_is_refused(tiny_config):
    train(tiny_config._replace(train_steps=1), progress=lambda s: None)
    with pytest.raises(CheckpointMismatch):
        train(tiny_config._replace(train_steps=2, image_size=64, use_autoencoder=False),
              progress=lambda s: None)


def test_teacher_stays_frozen(tiny_config):
    before = load_teacher_state(tiny_config)
    parts = train(tiny_config._replace(train_steps=2), progress=lambda s: None)
    assert all(torch.equal(before[k], v) for k, v in parts.teacher.state_dict().items())


def test_student_is_trained(tiny_config):
    torch.manual_seed(0)
    parts = train(tiny_config._replace(train_steps=2), progress=lambda s: None)
    first = parts.student[0].weight.clone()
    more = train(tiny_config._replace(train_steps=3), progress=lambda s: None)
    assert not torch.equal(first, more.student[0].weight)


def test_quantiles_are_ordered(tiny_config):
    q = train(tiny_config._replace(train_steps=2), progress=lambda s: None).quantiles
    assert q["st_start"] <= q["st_end"] and q["ae_start"] <= q["ae_end"]


def test_parts_are_left_in_eval_mode(tiny_config):
    parts = train(tiny_config._replace(train_steps=1), progress=lambda s: None)
    assert not parts.student.training and not parts.autoencoder.training


def test_training_without_autoencoder(tiny_config):
    config = tiny_config._replace(train_steps=1, image_size=64, use_autoencoder=False)
    parts = train(config, progress=lambda s: None)
    assert parts.autoencoder is None and parts.student[-1].out_channels == 384


def test_autoencoder_needs_size_256(tiny_config):
    with pytest.raises(ValueError, match="256"):
        train(tiny_config._replace(train_steps=1, image_size=64), progress=lambda s: None)


def test_training_with_penalty_images(tiny_config, penalty_folder):
    parts = train(tiny_config._replace(train_steps=1, penalty_root=penalty_folder),
                  progress=lambda s: None)
    assert parts.quantiles["st_start"] <= parts.quantiles["st_end"]


def test_learning_rate_drops_tenfold_at_95_percent():
    assert learning_rate(949, 1000) == pytest.approx(1e-4)
    assert learning_rate(950, 1000) == pytest.approx(1e-5)
