"""Train on one dataset, evaluate, and write the inspection profile.

Running the same command again continues from the last checkpoint in --out.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from defect_inspector.preprocessing import IMAGENET_MEAN, IMAGENET_STD, to_model_input
from defect_train.data import load_splits, read_image
from defect_train.evaluate import evaluate
from defect_train.export import to_anomaly_net, write_profile
from defect_train.train_loop import TrainConfig, train

PRINT_EVERY = 500
VERIFY_IMAGES = 4


def require_gpu(torch_module=torch):
    """Stop at once: on the processor, this training would take tens of hours."""
    if not torch_module.cuda.is_available():
        raise SystemExit("Colab nu a dat placă video. Alege Runtime > Change runtime type > "
                         "T4 GPU, apoi Runtime > Run all.")


def _print_progress(update):
    if update["step"] == 1 or update["step"] % PRINT_EVERY == 0:
        print("step {step}, loss {loss:.4f}".format(**update), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--penalty", type=Path, help="folder of unrelated images")
    parser.add_argument("--train-steps", type=int, default=70000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--no-autoencoder", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)

    if args.require_gpu:
        require_gpu()

    config = TrainConfig(
        dataset_root=args.dataset_root,
        teacher_weights=args.teacher,
        penalty_root=args.penalty,
        out_dir=args.out,
        image_size=args.image_size,
        use_autoencoder=not args.no_autoencoder,
        train_steps=args.train_steps,
        num_workers=args.num_workers,
    )
    parts = train(config, progress=_print_progress)
    net = to_anomaly_net(parts)

    splits = load_splits(args.dataset_root)
    device = next(net.parameters()).device
    metrics = evaluate(net, splits, args.image_size, device)
    metrics["train_steps"] = args.train_steps
    metrics["image_size"] = args.image_size
    metrics["use_autoencoder"] = config.use_autoencoder
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    images = np.concatenate([
        to_model_input(read_image(path), args.image_size, IMAGENET_MEAN, IMAGENET_STD)
        for path in splits.val[:VERIFY_IMAGES]
    ])
    write_profile(net, args.out / "profile", args.object_id, args.object_name,
                  args.image_size, metrics, images)
    print("AUROC {:.4f}, proposed threshold {:.4f}".format(metrics["image_auroc"], metrics["threshold"]))


if __name__ == "__main__":
    main()
