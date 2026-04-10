import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.attacks.lira import LiRA, compute_mse_signal
from src.data.eld import (
    build_user_datasets,
    concat_datasets,
    load_eld_matrix,
)
from src.pipeline.train_target import train_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def sample_shadow_split(dataset, train_ratio=0.5, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    num_examples = len(dataset)
    indices = rng.permutation(num_examples)

    split = int(train_ratio * num_examples)
    split = min(max(split, 1), num_examples - 1)

    train_idx = indices[:split].tolist()
    val_idx = indices[split:].tolist()

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train_shadow_models(aux_data, config, num_shadow=8, shadow_train_ratio=0.5, rng=None):
    rng = np.random.default_rng(config["seed"] + 1) if rng is None else rng
    shadow_models = []

    for shadow_idx in range(num_shadow):
        train_ds, val_ds = sample_shadow_split(
            aux_data,
            train_ratio=shadow_train_ratio,
            rng=rng,
        )
        shadow_config = dict(config)
        shadow_config["save_path"] = str(
            Path(config["output_dir"]) / f"shadow_{shadow_idx:02d}.pt"
        )
        model = train_model(train_ds, val_ds, shadow_config)
        shadow_models.append(
            {
                "model": model,
                "train_ds": train_ds,
                "val_ds": val_ds,
            }
        )

    return shadow_models


def run_lira(target_model, shadow_models, member_loader, nonmember_loader, device, batch_size):
    member_signals = []
    nonmember_signals = []

    for shadow in shadow_models:
        model = shadow["model"]
        train_loader = make_loader(shadow["train_ds"], batch_size=batch_size)
        val_loader = make_loader(shadow["val_ds"], batch_size=batch_size)

        member_signals.append(compute_mse_signal(model, train_loader, device))
        nonmember_signals.append(compute_mse_signal(model, val_loader, device))

    member_signals = np.concatenate(member_signals)
    nonmember_signals = np.concatenate(nonmember_signals)

    lira = LiRA()
    lira.fit(member_signals, nonmember_signals)

    target_member = compute_mse_signal(target_model, member_loader, device)
    target_nonmember = compute_mse_signal(target_model, nonmember_loader, device)

    scores_member = lira.score(target_member)
    scores_nonmember = lira.score(target_nonmember)

    return scores_member, scores_nonmember


def roc_curve_from_scores(member_scores, nonmember_scores):
    labels = np.concatenate(
        [
            np.ones(len(member_scores), dtype=np.int64),
            np.zeros(len(nonmember_scores), dtype=np.int64),
        ]
    )
    scores = np.concatenate([member_scores, nonmember_scores])
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]

    pos = max(int(labels.sum()), 1)
    neg = max(int((1 - labels).sum()), 1)

    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)

    tpr = np.concatenate([[0.0], tp / pos, [1.0]])
    fpr = np.concatenate([[0.0], fp / neg, [1.0]])

    return fpr, tpr


def compute_auc(member_scores, nonmember_scores):
    fpr, tpr = roc_curve_from_scores(member_scores, nonmember_scores)
    return float(np.trapz(tpr, fpr))


def compute_tpr_at_fpr(member_scores, nonmember_scores, target_fpr):
    fpr, tpr = roc_curve_from_scores(member_scores, nonmember_scores)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def summarize_attack(member_scores, nonmember_scores):
    return {
        "num_member_windows": int(len(member_scores)),
        "num_nonmember_windows": int(len(nonmember_scores)),
        "member_score_mean": float(np.mean(member_scores)),
        "nonmember_score_mean": float(np.mean(nonmember_scores)),
        "auc": compute_auc(member_scores, nonmember_scores),
        "tpr_at_1pct_fpr": compute_tpr_at_fpr(member_scores, nonmember_scores, 0.01),
        "tpr_at_0.1pct_fpr": compute_tpr_at_fpr(member_scores, nonmember_scores, 0.001),
    }


def resolve_user_split_counts(num_users, args):
    explicit = [
        args.num_train_users,
        args.num_val_users,
        args.num_test_users,
        args.num_aux_users,
    ]
    if all(value is not None for value in explicit):
        total = sum(explicit)
        if total > num_users:
            raise ValueError(
                f"Requested {total} total users, but only {num_users} are available."
            )
        return explicit

    num_train = max(int(num_users * args.train_user_ratio), 1)
    num_val = max(int(num_users * args.val_user_ratio), 1)
    num_test = max(int(num_users * args.test_user_ratio), 1)
    num_aux = num_users - num_train - num_val - num_test

    if num_aux < 1:
        raise ValueError(
            "Not enough users for the requested train/val/test ratios. "
            "Lower one of the target ratios or pass explicit user counts."
        )

    return [num_train, num_val, num_test, num_aux]


def prepare_datasets(args):
    rng = np.random.default_rng(args.seed)

    if args.synthetic:
        matrix = generate_synthetic_matrix(
            num_steps=args.synthetic_steps,
            num_users=args.synthetic_users,
            seed=args.seed,
        )
    elif args.data_path:
        matrix = load_eld_matrix(
            args.data_path,
            series_axis=args.series_axis,
            delimiter=args.delimiter,
        )
    else:
        raise ValueError("Provide --data-path or pass --synthetic for a toy experiment.")

    user_datasets = build_user_datasets(matrix, L=args.L, H=args.H)

    if len(user_datasets) < 4:
        raise ValueError("Need at least four user series for train/val/test/aux splits.")

    permuted_users = rng.permutation(len(user_datasets))
    num_train, num_val, num_test, num_aux = resolve_user_split_counts(
        len(user_datasets), args
    )

    train_user_ids = permuted_users[:num_train]
    val_user_ids = permuted_users[num_train:num_train + num_val]
    test_user_ids = permuted_users[num_train + num_val:num_train + num_val + num_test]
    aux_user_ids = permuted_users[
        num_train + num_val + num_test:num_train + num_val + num_test + num_aux
    ]

    target_train = concat_datasets([user_datasets[idx] for idx in train_user_ids])
    target_val = concat_datasets([user_datasets[idx] for idx in val_user_ids])
    target_test = concat_datasets([user_datasets[idx] for idx in test_user_ids])
    aux_dataset = concat_datasets([user_datasets[idx] for idx in aux_user_ids])

    return {
        "matrix_shape": tuple(int(x) for x in matrix.shape),
        "num_train_users": int(num_train),
        "num_val_users": int(num_val),
        "num_test_users": int(num_test),
        "num_aux_users": int(num_aux),
        "target_train": target_train,
        "target_val": target_val,
        "target_test": target_test,
        "aux_dataset": aux_dataset,
    }


def generate_synthetic_matrix(num_steps=512, num_users=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(num_steps, dtype=np.float32)
    columns = []

    for user_idx in range(num_users):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = 0.8 + 0.4 * rng.random()
        seasonal = amplitude * np.sin((2.0 * np.pi / 24.0) * t + phase)
        slow = 0.5 * np.sin((2.0 * np.pi / 168.0) * t + phase / 2.0)
        trend = 0.002 * user_idx * t / max(num_steps, 1)
        noise = rng.normal(0.0, 0.08, size=num_steps)
        series = seasonal + slow + trend + noise
        columns.append(series.astype(np.float32))

    return np.stack(columns, axis=1)


def build_config(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "H": args.H,
        "device": device,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "save_path": str(output_dir / "target.pt"),
    }


def run_experiment(args):
    seed_everything(args.seed)
    config = build_config(args)
    datasets = prepare_datasets(args)

    target_model = train_model(datasets["target_train"], datasets["target_val"], config)
    shadow_models = train_shadow_models(
        datasets["aux_dataset"],
        config,
        num_shadow=args.num_shadow,
        shadow_train_ratio=args.shadow_train_ratio,
    )

    member_loader = make_loader(datasets["target_train"], batch_size=args.batch_size)
    nonmember_loader = make_loader(datasets["target_test"], batch_size=args.batch_size)
    scores_member, scores_nonmember = run_lira(
        target_model,
        shadow_models,
        member_loader,
        nonmember_loader,
        device=config["device"],
        batch_size=args.batch_size,
    )

    summary = summarize_attack(scores_member, scores_nonmember)
    summary.update(
        {
            "device": config["device"],
            "matrix_shape": datasets["matrix_shape"],
            "num_train_users": datasets["num_train_users"],
            "num_val_users": datasets["num_val_users"],
            "num_test_users": datasets["num_test_users"],
            "num_aux_users": datasets["num_aux_users"],
            "num_shadow_models": args.num_shadow,
            "window_length": args.L,
            "horizon": args.H,
        }
    )

    output_path = Path(args.output_dir) / "results.json"
    output_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved results to {output_path}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a target forecaster and evaluate a LiRA attack."
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--series-axis", choices=["auto", "0", "1"], default="auto")
    parser.add_argument("--output-dir", type=str, default="artifacts/lira_run")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--H", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-shadow", type=int, default=8)
    parser.add_argument("--shadow-train-ratio", type=float, default=0.5)
    parser.add_argument("--num-train-users", type=int, default=None)
    parser.add_argument("--num-val-users", type=int, default=None)
    parser.add_argument("--num-test-users", type=int, default=None)
    parser.add_argument("--num-aux-users", type=int, default=None)
    parser.add_argument("--train-user-ratio", type=float, default=0.2)
    parser.add_argument("--val-user-ratio", type=float, default=0.2)
    parser.add_argument("--test-user-ratio", type=float, default=0.2)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-steps", type=int, default=512)
    parser.add_argument("--synthetic-users", type=int, default=12)

    args = parser.parse_args()
    args.series_axis = (
        args.series_axis if args.series_axis == "auto" else int(args.series_axis)
    )
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
