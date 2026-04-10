import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.eld import (
    build_user_datasets,
    concat_datasets,
    load_eld_matrix,
    load_eld_raw_txt,
    preprocess_eld_matrix,
    summarize_eld_preprocessing,
)
from src.pipeline.train_target import evaluate, train_model


PAPER_REFS = {
    "eld_lstm_record_lira_online_mse": {
        "table": "Table IV",
        "description": "ELD, LSTM, record-level LiRA online, MSE signal",
        "tpr_at_0.1pct_fpr": {"mean": 2.26, "std": 3.18},
        "tpr_at_0.01pct_fpr": {"mean": 0.84, "std": 1.19},
    },
    "eld_lstm_record_lira_offline_mse": {
        "table": "Table IV",
        "description": "ELD, LSTM, record-level LiRA offline, MSE signal",
        "tpr_at_0.1pct_fpr": {"mean": 0.03, "std": 0.03},
        "tpr_at_0.01pct_fpr": {"mean": 0.00, "std": 0.00},
    }
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def norm_logpdf(x, mean, std):
    std = max(float(std), 1e-8)
    z = (float(x) - float(mean)) / std
    return -0.5 * z * z - math.log(std) - 0.5 * math.log(2.0 * math.pi)


def make_loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_subset_from_indices(dataset, indices):
    return Subset(dataset, np.asarray(indices, dtype=np.int64).tolist())


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
            "Lower one of the ratios or pass explicit user counts."
        )
    return [num_train, num_val, num_test, num_aux]


def load_matrix(args):
    if args.synthetic:
        return generate_synthetic_matrix(
            num_steps=args.synthetic_steps,
            num_users=args.synthetic_users,
            seed=args.seed,
        )

    if not args.data_path:
        raise ValueError("Provide --data-path or pass --synthetic.")

    path = Path(args.data_path)
    if args.eld_raw_format:
        raw_matrix = load_eld_raw_txt(str(path))
        return preprocess_eld_matrix(
            raw_matrix,
            aggregate_factor=args.eld_aggregate_factor,
            min_valid_steps=args.eld_min_valid_steps,
            min_mean=args.eld_min_mean,
            max_mean=args.eld_max_mean,
            truncate_length=args.eld_truncate_length,
            mean_filter_mode=args.eld_mean_filter_mode,
            target_users=args.eld_target_users,
        )

    return load_eld_matrix(
        str(path),
        series_axis=args.series_axis,
        delimiter=args.delimiter,
    )


def load_eld_matrix_with_diagnostics(args):
    path = Path(args.data_path)
    raw_matrix = load_eld_raw_txt(str(path))
    diagnostics = summarize_eld_preprocessing(
        raw_matrix,
        aggregate_factor=args.eld_aggregate_factor,
        min_valid_steps=args.eld_min_valid_steps,
        min_mean=args.eld_min_mean,
        max_mean=args.eld_max_mean,
        truncate_length=args.eld_truncate_length,
        mean_filter_mode=args.eld_mean_filter_mode,
        target_users=args.eld_target_users,
    )
    matrix = preprocess_eld_matrix(
        raw_matrix,
        aggregate_factor=args.eld_aggregate_factor,
        min_valid_steps=args.eld_min_valid_steps,
        min_mean=args.eld_min_mean,
        max_mean=args.eld_max_mean,
        truncate_length=args.eld_truncate_length,
        mean_filter_mode=args.eld_mean_filter_mode,
        target_users=args.eld_target_users,
    )
    diagnostics["kept_series"] = None
    return matrix, diagnostics


def prepare_user_datasets(args, seed):
    local_args = argparse.Namespace(**vars(args))
    local_args.seed = seed
    preprocessing_diagnostics = None
    if args.eld_raw_format:
        matrix, preprocessing_diagnostics = load_eld_matrix_with_diagnostics(local_args)
    else:
        matrix = load_matrix(local_args)
    user_datasets = build_user_datasets(matrix, L=args.L, H=args.H)

    if args.require_exact_users is not None and len(user_datasets) != args.require_exact_users:
        raise ValueError(
            f"Paper-style preprocessing expected {args.require_exact_users} users, "
            f"but found {len(user_datasets)}."
        )

    if args.max_users is not None and len(user_datasets) > args.max_users:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(user_datasets), size=args.max_users, replace=False)
        user_datasets = [user_datasets[idx] for idx in sorted(chosen.tolist())]

    if len(user_datasets) < 4:
        raise ValueError("Need at least four user series for train/val/test/aux splits.")

    rng = np.random.default_rng(seed)
    permuted_users = rng.permutation(len(user_datasets))
    num_train, num_val, num_test, num_aux = resolve_user_split_counts(
        len(user_datasets), args
    )

    train_ids = permuted_users[:num_train]
    val_ids = permuted_users[num_train:num_train + num_val]
    test_ids = permuted_users[num_train + num_val:num_train + num_val + num_test]
    aux_ids = permuted_users[
        num_train + num_val + num_test:num_train + num_val + num_test + num_aux
    ]

    target_train = concat_datasets([user_datasets[idx] for idx in train_ids])
    target_val = concat_datasets([user_datasets[idx] for idx in val_ids])
    target_test = concat_datasets([user_datasets[idx] for idx in test_ids])
    aux_dataset = concat_datasets([user_datasets[idx] for idx in aux_ids])

    return {
        "matrix_shape": tuple(int(x) for x in matrix.shape),
        "num_total_users": int(len(user_datasets)),
        "num_train_users": int(num_train),
        "num_val_users": int(num_val),
        "num_test_users": int(num_test),
        "num_aux_users": int(num_aux),
        "target_train_users": [user_datasets[idx] for idx in train_ids],
        "target_test_users": [user_datasets[idx] for idx in test_ids],
        "aux_users": [user_datasets[idx] for idx in aux_ids],
        "target_train": target_train,
        "target_val": target_val,
        "target_test": target_test,
        "aux_dataset": aux_dataset,
        "preprocessing_diagnostics": preprocessing_diagnostics,
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
        columns.append((seasonal + slow + trend + noise).astype(np.float32))

    return np.stack(columns, axis=1)


def build_config(args, seed, output_dir, model_name):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "H": args.H,
        "device": device,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "seed": seed,
        "output_dir": str(output_dir),
        "save_path": str(output_dir / f"{model_name}.pt"),
    }


def compute_model_metrics(model, dataset, config):
    loader = make_loader(dataset, config["batch_size"])
    mae = evaluate(model, loader, config)

    mse_total = 0.0
    smape_total = 0.0
    nd_num = 0.0
    nd_den = 0.0
    count = 0

    device = config["device"]
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)

            mse_batch = torch.mean((y_hat - y) ** 2, dim=1)
            mse_total += mse_batch.sum().item()

            smape_batch = torch.mean(
                torch.abs(y_hat - y) / (torch.abs(y_hat) + torch.abs(y) + 1e-8),
                dim=1,
            )
            smape_total += smape_batch.sum().item()

            nd_num += torch.sum(torch.abs(y_hat - y)).item()
            nd_den += torch.sum(torch.abs(y)).item()
            count += x.size(0)

    return {
        "mse": mse_total / max(count, 1),
        "mae": mae,
        "smape": smape_total / max(count, 1),
        "nd": nd_num / max(nd_den, 1e-8),
    }


def compute_signal(model, dataset, device, batch_size, signal_name):
    loader = make_loader(dataset, batch_size=batch_size)
    values = []
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)

            if signal_name == "mse":
                signal = torch.mean((y_hat - y) ** 2, dim=1)
            elif signal_name == "mae":
                signal = torch.mean(torch.abs(y_hat - y), dim=1)
            else:
                raise ValueError(f"Unsupported signal: {signal_name}")

            values.extend(signal.cpu().numpy().tolist())

    return np.asarray(values, dtype=np.float64)


def train_target_model(datasets, args, seed, run_dir):
    config = build_config(args, seed, run_dir, "target")
    model = train_model(datasets["target_train"], datasets["target_val"], config)
    metrics = {
        "train": compute_model_metrics(model, datasets["target_train"], config),
        "val": compute_model_metrics(model, datasets["target_val"], config),
        "test": compute_model_metrics(model, datasets["target_test"], config),
    }
    return model, config, metrics


def train_shadow_models_offline(aux_users, args, config, seed, run_dir):
    rng = np.random.default_rng(seed + 1000)
    shadow_signals = []
    for shadow_idx in range(args.num_shadow):
        indices = rng.permutation(len(aux_users))
        split = len(aux_users) // 2
        train_users = [aux_users[idx] for idx in indices[:split]]
        val_users = [aux_users[idx] for idx in indices[split:]]
        train_ds = concat_datasets(train_users)
        val_ds = concat_datasets(val_users)

        shadow_config = dict(config)
        shadow_config["save_path"] = str(run_dir / f"shadow_{shadow_idx:02d}.pt")
        model = train_model(train_ds, val_ds, shadow_config)

        member_signal = compute_signal(
            model, train_ds, config["device"], config["batch_size"], args.signal
        )
        nonmember_signal = compute_signal(
            model, val_ds, config["device"], config["batch_size"], args.signal
        )

        shadow_signals.append(
            {
                "member_mean": float(np.mean(member_signal)),
                "member_std": float(np.std(member_signal) + 1e-8),
                "nonmember_mean": float(np.mean(nonmember_signal)),
                "nonmember_std": float(np.std(nonmember_signal) + 1e-8),
            }
        )

    return shadow_signals


def compute_online_candidate_stats(
    candidate_users,
    pool_users,
    args,
    config,
    seed,
    run_dir,
):
    candidate_dataset = concat_datasets(candidate_users)
    candidate_len = len(candidate_dataset)
    rng = np.random.default_rng(seed + 2000)

    in_signals = [[] for _ in range(candidate_len)]
    out_signals = [[] for _ in range(candidate_len)]

    candidate_record_masks = []
    for candidate_idx, ds in enumerate(candidate_users):
        candidate_record_masks.extend([candidate_idx] * len(ds))

    for shadow_idx in range(args.num_shadow):
        selected_user_ids = rng.choice(len(pool_users), size=len(pool_users) // 2, replace=False)
        selected_set = set(int(idx) for idx in selected_user_ids.tolist())
        shadow_train_users = [pool_users[idx] for idx in selected_user_ids]
        shadow_val_users = [pool_users[idx] for idx in range(len(pool_users)) if idx not in selected_set]
        shadow_train = concat_datasets(shadow_train_users)
        shadow_val = concat_datasets(shadow_val_users)
        shadow_config = dict(config)
        shadow_config["save_path"] = str(run_dir / f"shadow_online_{shadow_idx:02d}.pt")
        model = train_model(shadow_train, shadow_val, shadow_config)
        candidate_signal = compute_signal(
            model, candidate_dataset, config["device"], config["batch_size"], args.signal
        )

        for idx, signal in enumerate(candidate_signal.tolist()):
            candidate_user_idx = candidate_record_masks[idx]
            if candidate_user_idx in selected_set:
                in_signals[idx].append(signal)
            else:
                out_signals[idx].append(signal)

    stats = []
    for idx in range(candidate_len):
        in_values = in_signals[idx]
        out_values = out_signals[idx]

        if not in_values:
            in_values = out_values[:]
        if not out_values:
            out_values = in_values[:]

        stats.append(
            {
                "in_mean": float(np.mean(in_values)),
                "in_std": float(np.std(in_values) + 1e-8),
                "out_mean": float(np.mean(out_values)),
                "out_std": float(np.std(out_values) + 1e-8),
            }
        )

    return stats


def score_lira_online(target_signals, candidate_stats):
    scores = []
    for signal, stats in zip(target_signals.tolist(), candidate_stats):
        log_in = norm_logpdf(signal, stats["in_mean"], stats["in_std"])
        log_out = norm_logpdf(signal, stats["out_mean"], stats["out_std"])
        scores.append(log_in - log_out)
    return np.asarray(scores, dtype=np.float64)


def score_lira_offline(target_signals, shadow_stats):
    member_mean = float(np.mean([item["member_mean"] for item in shadow_stats]))
    member_std = float(np.mean([item["member_std"] for item in shadow_stats]))
    nonmember_mean = float(np.mean([item["nonmember_mean"] for item in shadow_stats]))
    nonmember_std = float(np.mean([item["nonmember_std"] for item in shadow_stats]))

    scores = []
    for signal in target_signals.tolist():
        log_in = norm_logpdf(signal, member_mean, member_std)
        log_out = norm_logpdf(signal, nonmember_mean, nonmember_std)
        scores.append(log_in - log_out)
    return np.asarray(scores, dtype=np.float64)


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
    return float(np.trapezoid(tpr, fpr))


def compute_tpr_at_fpr(member_scores, nonmember_scores, target_fpr):
    fpr, tpr = roc_curve_from_scores(member_scores, nonmember_scores)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def summarize_scores(member_scores, nonmember_scores):
    return {
        "num_member_windows": int(len(member_scores)),
        "num_nonmember_windows": int(len(nonmember_scores)),
        "member_score_mean": float(np.mean(member_scores)),
        "nonmember_score_mean": float(np.mean(nonmember_scores)),
        "auc": compute_auc(member_scores, nonmember_scores),
        "tpr_at_0.1pct_fpr": compute_tpr_at_fpr(member_scores, nonmember_scores, 0.001),
        "tpr_at_0.01pct_fpr": compute_tpr_at_fpr(member_scores, nonmember_scores, 0.0001),
        "tpr_at_1pct_fpr": compute_tpr_at_fpr(member_scores, nonmember_scores, 0.01),
    }


def mean_and_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def aggregate_run_summaries(run_summaries):
    metric_keys = [
        "auc",
        "tpr_at_0.1pct_fpr",
        "tpr_at_0.01pct_fpr",
        "tpr_at_1pct_fpr",
    ]
    aggregated = {
        "num_runs": len(run_summaries),
        "per_run": run_summaries,
    }
    for key in metric_keys:
        aggregated[key] = mean_and_std([run[key] for run in run_summaries])
    return aggregated


def compare_to_paper(experiment_key, metrics):
    if experiment_key not in PAPER_REFS:
        return None

    ref = PAPER_REFS[experiment_key]
    comparison = {"reference": ref}
    for metric_name in ["tpr_at_0.1pct_fpr", "tpr_at_0.01pct_fpr"]:
        if metric_name not in metrics:
            continue
        ours = metrics[metric_name]["mean"]
        paper_mean = ref[metric_name]["mean"]
        comparison[metric_name] = {
            "ours_mean": ours,
            "paper_mean": paper_mean,
            "delta": ours - paper_mean,
        }
    return comparison


def write_paper_report(args, final_summary, output_dir):
    report_path = output_dir / "paper_report.md"
    comparison = final_summary.get("paper_comparison")
    target_metrics = final_summary["target_model_metrics"]

    lines = [
        "# Paper Mapping Report",
        "",
        "## Selected Paper Configuration",
        "",
        "- Dataset: ELD",
        "- Target architecture: LSTM",
        "- Attack: record-level LiRA",
        f"- Signal: {args.signal.upper()}",
        f"- Setting: {args.attack_setting}",
        "",
        "## Section Mapping",
        "",
        "- Section III-B: implemented fixed-window forecasting with look-back `L=100` and prediction horizon `H=20`.",
        "- Section VI-A: implemented ELD-oriented loading and optional raw preprocessing, including hourly aggregation and household filtering.",
        "- Section VI-B: implemented user-based splits with the paper counts `20/20/20/40` for train/val/test/aux and optional `max_users=100` sampling.",
        "- Section VI-C: used the repo’s LSTM forecasting model as the paper-targeted architecture for this subset.",
        "- Section VI-D: used MAE training objective, Adam optimizer, early stopping with patience `3`, and configurable max epochs.",
        "- Section VI-F: implemented LiRA evaluation with `64` shadow models when the paper preset is enabled.",
        "- Section VII-B / Table IV: reported record-level TPR at fixed low FPRs.",
        "",
        "## Target Model Metrics",
        "",
        f"- Train MSE/MAE/SMAPE/ND: {target_metrics['train']}",
        f"- Val MSE/MAE/SMAPE/ND: {target_metrics['val']}",
        f"- Test MSE/MAE/SMAPE/ND: {target_metrics['test']}",
        "",
        "## Attack Results",
        "",
        f"- TPR@0.1% FPR mean±std: {final_summary['attack_metrics']['tpr_at_0.1pct_fpr']}",
        f"- TPR@0.01% FPR mean±std: {final_summary['attack_metrics']['tpr_at_0.01pct_fpr']}",
        "",
        "## Dataset Processing Check",
        "",
        f"- Diagnostics: {final_summary.get('preprocessing_diagnostics')}",
        "",
        "## Why This Corresponds To The Paper",
        "",
        "This run corresponds to the paper’s record-level ELD + LSTM + LiRA experiment when the dataset preprocessing yields the intended middle-band user pool and the experiment uses the same user split, model, shadow count, and fixed-FPR metrics.",
    ]

    if comparison:
        lines.extend(
            [
                "",
                "## Comparison To Paper",
                "",
                f"- Reference entry: {comparison['reference']['description']} ({comparison['reference']['table']})",
                f"- TPR@0.1% FPR delta: {comparison['tpr_at_0.1pct_fpr']}",
                f"- TPR@0.01% FPR delta: {comparison['tpr_at_0.01pct_fpr']}",
                "",
                "Differences from the paper can still arise from dataset-format differences before preprocessing and from the fallback mean-filter strategy when absolute paper thresholds do not transfer to this mirror.",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def run_single_experiment(args, run_seed, run_dir):
    seed_everything(run_seed)
    datasets = prepare_user_datasets(args, run_seed)
    target_model, config, target_metrics = train_target_model(datasets, args, run_seed, run_dir)

    target_member_signal = compute_signal(
        target_model,
        datasets["target_train"],
        config["device"],
        config["batch_size"],
        args.signal,
    )
    target_nonmember_signal = compute_signal(
        target_model,
        datasets["target_test"],
        config["device"],
        config["batch_size"],
        args.signal,
    )

    if args.attack_setting == "online":
        candidate_stats = compute_online_candidate_stats(
            datasets["target_train_users"],
            datasets["target_train_users"] + datasets["target_test_users"],
            args,
            config,
            run_seed,
            run_dir,
        )
        member_scores = score_lira_online(target_member_signal, candidate_stats)

        candidate_stats_nonmember = compute_online_candidate_stats(
            datasets["target_test_users"],
            datasets["target_train_users"] + datasets["target_test_users"],
            args,
            config,
            run_seed + 17,
            run_dir,
        )
        nonmember_scores = score_lira_online(target_nonmember_signal, candidate_stats_nonmember)
    else:
        shadow_stats = train_shadow_models_offline(
            datasets["aux_users"],
            args,
            config,
            run_seed,
            run_dir,
        )
        member_scores = score_lira_offline(target_member_signal, shadow_stats)
        nonmember_scores = score_lira_offline(target_nonmember_signal, shadow_stats)

    summary = summarize_scores(member_scores, nonmember_scores)
    summary.update(
        {
            "seed": run_seed,
            "device": config["device"],
            "matrix_shape": datasets["matrix_shape"],
            "num_total_users": datasets["num_total_users"],
            "num_train_users": datasets["num_train_users"],
            "num_val_users": datasets["num_val_users"],
            "num_test_users": datasets["num_test_users"],
            "num_aux_users": datasets["num_aux_users"],
            "num_shadow_models": args.num_shadow,
            "window_length": args.L,
            "horizon": args.H,
            "preprocessing_diagnostics": datasets["preprocessing_diagnostics"],
        }
    )

    return summary, target_metrics


def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summaries = []
    target_metrics_runs = []
    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        run_dir = output_dir / f"run_{run_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary, target_metrics = run_single_experiment(args, run_seed, run_dir)
        (run_dir / "results.json").write_text(json.dumps(summary, indent=2))
        run_summaries.append(summary)
        target_metrics_runs.append(target_metrics)

    attack_metrics = aggregate_run_summaries(run_summaries)
    final_summary = {
        "experiment_name": args.experiment_name,
        "attack_setting": args.attack_setting,
        "signal": args.signal,
        "num_runs": args.num_runs,
        "paper_preset": args.paper_preset,
        "attack_metrics": attack_metrics,
        "target_model_metrics": target_metrics_runs[0],
        "preprocessing_diagnostics": run_summaries[0].get("preprocessing_diagnostics"),
        "config": {
            "L": args.L,
            "H": args.H,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_shadow": args.num_shadow,
            "num_train_users": args.num_train_users,
            "num_val_users": args.num_val_users,
            "num_test_users": args.num_test_users,
            "num_aux_users": args.num_aux_users,
            "max_users": args.max_users,
        },
    }

    if args.paper_preset in PAPER_REFS:
        final_summary["paper_comparison"] = compare_to_paper(
            args.paper_preset,
            attack_metrics,
        )

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(final_summary, indent=2))
    report_path = write_paper_report(args, final_summary, output_dir)

    print(json.dumps(final_summary, indent=2))
    print(f"Saved results to {results_path}")
    print(f"Saved paper report to {report_path}")
    return final_summary


def apply_paper_preset(args):
    if args.paper_preset == "eld_lstm_record_lira_online_mse":
        args.experiment_name = "ELD LSTM record-level LiRA online MSE"
        args.attack_setting = "online"
        args.signal = "mse"
        args.require_exact_users = 100
        args.eld_raw_format = True
        args.L = 100
        args.H = 20
        args.epochs = 50
        args.patience = 3
        args.batch_size = 1024
        args.lr = 1e-3
        args.num_shadow = 64
        args.num_runs = 5
        args.max_users = 100
        args.num_train_users = 20
        args.num_val_users = 20
        args.num_test_users = 20
        args.num_aux_users = 40
    if args.paper_preset == "eld_lstm_record_lira_offline_mse":
        args.experiment_name = "ELD LSTM record-level LiRA offline MSE"
        args.attack_setting = "offline"
        args.signal = "mse"
        args.require_exact_users = 100
        args.eld_raw_format = True
        args.eld_mean_filter_mode = "absolute"
        args.eld_target_users = None
        args.L = 100
        args.H = 20
        args.epochs = 50
        args.patience = 3
        args.batch_size = 1024
        args.lr = 1e-3
        args.num_shadow = 64
        args.num_runs = 5
        args.max_users = 100
        args.num_train_users = 20
        args.num_val_users = 20
        args.num_test_users = 20
        args.num_aux_users = 40
    if args.paper_preset == "eld_lstm_record_lira_offline_mse_kaggle_fallback":
        args.experiment_name = "ELD LSTM record-level LiRA offline MSE (Kaggle fallback)"
        args.attack_setting = "offline"
        args.signal = "mse"
        args.require_exact_users = 100
        args.eld_raw_format = True
        args.eld_mean_filter_mode = "middle_n"
        args.eld_target_users = 100
        args.L = 100
        args.H = 20
        args.epochs = 50
        args.patience = 3
        args.batch_size = 1024
        args.lr = 1e-3
        args.num_shadow = 64
        args.num_runs = 5
        args.max_users = 100
        args.num_train_users = 20
        args.num_val_users = 20
        args.num_test_users = 20
        args.num_aux_users = 40
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an LSTM target model and evaluate paper-aligned record-level LiRA."
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--series-axis", choices=["auto", "0", "1"], default="auto")
    parser.add_argument("--output-dir", type=str, default="artifacts/lira_run")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--experiment-name", type=str, default="custom")
    parser.add_argument("--paper-preset", type=str, default=None)
    parser.add_argument("--attack-setting", choices=["online", "offline"], default="online")
    parser.add_argument("--signal", choices=["mse", "mae"], default="mse")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--H", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-shadow", type=int, default=8)
    parser.add_argument("--num-train-users", type=int, default=None)
    parser.add_argument("--num-val-users", type=int, default=None)
    parser.add_argument("--num-test-users", type=int, default=None)
    parser.add_argument("--num-aux-users", type=int, default=None)
    parser.add_argument("--train-user-ratio", type=float, default=0.2)
    parser.add_argument("--val-user-ratio", type=float, default=0.2)
    parser.add_argument("--test-user-ratio", type=float, default=0.2)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--require-exact-users", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-steps", type=int, default=512)
    parser.add_argument("--synthetic-users", type=int, default=12)
    parser.add_argument("--eld-raw-format", action="store_true")
    parser.add_argument("--eld-aggregate-factor", type=int, default=4)
    parser.add_argument("--eld-min-valid-steps", type=int, default=15000)
    parser.add_argument("--eld-min-mean", type=float, default=200.0)
    parser.add_argument("--eld-max-mean", type=float, default=2000.0)
    parser.add_argument("--eld-truncate-length", type=int, default=15000)
    parser.add_argument(
        "--eld-mean-filter-mode",
        choices=["absolute", "middle_n", "none"],
        default="absolute",
    )
    parser.add_argument("--eld-target-users", type=int, default=None)

    args = parser.parse_args()
    args.series_axis = args.series_axis if args.series_axis == "auto" else int(args.series_axis)
    if args.paper_preset:
        args = apply_paper_preset(args)
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
