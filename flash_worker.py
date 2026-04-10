from pathlib import Path

from runpod_flash import Endpoint, GpuType


@Endpoint(
    name="eld_lira_4090",
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    dependencies=["numpy", "scipy", "torch", "torchvision", "torchaudio", "kagglehub"],
)
async def eld_lira_4090(input_data: dict) -> dict:
    from argparse import Namespace

    from src.pipeline.run_lira import run_experiment

    default_eld_path = (
        "$HOME/.cache/kagglehub/datasets/eduardojst10/"
        "electricityloaddiagrams20112014/versions/1/df_kwh_adjusted.csv"
    )

    requested_path = input_data.get("data_path")
    if requested_path:
        data_path = Path(requested_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Requested data_path does not exist: {requested_path}")
    else:
        data_path = Path(default_eld_path)
        if not data_path.exists():
            import kagglehub

            dataset_root = Path(
                kagglehub.dataset_download("eduardojst10/electricityloaddiagrams20112014")
            )
            data_path = dataset_root / "df_kwh_adjusted.csv"
            if not data_path.exists():
                raise FileNotFoundError(f"Expected Kaggle dataset file at {data_path}")

    output_dir = input_data.get("output_dir", "artifacts/flash_eld_run")
    args = Namespace(
        data_path=str(data_path),
        delimiter=",",
        series_axis="auto",
        output_dir=output_dir,
        device=input_data.get("device", "cuda"),
        seed=int(input_data.get("seed", 7)),
        experiment_name=input_data.get(
            "experiment_name",
            "ELD LSTM record-level LiRA offline MSE (Kaggle fallback, Flash)",
        ),
        paper_preset=input_data.get(
            "paper_preset", "eld_lstm_record_lira_offline_mse_kaggle_fallback"
        ),
        attack_setting="offline",
        signal="mse",
        num_runs=int(input_data.get("num_runs", 1)),
        L=int(input_data.get("L", 100)),
        H=int(input_data.get("H", 20)),
        epochs=int(input_data.get("epochs", 50)),
        patience=int(input_data.get("patience", 3)),
        batch_size=int(input_data.get("batch_size", 1024)),
        lr=float(input_data.get("lr", 1e-3)),
        num_shadow=int(input_data.get("num_shadow", 64)),
        num_train_users=20,
        num_val_users=20,
        num_test_users=20,
        num_aux_users=40,
        train_user_ratio=0.2,
        val_user_ratio=0.2,
        test_user_ratio=0.2,
        max_users=100,
        require_exact_users=100,
        synthetic=False,
        synthetic_steps=512,
        synthetic_users=12,
        eld_raw_format=True,
        eld_aggregate_factor=4,
        eld_min_valid_steps=15000,
        eld_min_mean=200.0,
        eld_max_mean=2000.0,
        eld_truncate_length=15000,
        eld_mean_filter_mode="middle_n",
        eld_target_users=100,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = run_experiment(args)
    return result
