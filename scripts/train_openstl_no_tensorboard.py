#!/usr/bin/env python3
# Copyright helpers: CAIRI (OpenSTL) original train flow; launcher lives in Weather-pred.
"""Run OpenSTL ``tools/train.py`` flow with Lightning **CSV** logger (no TensorBoard).

Default OpenSTL uses TensorBoard; on some environments (protobuf / tensorboard / numpy 2)
that path crashes. This **launcher** in Weather-pred:

- swaps in ``CSVLogger`` (no edits inside OpenSTL).
- replaces WeatherBench-style test-set ``numpy`` concatenation with streaming aggregation
  (numerically equivalent MAE/MSE/RMSE) to avoid RAM OOM kills after Lightning ``Testing``
  reaches ~100%.

Invocation: working directory is forced to ``OPENSTL_ROOT`` (OpenSTL clone root).

Usage mirrors ``python tools/train.py ...`` exactly.
"""
from __future__ import annotations

import os
import os.path as osp
import sys
import warnings

warnings.filterwarnings("ignore")



def _paper_eval_banner_deterministic(
    *,
    mae_val: float,
    mse_val: float,
    rmse_val: float,
    dataname: str,
) -> str:
    """Human-readable bloc for deterministic test metrics + paper-oriented notes."""
    if "z500" in dataname:
        variable = "z500 (geopotential @ 500 hPa)"
        unit = r"m$^2$s$^{-2}$"
    else:
        variable = "t2m (2 m temperature)"
        unit = "K"

    lines = [
        "=" * 78,
        "PAPER EVAL — OpenSTL deterministic baseline (physical units)",
        "=" * 78,
        f"  dataname   : {dataname}",
        f"  variable   : {variable}",
        "  scaling    : denormalized (WeatherBench train mean / std)",
        "-" * 78,
        "  Point forecasts (always reported):",
        f"    MAE      ({unit}) : {mae_val:.6f}",
        f"    RMSE     ({unit}) : {rmse_val:.6f}",
        f"    MSE      ({unit}$^2$) : {mse_val:.6f}",
        "-" * 78,
        "  Probabilistic metrics (papers with Gaussian forecasts — project convention):",
        "    CRPS     : N/A  (needs ProbWrapper μ, log σ^2)",
        "    NLL      : N/A  (needs ProbWrapper μ, log σ^2)",
        "    ECE      : N/A  (needs predictive distribution — 10 equal-width CI bins per rules)",
        "  Use src/metrics.py + src/calibration.py after probabilistic inference.",
        "=" * 78,
    ]
    return "\n".join(lines)


def _apply_streaming_weather_test_aggregate() -> None:
    """Patch OpenSTL ``Base_method`` weather tests to aggregate metrics incrementally.

    The upstream ``test_step`` materializes predictions for **every** test batch then
    ``on_test_epoch_end`` concatenates them. On WeatherBench (~17k batches×12 frames)
    that multi-gigabyte array can trigger OOM/kill **after** Lightning reports
    Testing 100%. This patch computes the same scalar MAE/MSE/RMSE as ``metric()`` for
    ``spatial_norm=True`` and ``channel_names is None``.

    ASSUMPTION: Single-GPU Lightning test runs (OpenSTL defaults). Not applied when
    per-channel grouping is configured.
    """

    import numpy as np
    import os.path as osp

    from openstl.methods.base_method import Base_method as BM
    from openstl.utils import check_dir, print_log

    orig_test_step = BM.test_step
    orig_epoch_end = BM.on_test_epoch_end

    def _eligible(inst: BM) -> bool:
        sn = getattr(inst, "spatial_norm", False)
        ch = getattr(inst, "channel_names", None)
        return sn and ch is None and "weather" in str(getattr(inst.hparams, "dataname", ""))

    def _on_test_epoch_start_patch(self):
        super(BM, self).on_test_epoch_start()
        if _eligible(self):
            self._stream_test_sum_abs_chw = None
            self._stream_test_sum_sq_chw = None
            self._stream_test_tot_bt = 0.0

    BM.on_test_epoch_start = _on_test_epoch_start_patch

    def test_step_patch(self, batch, batch_idx):
        if not _eligible(self):
            return orig_test_step(self, batch, batch_idx)
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        p = pred_y.detach().cpu().numpy().astype(np.float64)
        tr = batch_y.detach().cpu().numpy().astype(np.float64)

        hm = getattr(self.hparams, "test_mean", None)
        hs = getattr(self.hparams, "test_std", None)
        if hm is not None and hs is not None:
            p = np.asarray(p) * hs + hm
            tr = np.asarray(tr) * hs + hm

        norm = float(np.prod(np.array(p.shape[-3:], dtype=np.int64)))
        bt = float(np.prod(np.array(p.shape[:2])))
        sum_abs_chw = np.sum(np.abs(p - tr) / norm, axis=(0, 1))
        sum_sq_chw = np.sum((p - tr) ** 2 / norm, axis=(0, 1))

        if getattr(self, "_stream_test_sum_abs_chw", None) is None:
            self._stream_test_sum_abs_chw = np.zeros_like(sum_abs_chw)
            self._stream_test_sum_sq_chw = np.zeros_like(sum_sq_chw)
            self._stream_test_tot_bt = 0.0
        self._stream_test_sum_abs_chw += sum_abs_chw
        self._stream_test_sum_sq_chw += sum_sq_chw
        self._stream_test_tot_bt += bt
        return {}

    BM.test_step = test_step_patch

    def on_test_epoch_end_patch(self):
        if not (
            _eligible(self)
            and getattr(self, "_stream_test_sum_abs_chw", None) is not None
            and getattr(self, "_stream_test_tot_bt", 0.0) > 0.0
        ):
            orig_epoch_end(self)
            return
        bt = float(self._stream_test_tot_bt)
        mae_val = float(np.sum(self._stream_test_sum_abs_chw / bt))
        mse_val = float(np.sum(self._stream_test_sum_sq_chw / bt))
        rmse_val = float(np.sqrt(np.sum(self._stream_test_sum_sq_chw / bt)))

        vals = {"mae": mae_val, "mse": mse_val, "rmse": rmse_val}
        parts = []
        for name in getattr(self, "metric_list", ["mse", "rmse", "mae"]):
            if name in vals:
                parts.append(f"{name}:{vals[name]}")
        eval_log = ", ".join(parts)
        paper_block = _paper_eval_banner_deterministic(
            mae_val=mae_val,
            mse_val=mse_val,
            rmse_val=rmse_val,
            dataname=str(getattr(self.hparams, "dataname", "unknown")),
        )

        if self.trainer.is_global_zero:
            # Escape Lightning/tqdm carriage-return so the metric line is not prefixed on the
            # same row (first character of ``mse:`` appears as ``se:`` in some consoles).
            print(file=sys.stderr, flush=True)
            print(file=sys.stdout, flush=True)
            print_log(paper_block)
            print_log("compact (grep-friendly): " + eval_log)

        folder_path = check_dir(osp.join(self.hparams.save_dir, "saved"))

        metrics_vec = np.array([mae_val, mse_val], dtype=np.float32)
        if self.trainer.is_global_zero:
            np.save(osp.join(folder_path, "metrics.npy"), metrics_vec)
            summary_path = osp.join(folder_path, "paper_eval_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(paper_block + "\n\n")
                f.write("compact: " + eval_log + "\n")

            def _unlink(name: str) -> None:
                fp = osp.join(folder_path, name)
                if osp.isfile(fp):
                    try:
                        os.remove(fp)
                    except OSError:
                        pass

            for np_data in ("inputs", "trues", "preds"):
                _unlink(np_data + ".npy")

    BM.on_test_epoch_end = on_test_epoch_end_patch


def _trainer_limit_kw() -> dict:
    """Build optional Lightning Trainer ``limit_*_batches`` kwargs from env.

    Reads positive ints from ``OPENSTL_LIMIT_TRAIN_BATCHES``,
    ``OPENSTL_LIMIT_VAL_BATCHES``, ``OPENSTL_LIMIT_TEST_BATCHES``.
    Lightning limits **optimization steps**. Approximate ceiling on train
    samples scanned per epoch: ``limit_train_batches * batch_size``.

    ASSUMPTION: Env values are ASCII integers.

    Returns:
        Keyword arguments for Lightning ``Trainer`` (possibly empty).
    """

    out: dict[str, int] = {}
    for env_key, trainer_key in (
        ("OPENSTL_LIMIT_TRAIN_BATCHES", "limit_train_batches"),
        ("OPENSTL_LIMIT_VAL_BATCHES", "limit_val_batches"),
        ("OPENSTL_LIMIT_TEST_BATCHES", "limit_test_batches"),
    ):
        raw = os.environ.get(env_key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            val = int(str(raw).strip(), 10)
        except ValueError:
            continue
        if val > 0:
            out[trainer_key] = val
    return out


def _register_weather_z500_dataset() -> None:
    """Register z500 WeatherBench preset without editing the OpenSTL tree."""
    from openstl.datasets import dataset_parameters

    dataset_parameters["weather_z500_5_625"] = {
        "in_shape": [12, 1, 32, 64],
        "pre_seq_length": 12,
        "aft_seq_length": 12,
        "total_length": 24,
        "data_name": "z",
        "levels": [500],
        "train_time": ["1979", "2015"],
        "val_time": ["2016", "2016"],
        "test_time": ["2017", "2018"],
        "metrics": ["mse", "rmse", "mae"],
    }

    import openstl.utils as utils_mod
    import openstl.utils.parser as parser_mod

    _orig_create = parser_mod.create_parser

    def _create_parser_with_z500():
        parser = _orig_create()
        for action in parser._actions:
            if action.dest == "dataname" and action.choices is not None:
                if "weather_z500_5_625" not in action.choices:
                    action.choices = list(action.choices) + ["weather_z500_5_625"]
        return parser

    parser_mod.create_parser = _create_parser_with_z500
    utils_mod.create_parser = _create_parser_with_z500


def main() -> None:
    _here = osp.dirname(osp.abspath(__file__))
    openstl_root = osp.abspath(
        os.environ.get(
            "OPENSTL_ROOT",
            osp.join(_here, "..", "..", "OpenSTL"),
        )
    )
    if not osp.isdir(openstl_root):
        raise FileNotFoundError(f"OPENSTL_ROOT not found: {openstl_root}")
    os.chdir(openstl_root)
    if openstl_root not in sys.path:
        sys.path.insert(0, openstl_root)

    _register_weather_z500_dataset()
    _apply_streaming_weather_test_aggregate()

    from lightning import Trainer
    from openstl.api.exp import BaseExperiment
    from openstl.utils import (create_parser, default_parser, get_dist_info,
                               load_config, update_config)

    def _init_trainer_no_tb(self, args, callbacks, strategy="auto"):
        from lightning.pytorch.loggers import CSVLogger
        from lightning.pytorch.callbacks import EarlyStopping

        logger = CSVLogger(save_dir=self.save_dir, name="lightning_csv")
        limit_kw = _trainer_limit_kw()

        # Early stopping: patience=10, min_delta=1e-4, min 15 epochs
        # monitors the same val_loss used by BestCheckpointCallback
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-4,
            mode="min",
            verbose=True,
        )
        # min_epochs guard: Lightning's EarlyStopping won't fire before epoch 15
        early_stop.stopped_epoch = 0  # will be updated by Lightning
        import lightning.pytorch.callbacks as _lc
        _lc_es = early_stop

        # Wrap to enforce min_epochs=15
        class _GuardedEarlyStopping(EarlyStopping):
            _min_epochs = 15

            def on_validation_end(self, trainer, pl_module):
                if trainer.current_epoch < self._min_epochs:
                    return
                super().on_validation_end(trainer, pl_module)

        guarded_es = _GuardedEarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-4,
            mode="min",
            verbose=True,
        )

        return Trainer(
            devices=args.gpus,
            max_epochs=args.epoch,
            strategy=strategy,
            accelerator="gpu",
            callbacks=callbacks + [guarded_es],
            logger=logger,
            enable_progress_bar=True,
            **limit_kw,
        )

    args = create_parser().parse_args()
    config = args.__dict__

    cfg_path = (
        osp.join("./configs", args.dataname, f"{args.method}.py")
        if args.config_file is None else args.config_file
    )
    if args.overwrite:
        config = update_config(config, load_config(cfg_path), exclude_keys=["method"])
    else:
        loaded_cfg = load_config(cfg_path)
        config = update_config(
            config,
            loaded_cfg,
            exclude_keys=[
                "method",
                "val_batch_size",
                "drop_path",
                "warmup_epoch",
            ],
        )
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]

    BaseExperiment._init_trainer = _init_trainer_no_tb

    print(">" * 35 + " training " + "<" * 35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    # Always test from best.ckpt (consistent with ProbWrapper "best by val").
    best_ckpt = osp.join(exp.save_dir, "checkpoints", "best.ckpt")
    if osp.isfile(best_ckpt):
        import torch as _torch
        ckpt = _torch.load(best_ckpt, map_location="cpu", weights_only=False)
        exp.method.load_state_dict(ckpt["state_dict"])
        print(f"Loaded best.ckpt for test: {best_ckpt}")
    else:
        print(f"WARNING: best.ckpt not found at {best_ckpt}, testing with last weights.")

    if rank == 0:
        print(">" * 35 + " testing  " + "<" * 35)
    exp.test()


if __name__ == "__main__":
    main()
