#!/usr/bin/env python3
"""Pure-Python launcher for OpenSTL deterministic WeatherBench baselines.

Reads ``configs/openstl_baseline.yaml`` (or ``--config``) for roots and hyperparameters,
then invokes ``scripts/train_openstl_no_tensorboard.py`` with cwd set to ``openstl_root``.

No Bash is required.

Example:

    cd /path/to/Weather-pred
    python scripts/run_deterministic.py --backbone simvp --mode smoke
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml


_DEFAULT_CONFIG = "configs/openstl_baseline.yaml"


def _as_path(root: Path, raw: Any) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p
    return root / p


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk.

    Args:
        path: Absolute path to a YAML config file.

    Returns:
        Parsed dict (non-empty YAML document required).
    """

    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data


def _set_limit_env(env: dict[str, str], limits: Mapping[str, Any]) -> None:
    """Copy batch-cap integers into Lightning env knobs when set."""

    key_map = (
        ("train_batches", "OPENSTL_LIMIT_TRAIN_BATCHES"),
        ("val_batches", "OPENSTL_LIMIT_VAL_BATCHES"),
        ("test_batches", "OPENSTL_LIMIT_TEST_BATCHES"),
    )
    for yaml_key, env_key in key_map:
        raw = limits.get(yaml_key)
        if raw is None or raw == "":
            continue
        try:
            v = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"limits.{yaml_key} must be int or null") from exc
        if v > 0:
            env[env_key] = str(v)


def _repo_anchor(config_path: Path) -> Path:
    """Infer Weather-pred root from YAML location.

    ASSUMPTION: Defaults live under ``<repo>/configs/*.yaml``. If not, anchors at cwd.

    Args:
        config_path: Path to the YAML file passed via ``--config``.

    Returns:
        Resolved directory used to resolve relative ``project_root`` / ``openstl_root``.
    """

    d = config_path.resolve().parent
    if d.name == "configs":
        return d.parent
    return Path.cwd().resolve()


def build_command(
    cfg: Mapping[str, Any],
    backbone: str,
    mode: str,
    *,
    anchor: Path,
    gpus_override: list[int] | None = None,
    cuda_visible_override: str | None = None,
) -> tuple[list[str], dict[str, str], Path, Path]:
    """Assemble launcher argv, environ patch, log path, and resolved OpenSTL root.

    Args:
        cfg: Loaded ``openstl_baseline.yaml`` mapping.
        backbone: Either ``simvp`` or ``convlstm``.
        mode: Either ``smoke`` or ``full``.
        anchor: Repository root inferred from YAML path (see :func:`_repo_anchor`).
        gpus_override: If set, overrides ``hardware.gpus`` Lightning device IDs.
        cuda_visible_override: If set (non-empty), sets ``CUDA_VISIBLE_DEVICES``.

    Returns:
        Tuple ``(exec_argv, environ_patch, log_path, openstl_root)``.
    """

    proj = anchor
    if cfg.get("project_root") is not None:
        proj = _as_path(anchor, cfg["project_root"]).resolve()
    openstl_root = _as_path(anchor, cfg["openstl_root"]).resolve()
    launcher_rel = cfg.get("launcher", "scripts/train_openstl_no_tensorboard.py")
    launcher = _as_path(proj, launcher_rel).resolve()

    if not launcher.is_file():
        raise FileNotFoundError(f"Launcher not found: {launcher}")
    if not openstl_root.is_dir():
        raise FileNotFoundError(f"openstl_root not found: {openstl_root}")

    bb_map = cfg.get("backbones", {})
    if backbone not in bb_map:
        raise KeyError(f"Unknown backbone {backbone!r}; expected one of {list(bb_map)}")

    bb = bb_map[backbone]
    method = str(bb["method"])
    cfg_rel = str(bb["cfg"])
    cfg_abs = (openstl_root / cfg_rel).resolve()
    if not cfg_abs.is_file():
        raise FileNotFoundError(f"OpenSTL config not found: {cfg_abs}")

    train = cfg.get("train", {})
    if mode == "smoke":
        epochs_key = "epochs_smoke"
        postfix = "smoke"
    elif mode == "fast":
        epochs_key = "epochs_fast"
        postfix = "det_baseline_fast"
    else:
        epochs_key = "epochs_full"
        postfix = "det_baseline"
    epochs = int(train[epochs_key])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ex_name = f"{backbone}_{postfix}_{stamp}"

    batch_size = int(train.get("batch_size", 16))
    val_batch_size = int(train.get("val_batch_size", 16))
    use_fp16 = bool(train.get("use_fp16", False))
    if mode == "fast":
        batch_size = int(train.get("batch_size_fast", batch_size))
        val_batch_size = int(train.get("val_batch_size_fast", val_batch_size))
        use_fp16 = bool(train.get("use_fp16_fast", use_fp16))

    work_rel = cfg.get("work_dirs", "results/openstl_work_dirs")
    log_rel = cfg.get("baseline_logs", "results/baseline_logs")
    work_dir = _as_path(proj, work_rel).resolve()
    log_dir = _as_path(proj, log_rel).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{ex_name}.log"

    args = [
        sys.executable,
        "-u",
        str(launcher),
        "-d",
        str(cfg["dataname"]),
        "--data_root",
        str(cfg.get("data_root", "./data")),
        "-m",
        method,
        "-c",
        str(cfg_abs),
        "--ex_name",
        ex_name,
        "--res_dir",
        str(work_dir),
        "-e",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--val_batch_size",
        str(val_batch_size),
        "--num_workers",
        str(int(train.get("num_workers", 4))),
        "--seed",
        str(int(train.get("seed", 42))),
        "--log_step",
        str(int(train.get("log_step", 1))),
    ]
    if use_fp16:
        args.append("--fp16")

    hw = cfg.get("hardware") or {}
    gpus = list(gpus_override) if gpus_override is not None else hw.get("gpus", [0])
    if isinstance(gpus, (int, float)):
        gpus = [gpus]
    gpus_int = [int(x) for x in gpus]
    args.extend(["--gpus", *[str(g) for g in gpus_int]])

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["OPENSTL_ROOT"] = str(openstl_root)
    if cuda_visible_override is not None and str(cuda_visible_override).strip() != "":
        cuda_vis = str(cuda_visible_override).strip()
    else:
        raw_vis = hw.get("cuda_visible_devices")
        cuda_vis = None if raw_vis in (None, "") else str(raw_vis).strip()
    if cuda_vis:
        env["CUDA_VISIBLE_DEVICES"] = cuda_vis
    _set_limit_env(env, cfg.get("limits", {}) or {})

    return args, env, log_path, openstl_root


def _stream_subprocess(
    args: Iterable[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
) -> int:
    """Run child process, streaming combined stdout/stderr to console and log file.

    Args:
        args: argv list including executable as ``args[0]``.
        cwd: Working directory for the child (OpenSTL root).
        env: Full environment mapping.
        log_path: Destination plaintext log.

    Returns:
        Process exit code.
    """

    proc = subprocess.Popen(
        list(args),
        cwd=str(cwd),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8") as lf:
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
            lf.flush()
    return int(proc.wait())


def main() -> None:
    """CLI entry: parse args, apply seeds, run OpenSTL launcher."""

    here = Path(__file__).resolve().parent
    default_cfg = here.parent / _DEFAULT_CONFIG

    parser = argparse.ArgumentParser(
        description="Run OpenSTL deterministic baseline (YAML paths, Python only).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help=f"YAML config (default: {_DEFAULT_CONFIG} under Weather-pred).",
    )
    parser.add_argument(
        "--backbone",
        choices=("simvp", "convlstm", "tau"),
        required=True,
        help="SimVP, TAU, or ConvLSTM preset from YAML ``backbones``.",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full", "fast"),
        default="full",
        help="smoke: 1 epoch; full: epochs_full; fast: epochs_fast (larger batch + fp16).",
    )
    parser.add_argument(
        "--cuda-visible-device",
        type=str,
        default=None,
        help="Single physical GPU index for CUDA_VISIBLE_DEVICES (concurrent jobs). Overrides YAML.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="Lightning device ids passed to OpenSTL (--gpus). Default: YAML hardware.gpus.",
    )
    args = parser.parse_args()

    cfg_path = args.config
    if not cfg_path.is_file():
        raise FileNotFoundError(f"--config not found: {cfg_path}")

    cfg = _load_yaml(cfg_path)
    anchor = _repo_anchor(cfg_path)

    seed = int(cfg.get("train", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    proj = anchor
    if cfg.get("project_root") is not None:
        proj = _as_path(anchor, cfg["project_root"]).resolve()
    os.chdir(proj)

    exec_argv, env, log_path, openstl_root = build_command(
        cfg,
        args.backbone,
        args.mode,
        anchor=anchor,
        gpus_override=args.gpus,
        cuda_visible_override=args.cuda_visible_device,
    )

    print("OpenSTL deterministic (Python launcher)")
    print(f"  project_root: {proj}")
    print(f"  openstl_root: {openstl_root}")
    print(f"  backbone:     {args.backbone}")
    print(f"  mode:         {args.mode}")
    if env.get("CUDA_VISIBLE_DEVICES"):
        print(f"  CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
    print(f"  log_file:    {log_path}")
    sys.stdout.flush()

    rc = _stream_subprocess(exec_argv, cwd=openstl_root, env=env, log_path=log_path)
    sys.exit(rc)


if __name__ == "__main__":
    main()
