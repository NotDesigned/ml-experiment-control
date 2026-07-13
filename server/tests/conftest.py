from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

A1_RUN_DIR = (FIXTURES / "runs" / "fusion-len256-gate-h100-20260711"
              / "elf-a1-frozen-t5-l256-s42-h100-v1")
SMOKE_RUN_DIR = (FIXTURES / "runs" / "backend-smoke-slurm-probe-20260712T0105"
                 / "elf-smoke-slurm-l40s-probe-20260712T0105")

# A1 ground truth (UTC): scheduler observed 14:31:48, worker status written
# 12:28:33, last train_metrics record ~13:57 (step 3700).
A1_SCHEDULER_TS = 1783780308.755999   # 2026-07-11T14:31:48.755999Z
A1_WORKER_TS = 1783772913.149907      # 2026-07-11T12:28:33.149907Z


@pytest.fixture
def a1_run_dir() -> Path:
    assert A1_RUN_DIR.is_dir(), "A1 fixture missing"
    return A1_RUN_DIR


@pytest.fixture
def smoke_run_dir() -> Path:
    assert SMOKE_RUN_DIR.is_dir(), "smoke fixture missing"
    return SMOKE_RUN_DIR
