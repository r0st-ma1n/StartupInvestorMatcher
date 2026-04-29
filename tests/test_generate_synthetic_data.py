from __future__ import annotations

import csv
import shutil
import uuid
from pathlib import Path

import pytest

from app.services import load_investors_csv, load_startups_csv
from scripts.generate_synthetic_data import generate_dataset


def test_generate_synthetic_data_creates_compatible_csvs() -> None:
    output_dir = _workspace_temp_dir()

    summary = generate_dataset(output_dir=output_dir, size="small", overwrite=False, seed=7)

    startups_path = output_dir / "startups.csv"
    investors_path = output_dir / "investors.csv"
    ground_truth_path = output_dir / "ground_truth.csv"

    assert startups_path.exists()
    assert investors_path.exists()
    assert ground_truth_path.exists()
    assert summary["startup_count"] == 12
    assert summary["investor_count"] == 24
    assert summary["ground_truth_count"] >= 12

    startups = load_startups_csv(startups_path)
    investors = load_investors_csv(investors_path)

    assert len(startups) == 12
    assert len(investors) == 24

    with ground_truth_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows
    assert set(rows[0]) == {"startup_id", "investor_id"}

    _cleanup_temp_dir(output_dir)


def test_generate_synthetic_data_requires_overwrite_for_existing_files() -> None:
    output_dir = _workspace_temp_dir()

    generate_dataset(output_dir=output_dir, size="small", overwrite=False, seed=1)

    with pytest.raises(FileExistsError):
        generate_dataset(output_dir=output_dir, size="small", overwrite=False, seed=2)

    _cleanup_temp_dir(output_dir)


def test_generate_synthetic_data_overwrite_recreates_files() -> None:
    output_dir = _workspace_temp_dir()

    generate_dataset(output_dir=output_dir, size="small", overwrite=False, seed=1)
    first_content = (output_dir / "startups.csv").read_text(encoding="utf-8")

    generate_dataset(output_dir=output_dir, size="small", overwrite=True, seed=2)
    second_content = (output_dir / "startups.csv").read_text(encoding="utf-8")

    assert first_content != second_content

    _cleanup_temp_dir(output_dir)


def test_generate_synthetic_data_is_reproducible_with_seed() -> None:
    left_dir = _workspace_temp_dir()
    right_dir = _workspace_temp_dir()

    generate_dataset(output_dir=left_dir, size="small", overwrite=False, seed=11)
    generate_dataset(output_dir=right_dir, size="small", overwrite=False, seed=11)

    assert (left_dir / "startups.csv").read_text(encoding="utf-8") == (
        right_dir / "startups.csv"
    ).read_text(encoding="utf-8")
    assert (left_dir / "investors.csv").read_text(encoding="utf-8") == (
        right_dir / "investors.csv"
    ).read_text(encoding="utf-8")
    assert (left_dir / "ground_truth.csv").read_text(encoding="utf-8") == (
        right_dir / "ground_truth.csv"
    ).read_text(encoding="utf-8")

    _cleanup_temp_dir(left_dir)
    _cleanup_temp_dir(right_dir)


def _workspace_temp_dir() -> Path:
    temp_dir = Path("tests") / "_tmp" / uuid.uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _cleanup_temp_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
