"""
Tests for SessionState and DatasetContext (Phase 3).
"""
from pathlib import Path
import pandas as pd
import pytest

from proton_agent.context import (
    DatasetContext,
    DatasetNotFoundError,
    InvalidDatasetError,
    NoDatasetLoadedError,
    SessionState,
)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "age": [25, 30, None, 45],
            "name": ["Alice", "Bob", "Charlie", None],
            "score": [85.5, 90.0, 78.0, 92.5],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_dataset_context_properties(sample_csv: Path):
    df = pd.read_csv(sample_csv)
    ctx = DatasetContext(filepath=sample_csv, df=df)

    assert ctx.rows == 4
    assert ctx.columns_count == 3
    assert ctx.shape == (4, 3)
    assert ctx.columns == ["age", "name", "score"]
    assert ctx.total_missing == 2
    assert ctx.missing_counts["age"] == 1
    assert ctx.missing_counts["name"] == 1
    assert ctx.missing_counts["score"] == 0

    col_info = ctx.get_column_info()
    assert len(col_info) == 3
    assert col_info[0]["name"] == "age"
    assert col_info[0]["missing"] == 1
    assert col_info[0]["non_null"] == 3


def test_session_state_load_success(sample_csv: Path):
    session = SessionState()
    assert session.current_dataset is None

    dataset = session.load_dataset(sample_csv)
    assert dataset.rows == 4
    assert session.current_dataset is not None
    assert session.current_dataset.name == "sample.csv"
    assert len(session.history) == 1


def test_session_state_load_not_found(tmp_path: Path):
    session = SessionState()
    non_existent = tmp_path / "does_not_exist.csv"

    with pytest.raises(DatasetNotFoundError):
        session.load_dataset(non_existent)


def test_session_state_load_invalid(tmp_path: Path):
    session = SessionState()
    bad_file = tmp_path / "invalid.csv"
    bad_file.write_bytes(b"\x00\x00\x00\x00\xff\xfe")

    invalid_path = tmp_path / "somedir"
    invalid_path.mkdir()

    with pytest.raises(DatasetNotFoundError):
        session.load_dataset(invalid_path)


def test_session_state_inspect_without_dataset():
    session = SessionState()
    with pytest.raises(NoDatasetLoadedError):
        session.inspect_dataset()


def test_session_state_clear(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)
    assert session.current_dataset is not None

    session.clear()
    assert session.current_dataset is None
    assert len(session.history) == 0


def test_session_state_load_excel(tmp_path: Path):
    session = SessionState()
    xlsx_file = tmp_path / "data.xlsx"
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
    df.to_excel(xlsx_file, index=False)

    dataset = session.load_dataset(xlsx_file)
    assert dataset.rows == 3
    assert dataset.columns_count == 2
    assert dataset.columns == ["col1", "col2"]
