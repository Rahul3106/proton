"""
Phase E Tests: Data Cleaning Upgrades (Outliers, Typo Standardization, Datetime, and Text NLP Features).
"""
from pathlib import Path
import pandas as pd
import pytest

from proton_agent.context import SessionState
from proton_agent.tools import CleanDatasetTool


@pytest.fixture
def complex_dirty_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "complex_dirty.csv"
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "income": [30000.0, 32000.0, 35000.0, 28000.0, 31000.0, 33000.0, 29000.0, 34000.0, 1000000.0, -50000.0],
            "gender": ["Male", "male", "MALE", "Female", "female", "feamle", "Female", "male", "female", "Male"],
            "signup_date": [
                "2023-01-15 08:30:00",
                "2023-02-20 14:00:00",
                "2023-03-05 09:15:00",
                "2023-04-12 18:45:00",
                "2023-05-18 11:20:00",
                "2023-06-25 16:10:00",
                "2023-07-04 10:00:00",
                "2023-08-19 20:30:00",
                "2023-09-22 13:40:00",
                "2023-10-30 07:50:00",
            ],
            "customer_feedback": [
                "The product quality is good and service was fast and great!",
                "Terrible experience, very slow delivery and broken items.",
                "Excellent support team, smooth experience and happy with purchase.",
                "Bad quality and terrible support, fail on all fronts.",
                "Amazing results and wonderful service, love this product!",
                "Poor performance with multiple bugs and error issues.",
                "Clean interface, fast loading and good features.",
                "Worst product ever, dirty packaging and awful delay.",
                "Great experience overall, positive outcome!",
                "Broken on arrival, terrible packaging and dirty box.",
            ],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_outlier_handling_clip(complex_dirty_csv: Path):
    session = SessionState()
    session.load_dataset(complex_dirty_csv)

    tool = CleanDatasetTool()
    # 1. Preview
    prev = tool.execute(session, dry_run=True, outlier_strategy="clip")
    assert prev.success is True
    assert prev.data["dry_run"] is True
    assert "income" in prev.data["outliers_detected"]
    assert prev.data["outliers_detected"]["income"] >= 2

    # 2. Apply
    res = tool.execute(session, dry_run=False, outlier_strategy="clip")
    assert res.success is True
    df_clean = session.current_dataset.df
    # Incomes clipped to reasonable bounds
    assert df_clean["income"].max() < 500000.0
    assert df_clean["income"].min() > -10000.0


def test_outlier_handling_remove(complex_dirty_csv: Path):
    session = SessionState()
    session.load_dataset(complex_dirty_csv)

    tool = CleanDatasetTool()
    res = tool.execute(session, dry_run=False, outlier_strategy="remove")
    assert res.success is True
    df_clean = session.current_dataset.df
    assert len(df_clean) < 10


def test_category_typo_standardization(complex_dirty_csv: Path):
    session = SessionState()
    session.load_dataset(complex_dirty_csv)

    tool = CleanDatasetTool()
    # 1. Dry run preview
    prev = tool.execute(session, dry_run=True)
    assert prev.success is True
    assert "gender" in prev.data["typos_detected"]
    mapping = prev.data["typos_detected"]["gender"]
    assert any(k.lower() in ("male", "feamle") for k in mapping.keys())

    # 2. Apply
    res = tool.execute(session, dry_run=False)
    assert res.success is True
    df_clean = session.current_dataset.df
    unique_genders = set(df_clean["gender"].unique())
    # Should be reduced to only 2 clean canonical categories
    assert len(unique_genders) <= 2


def test_datetime_feature_extraction(complex_dirty_csv: Path):
    session = SessionState()
    session.load_dataset(complex_dirty_csv)

    tool = CleanDatasetTool()
    res = tool.execute(session, dry_run=False, extract_datetime=True)
    assert res.success is True
    df_clean = session.current_dataset.df

    # Original dropped, temporal attributes extracted
    assert "signup_date" not in df_clean.columns
    assert "signup_date_year" in df_clean.columns
    assert "signup_date_month" in df_clean.columns
    assert "signup_date_day" in df_clean.columns
    assert "signup_date_day_of_week" in df_clean.columns
    assert "signup_date_is_weekend" in df_clean.columns
    assert "signup_date_days_since_ref" in df_clean.columns


def test_text_feature_extraction(complex_dirty_csv: Path):
    session = SessionState()
    session.load_dataset(complex_dirty_csv)

    tool = CleanDatasetTool()
    res = tool.execute(session, dry_run=False, extract_text_nlp=True)
    assert res.success is True
    df_clean = session.current_dataset.df

    # Original text dropped, NLP features extracted
    assert "customer_feedback" not in df_clean.columns
    assert "customer_feedback_char_count" in df_clean.columns
    assert "customer_feedback_word_count" in df_clean.columns
    assert "customer_feedback_uppercase_ratio" in df_clean.columns
    assert "customer_feedback_digit_ratio" in df_clean.columns
    assert "customer_feedback_sentiment_polarity" in df_clean.columns


def test_clean_dataset_all_capabilities_end_to_end(complex_dirty_csv: Path):
    session = SessionState()
    session.load_dataset(complex_dirty_csv)

    tool = CleanDatasetTool()
    res = tool.execute(session, dry_run=False, outlier_strategy="clip")
    assert res.success is True

    df_clean = session.current_dataset.df
    # All columns must be numeric or clean discrete categories
    for col in df_clean.columns:
        assert pd.api.types.is_numeric_dtype(df_clean[col]) or df_clean[col].nunique() <= 3
    assert len(session.transformations) >= 3
