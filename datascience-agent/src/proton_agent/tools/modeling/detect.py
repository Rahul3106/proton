"""
Detect Problem Type Tool: identifies ML task type (classification vs regression) and target variable.
"""
from __future__ import annotations

from typing import Any
import pandas as pd

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


def infer_target_column(df: pd.DataFrame) -> str:
    """
    Intelligently identifies the most likely target column from a DataFrame,
    avoiding ID columns, URLs, hashes, and filenames.
    """
    columns = list(df.columns)
    if not columns:
        return ""

    # 1. Exact match for target keywords
    likely_exact = ["target", "label", "class", "outcome", "churn", "status", "price", "sales", "survived", "species", "result", "y", "category", "target_class"]
    for cand in likely_exact:
        for c in columns:
            if c.lower() == cand:
                return c

    # 2. Check for columns ending with or containing target/label/outcome/class (e.g. adaptation_label_TARGET)
    for c in reversed(columns):
        c_lower = c.lower()
        if c_lower.endswith("_target") or c_lower.endswith("_label") or c_lower.startswith("target_") or c_lower.startswith("label_"):
            return c
        if "target" in c_lower or "label" in c_lower or "outcome" in c_lower:
            return c

    # 3. Filter out columns that are clearly URLs, IDs, hashes, timestamps, filenames, or raw text excerpts
    bad_keywords = ("id", "index", "url", "link", "hash", "unnamed", "email", "uuid", "path", "filename", "timestamp", "guid", "href", "excerpt", "text", "raw")
    suitable_cols = []
    for c in reversed(columns):
        c_lower = c.lower()
        if any(c_lower == bad or c_lower.startswith(bad + "_") or c_lower.endswith("_" + bad) or bad in c_lower for bad in bad_keywords):
            continue
        # Also skip columns where every single row is a unique text string (unique URLs/text IDs)
        if len(df) > 1 and df[c].nunique() == len(df) and not pd.api.types.is_numeric_dtype(df[c]):
            continue
        suitable_cols.append(c)

    if suitable_cols:
        return suitable_cols[0]

    # Fallback to last non-URL/non-ID column
    for c in reversed(columns):
        c_lower = c.lower()
        if not ("url" in c_lower or "id" in c_lower or "link" in c_lower):
            return c

    return columns[-1]


def infer_problem_type(series: pd.Series) -> tuple[str, str]:
    """
    Accurately infers whether the task is classification or regression.
    Returns (problem_type, detail_message).
    """
    # 1. Non-numeric data types (string, object, category, boolean, etc.) -> Always Classification
    if not pd.api.types.is_numeric_dtype(series):
        unique_vals = int(series.nunique(dropna=True))
        return "classification", f"Target '{series.name}' is a categorical variable with {unique_vals} unique class(es)."

    # 2. Low-cardinality numbers (<= 20 unique values or boolean) -> Classification
    unique_vals = int(series.nunique(dropna=True))
    if unique_vals <= 20:
        return "classification", f"Target '{series.name}' has {unique_vals} distinct discrete class value(s)."

    # 3. Continuous numeric variables (> 20 unique values) -> Regression
    return "regression", f"Target '{series.name}' is a continuous numerical variable."


class DetectProblemTypeTool(BaseTool):
    """Detects target column and determines whether the problem type is classification or regression."""

    name: str = "detect_problem_type"
    description: str = "Infer target variable and determine ML problem type (classification vs regression)."

    def execute(self, session: SessionState, target_column: str | None = None, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df
        if target_column and target_column in df.columns:
            target = target_column
        elif session.target_column and session.target_column in df.columns:
            target = session.target_column
        else:
            target = infer_target_column(df)

        session.target_column = target
        series = df[target]

        problem_type, detail = infer_problem_type(series)
        unique_vals = int(series.nunique(dropna=True))
        num_classes = unique_vals if problem_type == "classification" else None

        # Optional generative AI recommendation from configured Gemini API
        ai_recommendation = None
        try:
            from proton_agent.agent.llm import get_default_llm_client
            llm = get_default_llm_client()
            if hasattr(llm, "api_key") and llm.api_key:
                prompt = (
                    f"Dataset Columns: {', '.join(df.columns[:25])}\n"
                    f"Target column detected: '{target}'\n"
                    f"Data type: {series.dtype}\n"
                    f"Task type: {problem_type}\n\n"
                    "In 1-2 concise sentences, confirm why this target variable and problem type are suitable, and name the top 2 machine learning algorithms best suited for it."
                )
                ai_recommendation = llm.generate_response(prompt).strip()
        except Exception:
            pass

        return ToolResult(
            success=True,
            data={
                "target_column": target,
                "problem_type": problem_type,
                "unique_values_count": unique_vals,
                "num_classes": num_classes,
                "target_dtype": str(series.dtype),
                "detail": detail,
                "ai_recommendation": ai_recommendation,
            },
        )
