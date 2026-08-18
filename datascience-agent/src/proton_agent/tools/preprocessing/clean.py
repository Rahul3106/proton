"""
Clean Dataset Tool: autonomous data cleaning and quality remediation.
Includes duplicate removal, constant dropping, missing value imputation,
outlier clipping/removal, category typo standardization, datetime feature extraction,
and text NLP attribute engineering.
"""
from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


_POSITIVE_WORDS = {
    "good", "great", "excellent", "positive", "love", "like", "best", "happy",
    "success", "clean", "wonderful", "fast", "top", "perfect", "amazing", "smooth"
}
_NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "negative", "hate", "worst", "fail", "failure",
    "dirty", "slow", "error", "broken", "issue", "bug", "delayed", "problem", "awful"
}


def _calc_sentiment(text: str) -> float:
    words = set(text.lower().split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


class CleanDatasetTool(BaseTool):
    """Autonomously cleans active dataset: imputes missing, removes duplicates, handles outliers, fixes typos, extracts datetime and text features."""

    name: str = "clean_dataset"
    description: str = (
        "Automatically clean dataset: impute missing values, remove duplicates, drop constant columns, "
        "clip/remove outliers, standardize category typos, extract datetime features, and compute text NLP features."
    )

    def execute(
        self,
        session: SessionState,
        dry_run: bool = False,
        outlier_strategy: str = "clip",
        fix_typos: bool = True,
        extract_datetime: bool = True,
        extract_text_nlp: bool = True,
        drop_datetime_original: bool = True,
        drop_text_original: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df.copy()
        initial_rows = len(df)
        initial_cols = len(df.columns)
        pending_changes: List[Dict[str, Any]] = []

        # ── 1. Detect and remove duplicate rows ───────────────────────────────
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            pending_changes.append({
                "type": "remove_duplicates",
                "description": f"Remove {dup_count:,} duplicate row(s)",
                "impact": f"Rows: {initial_rows} → {initial_rows - dup_count}",
            })
            if not dry_run:
                df = df.drop_duplicates().reset_index(drop=True)
                session.record_transformation(f"Removed {dup_count:,} duplicate row(s)")

        # ── 2. Detect constant (zero-variance) columns ────────────────────────
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
        if constant_cols:
            pending_changes.append({
                "type": "drop_constant_columns",
                "description": f"Drop {len(constant_cols)} zero-variance column(s): {', '.join(constant_cols)}",
                "impact": f"Columns: {len(df.columns)} → {len(df.columns) - len(constant_cols)}",
            })
            if not dry_run:
                df = df.drop(columns=constant_cols)
                session.record_transformation(f"Dropped {len(constant_cols)} zero-variance column(s): {', '.join(constant_cols)}")

        # ── 3. Category standardization & Typo Correction ─────────────────────
        typo_replacements: Dict[str, Dict[str, str]] = {}
        if fix_typos:
            for col in df.select_dtypes(include=["object", "category", "string"]).columns:
                series_clean = df[col].dropna().astype(str)
                if series_clean.empty or series_clean.nunique() <= 1 or series_clean.nunique() > 50:
                    continue
                if series_clean.str.len().mean() > 35:
                    continue

                vc = series_clean.value_counts()
                lower_counts: Dict[str, int] = {}
                for val, count in vc.items():
                    low = val.strip().lower()
                    lower_counts[low] = lower_counts.get(low, 0) + count

                lower_to_canonical: Dict[str, str] = {}
                for low in lower_counts:
                    matching_cases = [v for v in vc.index if v.strip().lower() == low]
                    best_case = max(matching_cases, key=lambda v: (vc[v], -len(v)))
                    lower_to_canonical[low] = best_case

                canonical_lowers = sorted(lower_counts.keys(), key=lambda l: lower_counts[l], reverse=True)
                col_map: Dict[str, str] = {}

                for val in vc.index:
                    low = val.strip().lower()
                    canonical_low = low
                    # Fuzzy match rare/typo variations to a dominant canonical candidate
                    if lower_counts[low] <= max(1, len(series_clean) * 0.25):
                        for cand in canonical_lowers:
                            if cand != low and lower_counts[cand] >= 2 * lower_counts[low]:
                                ratio = difflib.SequenceMatcher(None, low, cand).ratio()
                                if ratio >= 0.82:
                                    canonical_low = cand
                                    break

                    target_val = lower_to_canonical[canonical_low]
                    if val != target_val:
                        col_map[val] = target_val

                if col_map:
                    typo_replacements[col] = col_map

            if typo_replacements:
                desc_items = []
                for c, mapping in typo_replacements.items():
                    map_str = ", ".join([f"'{k}' → '{v}'" for k, v in mapping.items()])
                    desc_items.append(f"{c}: {map_str}")
                pending_changes.append({
                    "type": "standardize_categories",
                    "description": f"Standardize category typos across {len(typo_replacements)} column(s)",
                    "impact": "; ".join(desc_items),
                })
                if not dry_run:
                    for c, mapping in typo_replacements.items():
                        df[c] = df[c].replace(mapping)
                    session.record_transformation(f"Standardized category typos in columns: {', '.join(typo_replacements.keys())}")


        # ── 4. Datetime feature extraction ───────────────────────────────────
        datetime_cols_found: List[str] = []
        if extract_datetime:
            for col in list(df.columns):
                is_dt = pd.api.types.is_datetime64_any_dtype(df[col])
                if not is_dt and df[col].dtype in ("object", "string"):
                    sample = df[col].dropna().head(20)
                    if len(sample) >= 3:
                        try:
                            # Avoid converting purely numeric or single-word strings
                            if not sample.astype(str).str.isnumeric().all():
                                parsed = pd.to_datetime(sample, errors="coerce")
                                if parsed.notnull().mean() >= 0.8:
                                    is_dt = True
                        except Exception:
                            is_dt = False

                if is_dt:
                    datetime_cols_found.append(col)

            if datetime_cols_found:
                dt_descs = []
                for col in datetime_cols_found:
                    dt_descs.append(f"{col} (extract year, month, day, day_of_week, hour, is_weekend)")
                pending_changes.append({
                    "type": "extract_datetime_features",
                    "description": f"Extract temporal features from {len(datetime_cols_found)} datetime column(s)",
                    "impact": "; ".join(dt_descs) + (" (drop raw column)" if drop_datetime_original else ""),
                })
                if not dry_run:
                    for col in datetime_cols_found:
                        dt_series = pd.to_datetime(df[col], errors="coerce")
                        ref_date = dt_series.min() if not dt_series.dropna().empty else pd.Timestamp("2000-01-01")
                        df[f"{col}_year"] = dt_series.dt.year.fillna(2000).astype(int)
                        df[f"{col}_month"] = dt_series.dt.month.fillna(1).astype(int)
                        df[f"{col}_day"] = dt_series.dt.day.fillna(1).astype(int)
                        df[f"{col}_day_of_week"] = dt_series.dt.dayofweek.fillna(0).astype(int)
                        if (dt_series.dt.hour > 0).any():
                            df[f"{col}_hour"] = dt_series.dt.hour.fillna(0).astype(int)
                        df[f"{col}_is_weekend"] = dt_series.dt.dayofweek.apply(lambda d: 1 if d in (5, 6) else 0)
                        df[f"{col}_days_since_ref"] = (dt_series - ref_date).dt.days.fillna(0).astype(int)

                        if drop_datetime_original:
                            df = df.drop(columns=[col])

                    session.record_transformation(f"Extracted temporal features from datetime columns: {', '.join(datetime_cols_found)}")

        # ── 5. Text / NLP Feature Extraction ──────────────────────────────────
        text_cols_found: List[str] = []
        if extract_text_nlp:
            for col in list(df.select_dtypes(include=["object", "category", "string"]).columns):
                s_nonnull = df[col].dropna().astype(str)
                if not s_nonnull.empty:
                    avg_len = s_nonnull.str.len().mean()
                    avg_words = s_nonnull.apply(lambda x: len(x.split())).mean()
                    nunique = s_nonnull.nunique()
                    # Identify text paragraphs / multi-word sentences
                    if avg_len > 25 or avg_words > 4 or (nunique > 20 and avg_words > 2):
                        text_cols_found.append(col)

            if text_cols_found:
                pending_changes.append({
                    "type": "extract_text_nlp",
                    "description": f"Extract NLP attributes from {len(text_cols_found)} free-form text column(s)",
                    "impact": f"Engineered char_count, word_count, uppercase_ratio, digit_ratio, sentiment for: {', '.join(text_cols_found)}",
                })
                if not dry_run:
                    for col in text_cols_found:
                        s_str = df[col].fillna("").astype(str)
                        df[f"{col}_char_count"] = s_str.str.len().astype(int)
                        df[f"{col}_word_count"] = s_str.apply(lambda x: len(x.split())).astype(int)
                        df[f"{col}_uppercase_ratio"] = s_str.apply(lambda x: sum(1 for c in x if c.isupper()) / max(1, len(x))).round(3)
                        df[f"{col}_digit_ratio"] = s_str.apply(lambda x: sum(1 for c in x if c.isdigit()) / max(1, len(x))).round(3)
                        df[f"{col}_sentiment_polarity"] = s_str.apply(_calc_sentiment)

                        if drop_text_original:
                            df = df.drop(columns=[col])

                    session.record_transformation(f"Extracted NLP features and dropped raw text from: {', '.join(text_cols_found)}")

        # ── 6. Missing value imputation ───────────────────────────────────────
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "string", "boolean"]).columns.tolist()

        imputed_numeric: List[str] = []
        imputed_categorical: List[str] = []

        for col in num_cols:
            missing = int(df[col].isnull().sum())
            if missing > 0:
                median_val = float(df[col].median()) if not df[col].isnull().all() else 0.0
                imputed_numeric.append(f"{col} ({missing} missing → fill median {median_val:.2f})")
                if not dry_run:
                    df[col] = df[col].fillna(median_val)

        for col in cat_cols:
            missing = int(df[col].isnull().sum())
            if missing > 0:
                mode_res = df[col].mode()
                mode_val = mode_res.iloc[0] if not mode_res.empty else "Missing"
                imputed_categorical.append(f"{col} ({missing} missing → fill mode '{mode_val}')")
                if not dry_run:
                    df[col] = df[col].fillna(mode_val)

        if imputed_numeric or imputed_categorical:
            all_imp = imputed_numeric + imputed_categorical
            pending_changes.append({
                "type": "impute_missing",
                "description": f"Impute missing values across {len(all_imp)} column(s)",
                "impact": "; ".join(all_imp),
            })
            if not dry_run:
                session.record_transformation(f"Imputed missing values across {len(all_imp)} column(s)")


        # ── 7. Outlier Detection & Handling (IQR) ─────────────────────────────
        outliers_detected: Dict[str, Dict[str, Any]] = {}
        strategy = outlier_strategy.lower()

        if strategy in ("clip", "remove", "winsorize"):
            for col in df.select_dtypes(include=["number"]).columns:
                if df[col].nunique(dropna=True) <= 2 or "id" in col.lower() or col.endswith("_is_weekend") or col.endswith("_year"):
                    continue
                q1 = float(df[col].quantile(0.25))
                q3 = float(df[col].quantile(0.75))
                iqr = q3 - q1
                if iqr <= 1e-6:
                    continue
                lower_b = q1 - 1.5 * iqr
                upper_b = q3 + 1.5 * iqr
                mask = (df[col] < lower_b) | (df[col] > upper_b)
                outlier_count = int(mask.sum())
                if outlier_count > 0:
                    outliers_detected[col] = {
                        "count": outlier_count,
                        "lower_bound": round(lower_b, 2),
                        "upper_bound": round(upper_b, 2),
                        "indices": df[mask].index.tolist(),
                    }

            if outliers_detected:
                action_desc = "Clip outliers to bounds" if strategy in ("clip", "winsorize") else "Remove outlier rows"
                details = [
                    f"{c}: {info['count']} outliers ({info['lower_bound']} to {info['upper_bound']})"
                    for c, info in outliers_detected.items()
                ]
                pending_changes.append({
                    "type": "outlier_handling",
                    "description": f"{action_desc} across {len(outliers_detected)} numeric column(s)",
                    "impact": "; ".join(details),
                })
                if not dry_run:
                    if strategy in ("clip", "winsorize"):
                        for c, info in outliers_detected.items():
                            df[c] = df[c].clip(lower=info["lower_bound"], upper=info["upper_bound"])
                        session.record_transformation(f"Clipped outliers across {len(outliers_detected)} numeric column(s)")
                    elif strategy == "remove":
                        all_outlier_idx = set()
                        for info in outliers_detected.values():
                            all_outlier_idx.update(info["indices"])
                        df = df.drop(index=list(all_outlier_idx)).reset_index(drop=True)
                        session.record_transformation(f"Removed {len(all_outlier_idx)} outlier rows across dataset")

        # ── Output Construction ───────────────────────────────────────────────
        if dry_run:
            return ToolResult(
                success=True,
                data={
                    "dry_run": True,
                    "pending_changes": pending_changes,
                    "initial_shape": [initial_rows, initial_cols],
                    "change_count": len(pending_changes),
                    "typos_detected": typo_replacements,
                    "datetime_columns": datetime_cols_found,
                    "text_columns": text_cols_found,
                    "outliers_detected": {c: info["count"] for c, info in outliers_detected.items()},
                    "message": f"Dry-run complete: {len(pending_changes)} data cleaning transformation(s) planned.",
                },
            )

        # Update active in-memory dataset
        dataset.df = df
        final_rows = len(df)
        final_cols = len(df.columns)

        return ToolResult(
            success=True,
            data={
                "dry_run": False,
                "initial_shape": [initial_rows, initial_cols],
                "final_shape": [final_rows, final_cols],
                "duplicate_rows_removed": dup_count,
                "constant_columns_dropped": constant_cols,
                "imputed_numeric_columns": imputed_numeric,
                "imputed_categorical_columns": imputed_categorical,
                "typos_standardized": typo_replacements,
                "datetime_columns_transformed": datetime_cols_found,
                "text_columns_transformed": text_cols_found,
                "outliers_handled": {c: info["count"] for c, info in outliers_detected.items()},
                "transformations_applied": [c["description"] for c in pending_changes],
                "message": (
                    f"Dataset successfully cleaned! Shape transformed from [{initial_rows} × {initial_cols}] to [{final_rows} × {final_cols}]. "
                    f"Applied {len(pending_changes)} transformation(s)."
                ),
            },
        )
