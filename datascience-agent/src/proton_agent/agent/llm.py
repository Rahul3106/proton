"""
LLM Provider Abstraction.
Per Section 17, decouples the agent from specific LLM providers.
Includes Gemini provider integration and a heuristic fallback client.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any, List, Optional
import warnings

# Suppress SDK deprecation and future warnings from printing to user terminal
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="google")


def _load_dotenv() -> None:
    """Helper to auto-load .env and .env.example variables if present in workspace."""
    for env_path in [Path(".env"), Path(".env.example"), Path("../.env"), Path.home() / ".env"]:
        if env_path.exists() and env_path.is_file():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and v and not v.startswith("your_"):
                            if not os.environ.get(k) or os.environ.get(k, "").startswith("your_"):
                                os.environ[k] = v
            except Exception:
                pass


_load_dotenv()


class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def select_tool(self, query: str, tools: List[dict[str, str]]) -> Optional[str]:
        """Selects the name of the most appropriate tool for the user query."""
        pass

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generates a text response for the given prompt."""
        pass


class RuleLLMClient(LLMClient):
    """Deterministic, heuristic intent matcher used when no API key is available."""

    def select_tool(self, query: str, tools: List[dict[str, str]]) -> Optional[str]:
        q = query.lower()

        # Questions asking 'how can I', 'can I', 'how to' should route to Q&A reasoning
        question_starters = ["how i can", "how can i", "can i", "how do i", "how to", "what is", "should i", "do i need", "is it necessary", "would you recommend"]
        if any(qs in q for qs in question_starters):
            return None

        import re

        def match_word(word: str) -> bool:
            return bool(re.search(r"\b" + re.escape(word) + r"\b", q))

        # Direct intent mappings
        if match_word("load") or "read csv" in q or "open dataset" in q or "import data" in q:
            return "load_dataset"
        if any(w in q for w in ["generate report", "create report", "report", "eda report", "model report"]):
            return "generate_report"
        if any(w in q for w in ["plot distribution", "distribution plot", "histogram plot", "plot hist"]):
            return "plot_distribution"
        if any(w in q for w in ["plot correlation", "correlation heatmap", "heatmap", "correlation plot"]):
            return "plot_correlation_matrix"
        if any(w in q for w in ["plot feature importance", "importance plot", "feature plot"]):
            return "plot_feature_importance"
        if match_word("clean") or "clean dataset" in q or "clean data" in q or "sanitize" in q:
            return "clean_dataset"
        if any(w in q for w in ["impute", "fill missing", "drop missing", "fix nulls", "fill nulls"]):
            return "handle_missing_values"
        if any(w in q for w in ["encode", "one-hot", "onehot", "label encode", "categorical encode"]):
            return "encode_features"
        if any(w in q for w in ["scale", "standardize", "normalize", "scaling"]):
            return "scale_features"
        if any(w in q for w in ["missing", "null", "na", "nan", "blank"]):
            return "detect_missing_values"
        if any(w in q for w in ["duplicate", "dupes", "repeated", "same row"]):
            return "detect_duplicates"
        if any(w in q for w in ["type", "dtype", "categorical", "numerical", "boolean", "column class"]):
            return "inspect_data_types"
        if any(w in q for w in ["correlation", "correlate", "relationship", "feature pairs"]):
            return "calculate_correlation"
        if any(w in q for w in ["statistic", "stats", "mean", "median", "std", "skew"]):
            return "calculate_statistics"
        if any(w in q for w in ["describe", "summary", "summarize", "overview"]):
            return "describe_dataset"
        if any(w in q for w in ["inspect", "view dataset", "preview", "head"]):
            return "inspect_dataset"

        # New tool routing — explainability, persistence, export
        if any(w in q for w in ["explain prediction", "explain last prediction", "explain row", "why this prediction", "explain single prediction"]):
            return "explain_prediction"
        if any(w in q for w in ["explain", "confusion matrix", "roc curve", "roc auc", "shap", "classification report", "per class", "residuals"]):
            return "explain_model"

        if any(w in q for w in ["export model", "save model", "persist model", "serialize model"]):
            return "export_model"
        if any(w in q for w in ["load model", "restore model", "open model"]):
            return "load_model"
        if any(w in q for w in ["export predictions", "save predictions", "save results", "export results", "download predictions"]):
            return "export_predictions"
        if any(w in q for w in ["drift", "covariate shift", "distribution shift", "data drift", "detect drift", "dataset drift", "check drift"]):
            return "detect_drift"
        if any(w in q for w in ["tune", "hyperparameter", "grid search", "random search", "optimize model", "best params"]):
            return "train_model"


        return None

    def generate_response(self, prompt: str) -> str:
        lines = prompt.splitlines()
        user_q = next((l.replace("User Question:", "").strip().strip('"') for l in lines if l.startswith("User Question:")), prompt)

        analysis = [
            f"Query: \"{user_q}\"",
            "",
        ]

        uq = user_q.lower()
        if "model" in uq and ("test" in uq or "upload" in uq or "predict" in uq):
            analysis.append("✓ Direct Answer: How to train & test machine learning models in PROTON:")
            analysis.append("  1. Load your training data: load <path/to/training_dataset.csv>")
            analysis.append("  2. Train & evaluate candidate models: train model  (or 'compare models')")
            analysis.append("  3. Generate test predictions on test data: load <path/to/test_dataset.csv> and run: predict")
            analysis.append("")

        has_dataset = any("ACTIVE DATASET CONTEXT" in l for l in lines)
        if has_dataset:
            ds_name = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("Dataset Name:")), "active dataset")
            ds_shape = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("Dimensions:")), "loaded")
            ds_cols = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("Columns (")), "features available")
            missing_cells = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("Missing Values:")), "0 total missing cells")

            if "clean" in uq and any(p in uq for p in ["should i", "do i need", "is it necessary", "would you recommend"]):
                analysis.append("✓ Advisory Assessment: Data Cleaning Guidance")
                analysis.append(f"  • Missing Values: {missing_cells}")
                analysis.append("  • Recommendation: Cleaning is recommended if your dataset contains missing values, duplicates, or constant columns.")
                analysis.append("  • To proceed with automated cleaning, run: clean dataset")
                analysis.append("")
            elif any(w in uq for w in ["any", "is there", "it here", "contain", "have", "has", "find", "present"]) and ("column" in uq or "col" in uq):
                actual_cols = [c.strip() for c in ds_cols.split(",")]
                target_col_found = None
                for word in uq.replace("?", "").replace("!", "").split():
                    word_clean = word.strip(" '\".,;:()")
                    if word_clean and word_clean in [c.lower() for c in actual_cols]:
                        target_col_found = next(c for c in actual_cols if c.lower() == word_clean)
                        break

                if target_col_found:
                    analysis.append(f"✓ Direct Answer: Yes! The column '{target_col_found}' IS present in dataset '{ds_name}'.")
                    analysis.append("")
                else:
                    words = uq.replace("?", "").replace("!", "").split()
                    candidate_word = None
                    for idx, w in enumerate(words):
                        if ("column" in w or "col" in w) and idx > 0:
                            candidate_word = words[idx - 1].strip(" '\".,;:()")
                            break
                    if candidate_word and candidate_word.lower() not in ["any", "a", "the", "this", "my", "which", "what"]:
                        analysis.append(f"✗ Direct Answer: No, the column '{candidate_word}' is NOT present in dataset '{ds_name}'.")
                        analysis.append(f"  Available Features: {ds_cols}")
                        analysis.append("")
                    else:
                        analysis.append(f"Data Schema Overview for '{ds_name}':")
                        analysis.append(f"  Available Features: {ds_cols}")
                        analysis.append("")
            elif "column" in uq and any(w in uq for w in ["how many", "count", "number of", "how much"]):
                cols_str = ds_shape.split("×")[1].strip() if "×" in ds_shape else ds_shape
                analysis.append(f"✓ Direct Answer: The active dataset '{ds_name}' has {cols_str}.")
                analysis.append(f"  Column List: {ds_cols}")
                analysis.append("")
            elif "row" in uq and any(w in uq for w in ["how many", "count", "number of", "how much"]):
                rows_str = ds_shape.split("×")[0].strip() if "×" in ds_shape else ds_shape
                analysis.append(f"✓ Direct Answer: The active dataset '{ds_name}' has {rows_str}.")
                analysis.append("")

            analysis.extend([
                "Data Context Overview:",
                f"  • Dataset: {ds_name} ({ds_shape})",
                f"  • Features: {ds_cols}",
                "",
                "Recommendation:",
                "  • Run 'analyze dataset' for complete 7-step EDA (quality, distributions, correlations).",
                "  • Run 'clean dataset' to resolve missing values & duplicates.",
                "  • Run 'train model' to evaluate machine learning algorithms on this dataset.",
                "",
                "[Tip: Set a valid GEMINI_API_KEY in .env file to enable generative AI reasoning.]",
            ])
        else:
            analysis.extend([
                "Data Context Analysis:",
                "  • No dataset is currently loaded in this session.",
                "  • Load a dataset using 'load <path/to/file.csv>' (e.g. 'load customer.csv').",
                "",
                "[Tip: Set a valid GEMINI_API_KEY in .env file to enable generative AI reasoning.]",
            ])

        return "\n".join(analysis)


class GeminiClient(LLMClient):
    """Google Gemini LLM client provider."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash") -> None:
        _load_dotenv()
        raw_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.api_key = raw_key if raw_key and not raw_key.startswith("your_") else None
        self.model_name = model_name
        self._fallback = RuleLLMClient()

    def select_tool(self, query: str, tools: List[dict[str, str]]) -> Optional[str]:
        # Fast-path: Check deterministic rule matcher first (<1ms)
        fast_match = self._fallback.select_tool(query, tools)
        if fast_match:
            return fast_match

        if not self.api_key:
            return None

        # Candidate Gemini models for complex/fuzzy queries
        models_to_try = [self.model_name, "gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
        for m_name in models_to_try:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(m_name)

                tool_descriptions = "\n".join([f"- {t['name']}: {t['description']}" for t in tools])
                prompt = (
                    f"You are a Data Science Agent. Match the user's natural language query to the exact name of ONE tool from the available tools.\n\n"
                    f"Available Tools:\n{tool_descriptions}\n\n"
                    f"User Query: \"{query}\"\n\n"
                    f"Return ONLY the exact tool name. If no tool matches, return 'NONE'."
                )
                response = model.generate_content(prompt)
                result = response.text.strip()
                if result and result.upper() != "NONE":
                    for t in tools:
                        if t["name"].lower() in result.lower():
                            return t["name"]
                break
            except Exception:
                continue

        return None


    def generate_response(self, prompt: str) -> str:
        if not self.api_key:
            return self._fallback.generate_response(prompt)

        models_to_try = [self.model_name, "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-latest"]
        for m_name in models_to_try:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(m_name)
                res = model.generate_content(prompt)
                if res and res.text:
                    return res.text.strip()
            except Exception:
                continue

        return self._fallback.generate_response(prompt)


def get_default_llm_client() -> LLMClient:
    """Returns GeminiClient if valid API key present, otherwise RuleLLMClient."""
    _load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key and not api_key.startswith("your_"):
        return GeminiClient(api_key=api_key)
    return RuleLLMClient()
