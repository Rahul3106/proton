# PROTON — Autonomous Data Science Agent

**Package:** `proton-agent` | **Version:** `v0.1.0` | **License:** MIT  
**CLI Commands:** `prot` | `proton` (alias)  
**Run:** `uv run prot start`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Core Features](#4-core-features)
5. [Demo Session](#5-demo-session)
6. [CLI Command Reference](#6-cli-command-reference)
7. [REST API Reference](#7-rest-api-reference)
8. [Web Interface](#8-web-interface)
9. [Technology Stack](#9-technology-stack)
10. [Setup and Installation](#10-setup-and-installation)
11. [Testing](#11-testing)
12. [Deployment](#12-deployment)
13. [Limitations and Roadmap](#13-limitations-and-roadmap)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Overview

**PROTON** is an autonomous, end-to-end Data Science Agent that lets users perform full data science
workflows using natural language through an interactive terminal REPL. No coding is required —
users type plain English commands and the agent plans, executes, and reports results automatically.

**What PROTON can do:**

- Load and inspect datasets in multiple formats (CSV, Excel, Parquet, JSON, TSV)
- Run a 7-step autonomous Exploratory Data Analysis (EDA) pipeline
- Detect missing values, duplicates, outliers, and data quality issues
- Clean, impute, encode, and scale data with a preview-before-apply guardrail
- Train and compare machine learning models with optional hyperparameter tuning
- Generate plots, reports, and prediction exports
- Serve everything through a FastAPI REST + SSE backend and a terminal-themed web UI
- Answer ad-hoc data science questions via Google Gemini AI integration

### Why not just use a chatbot or Jupyter notebook?

Most AI data science tools let the LLM *fabricate* results — writing numbers or conclusions that
were never computed. PROTON uses a **deterministic tool-registry architecture**: the LLM only plans
which tool to call and explains the output afterward; every actual data science result is computed
by a registered, deterministic Python tool. The LLM never invents a metric, a model score, or a
dataset fact. Every destructive action (cleaning, imputation, encoding) shows a diff and requires
explicit user confirmation before it runs.

---

## 2. Architecture

```
User Input (CLI / Web UI)
        |
        v
  +-------------+
  |  CLI / REPL |  <- prompt_toolkit (TAB completion, arrow key history)
  +------+------+
         |  natural language query
         v
  +--------------+
  |    Agent     |  <- Intent classification, meta-question routing, guardrails
  +------+-------+
         |
    +----+--------------------+
    |                         |
    v                         v
 Planner                 GeminiClient
 (multi-step plan)       (LLM fallback Q&A)
    |
    v
 Executor --> Tool Registry --> Tool (EDA / Clean / Model / Plot / Report)
                                       |
                                       v
                                 SessionState <-> DatasetContext
                                       |
                                       v
                                  outputs/
                              (plots, reports, models, predictions)
```

**Key Layers:**

| Layer      | Module                 | Role                                                           |
|:-----------|:-----------------------|:---------------------------------------------------------------|
| CLI / REPL | `proton_agent.cli`     | Interactive terminal entry point                               |
| Agent      | `proton_agent.agent`   | Intent classification, multi-step planning, LLM routing        |
| Tools      | `proton_agent.tools`   | 20+ registered tool implementations                            |
| Session    | `proton_agent.context` | Stateful in-memory session (dataset, model, history)           |
| API        | `proton_agent.api`     | FastAPI REST + SSE backend                                     |
| Config     | `proton_agent.config`  | Settings, environment, API key management                      |
| UI         | `proton_agent.ui`      | Terminal banner, Rich output formatting                        |

---

## 3. Project Structure

```
datascience-agent/
+-- src/
|   +-- proton_agent/              # Python package (import as: proton_agent)
|       +-- agent/                 # Core agent: planner, executor, LLM client
|       +-- api/                   # FastAPI server, session manager, static web UI
|       |   +-- static/            # Terminal-themed web UI (index.html, favicon.svg)
|       +-- cli/                   # CLI app entry point, REPL loop
|       +-- config/                # Settings, environment loading
|       +-- context/               # SessionState, DatasetContext
|       +-- tools/                 # Tool registry and all tool implementations
|       |   +-- analysis/          # EDA, statistics, correlation, outliers
|       |   +-- data/              # Load, inspect, detect duplicates/missing
|       |   +-- modeling/          # Train, compare, tune, predict, explain, export
|       |   +-- preprocessing/     # Clean, impute, encode, scale
|       |   +-- reporting/         # Report generation
|       |   +-- visualization/     # Distribution plots, heatmaps, feature importance
|       +-- ui/                    # Banner, display helpers
+-- tests/                         # 82 unit tests (pytest)
+-- outputs/                       # Generated plots, reports, models, predictions
+-- Dockerfile
+-- docker-compose.yml
+-- pyproject.toml                  # pip package name: proton-agent
+-- PROJECT_SUMMARY.md
```

> **Package name note:** The distributable package is `proton-agent` (pyproject.toml).
> The importable Python package is `proton_agent` (the `src/proton_agent/` directory).
> There is no `datascience_agent` package; that name only appears in the repository folder name.

---

## 4. Core Features

### 4.1 Multi-Format Dataset Ingestion

| Format  | Extensions                                        |
|:--------|:--------------------------------------------------|
| CSV     | `.csv` (UTF-8; auto-fallback to latin-1/cp1252)  |
| Excel   | `.xlsx`, `.xls`, `.xlsm`, `.xlsb`                |
| Parquet | `.parquet`                                        |
| JSON    | `.json`                                           |
| TSV     | `.tsv`, `.tab`                                    |

**Dedicated load commands:**

- `load train <filepath>` — loads the primary training dataset into the active session
- `load test <filepath>` — loads a separate evaluation/test dataset

---

### 4.2 Autonomous EDA Pipeline

Running `analyze dataset` triggers the **7-step autonomous EDA pipeline**:

1. Dataset inspection (shape, schema, sample rows)
2. Data type profiling (numerical vs. categorical columns)
3. Missing value analysis
4. Duplicate row detection
5. Statistical summary (mean, std, median, min, max, skewness)
6. Pearson correlation matrix computation
7. Data quality health score with actionable warnings

---

### 4.3 Smart Intent Classification and Guardrails

| Guardrail                     | Behavior                                                                        |
|:------------------------------|:--------------------------------------------------------------------------------|
| **Advisory filtering**        | Queries like "should I clean this?" return guidance instead of mutating data    |
| **Meta-question routing**     | Agent-logic questions route to Gemini Q&A instead of triggering a plan          |
| **Column validation**         | Visualization commands validate column existence before execution               |
| **Target leakage protection** | `id`, `index`, `Unnamed: 0` are auto-excluded from model training               |
| **Preview-before-apply**      | `clean dataset` shows a dry-run diff and prompts `[y/N]` before modifying data  |

---

### 4.4 Machine Learning Pipeline

| Capability                      | Command or Endpoint                                       |
|:--------------------------------|:----------------------------------------------------------|
| Train best model automatically  | `train model`                                             |
| Train a specific algorithm      | `train decision tree`, `train random forest`              |
| Compare all algorithms          | `compare models`                                          |
| Hyperparameter tuning           | `tune model` or `train model with tuning`                 |
| Predictions on test set         | `predict`                                                 |
| Single-row inference            | `POST /api/session/{id}/predict-row`                      |
| Model explainability            | `explain model` — Confusion Matrix, ROC/AUC, SHAP        |
| Export trained model            | `export model [name]` → `outputs/models/<name>.pkl`      |
| Load saved model                | `load model <file>`                                       |
| Export predictions to CSV       | `export predictions <file.csv>`                           |

**Supported algorithms (classification):** Random Forest, Decision Tree, Gradient Boosting, Logistic Regression

---

### 4.5 Gemini AI Integration

- **Default model** (configured in `settings.py`): `gemini-1.5-flash`
- **Runtime fallback chain** (in `llm.py`): `gemini-1.5-flash` → `gemini-2.0-flash` → `gemini-1.5-pro`
  (tried in order; first successful response is used)
- **API key source:** `.env` or `.env.example` (`GEMINI_API_KEY`)
- **No-key mode:** If no valid API key is present, `RuleLLMClient` provides deterministic
  heuristic responses. All tool results remain fully functional; only the LLM Q&A fallback is
  disabled.

---

### 4.6 REPL Interactivity

- **TAB autocompletion** via `prompt_toolkit` for all 20+ built-in commands
- **Arrow key navigation** (Up/Down) for session command history
- **Prompt:** `prot > `
- **Startup banner:** Centered ASCII art, version badge, LLM status indicator, loaded dataset
  table, quick command showcase

---

## 5. Demo Session

A realistic end-to-end session from load to prediction:

```
prot > load train data/titanic.csv
  ✔ Loaded train dataset: titanic.csv  (891 rows x 12 cols)

prot > analyze dataset
  [Step 1/7] Inspecting schema...
  [Step 2/7] Profiling data types: 5 numerical, 7 categorical
  [Step 3/7] Missing values: Age (177), Cabin (687), Embarked (2)
  [Step 4/7] Duplicates: 0 found
  [Step 5/7] Statistics computed (mean, std, skewness...)
  [Step 6/7] Correlation matrix computed
  [Step 7/7] Data quality score: 61/100 — 3 warnings

prot > clean dataset
  Preview (dry-run):
    - Impute Age (177 nulls) with mean (29.7)
    - Drop Cabin (77% missing)
    - Impute Embarked (2 nulls) with mode (S)
  Apply these changes? [y/N]: y
  ✔ Dataset cleaned. 891 rows x 11 cols

prot > compare models
  Training 4 models on target: Survived...
  ✔ Random Forest        Accuracy: 0.838   F1: 0.814
  ✔ Gradient Boosting    Accuracy: 0.831   F1: 0.806
  ✔ Decision Tree        Accuracy: 0.810   F1: 0.788
  ✔ Logistic Regression  Accuracy: 0.801   F1: 0.776
  Best model: Random Forest (auto-saved)

prot > explain model
  Confusion Matrix, ROC/AUC (AUC = 0.91), Classification Report saved.
  SHAP feature importance: Sex > Fare > Age > Pclass > SibSp

prot > predict
  ✔ Predictions written to outputs/predictions.csv (891 rows)
```

---

## 6. CLI Command Reference

| Category             | Command                          | Description                                               |
|:---------------------|:---------------------------------|:----------------------------------------------------------|
| **Ingestion**        | `load train <file>`              | Load training dataset into session                        |
|                      | `load test <file>`               | Load separate test dataset                                |
|                      | `inspect dataset`                | View dimensions, schema, and sample rows                  |
| **EDA and Quality**  | `analyze dataset`                | Run 7-step autonomous EDA pipeline                        |
|                      | `audit data quality`             | Generate dataset health score and warnings                |
|                      | `detect missing values`          | Profile missing values per column                         |
|                      | `detect duplicates`              | Find and count duplicate rows                             |
|                      | `inspect data types`             | Categorize columns into numerical/categorical             |
|                      | `calculate correlation`          | Compute Pearson correlation matrix                        |
|                      | `calculate statistics`           | Generate mean, std, median, min, max, skewness            |
| **Preprocessing**    | `clean dataset`                  | Preview and apply automated cleaning (with [y/N] confirm) |
|                      | `impute missing`                 | Fill nulls using mean/mode imputation                     |
|                      | `encode features`                | Apply one-hot encoding to categorical columns             |
|                      | `scale features`                 | Apply StandardScaler to numerical columns                 |
| **Machine Learning** | `train model`                    | Auto-select and train the best model                      |
|                      | `train decision tree`            | Train only a Decision Tree                                |
|                      | `train random forest`            | Train only a Random Forest                                |
|                      | `compare models`                 | Evaluate all algorithms, auto-save the best               |
|                      | `tune model`                     | Run RandomizedSearchCV hyperparameter tuning              |
|                      | `predict`                        | Generate predictions on test data                         |
|                      | `explain model`                  | Show Confusion Matrix, ROC/AUC, SHAP                     |
|                      | `export model [name]`            | Serialize model to `.pkl`                                 |
|                      | `load model <file>`              | Restore serialized model into session                     |
|                      | `export predictions <file.csv>`  | Export prediction CSV to `outputs/`                      |
| **Visualization**    | `plot distribution of <col>`     | Save distribution histogram PNG                           |
|                      | `plot correlation matrix`        | Save correlation heatmap PNG                              |
|                      | `plot feature importance`        | Save feature importance bar chart PNG                     |
| **Reporting**        | `generate report`                | Save comprehensive Markdown report                        |
|                      | `config show`                    | Display LLM provider, API key status, session metrics     |
|                      | `clear`                          | Clear terminal output                                     |
|                      | `help`                           | Display command menu                                      |
|                      | `exit`                           | Exit the agent session                                    |

---

## 7. REST API Reference

The FastAPI server (`proton_agent.api.server`) exposes the following HTTP endpoints.

### System and Health

| Endpoint        | Method | Description                                                                |
|:----------------|:------:|:---------------------------------------------------------------------------|
| `/`             | GET    | Serves web UI for browsers; returns JSON health check for API clients      |
| `/health`       | GET    | Service status, version, LLM info, active sessions, registered tool count  |
| `/api/config`   | GET    | LLM model, provider, temperature, masked API key, output directory paths   |

### Session Lifecycle

| Endpoint              | Method | Description                                                       |
|:----------------------|:------:|:------------------------------------------------------------------|
| `/api/session`        | POST   | Create a new in-memory session (optional custom `session_id`)    |
| `/api/sessions`       | GET    | List all active session IDs                                       |
| `/api/session/{id}`   | GET    | Inspect session state (dataset, model, history, transformations)  |
| `/api/session/{id}`   | DELETE | Terminate and clear a session                                     |

### Data Ingestion and Exploration

| Endpoint                            | Method | Description                                                            |
|:------------------------------------|:------:|:-----------------------------------------------------------------------|
| `/api/session/{id}/load`            | POST   | Ingest dataset via multipart file upload or JSON filepath body         |
| `/api/session/{id}/sample`          | GET    | Paginated records with schema (params: `page`, `page_size`, `is_test`) |
| `/api/session/{id}/revert`          | POST   | Revert active dataset to its pristine original state                   |
| `/api/session/{id}/target-column`   | POST   | Set or clear the active target column                                  |

### Tools and Agent

| Endpoint                             | Method | Description                                                   |
|:-------------------------------------|:------:|:--------------------------------------------------------------|
| `/api/tools`                         | GET    | List all registered tools with names and descriptions         |
| `/api/session/{id}/tool/{name}`      | POST   | Execute any registered tool dynamically (JSON kwargs in body) |
| `/api/session/{id}/query`            | POST   | Natural language agent query with multi-step planning         |
| `/api/session/{id}/stream`           | GET    | SSE live stream of step execution events                      |
| `/api/session/{id}/suggest-models`   | GET    | Intelligent model recommendations based on dataset profile    |

### Machine Learning and Predictions

| Endpoint                          | Method | Description                                         |
|:----------------------------------|:------:|:----------------------------------------------------|
| `/api/session/{id}/predict-row`   | POST   | Single-row inference with class probability output  |
| `/api/session/{id}/experiments`   | GET    | List all experiment and training run records        |

### Persistence and Artifacts

| Endpoint                           | Method | Description                                          |
|:-----------------------------------|:------:|:-----------------------------------------------------|
| `/api/artifacts/{path}`            | GET    | Serve plots, reports, `.pkl` models, prediction CSVs  |
| `/api/session/{id}/export-state`   | GET    | Export full session state to JSON                    |
| `/api/session/{id}/import-state`   | POST   | Restore session state from exported JSON             |

### SSE Event Types

| Event             | Payload Fields                                                  |
|:------------------|:----------------------------------------------------------------|
| `plan_created`    | `goal`, `total_steps`, `steps[]`                                |
| `step_start`      | `step_number`, `description`, `tool_name`                       |
| `step_complete`   | `step_number`, `tool_name`, `success`, `data` or `error`        |
| `plan_complete`   | `is_complete`, `completed_steps`, `total_steps`, `summary`      |
| `qa_response`     | `explanation` (emitted when a query has no plan steps)          |

---

## 8. Web Interface

The web UI is served statically from `src/proton_agent/api/static/` at the root URL `/`.

### Visual Design

| Property | Details                                                            |
|:---------|:-------------------------------------------------------------------|
| Theme    | Cyberpunk dark — charcoal `#090b10` background, neon cyan accents |
| Font     | JetBrains Mono (monospace terminal aesthetic)                      |
| Icons    | Inline SVG throughout; custom neon delta favicon (`favicon.svg`)   |
| Banner   | Centered glowing ASCII PROTON art with PROTON header badge         |

### Live Status Panel

- LLM engine indicator: `ONLINE Gemini 1.5 Flash` or `HEURISTIC MODE`
- Dataset status badges: `Loaded: iris.csv` or `None` for both train and test sets
- Agent version badge (`v0.1.0`)

### Interactive Components

| Component             | Description                                                                              |
|:----------------------|:-----------------------------------------------------------------------------------------|
| `prot >` Prompt Bar   | Monospace input with glowing focus state and EXECUTE button                              |
| Autocomplete Chips    | Dynamic suggestion chips that filter as the user types and execute on click              |
| Quick-Action Cards    | 6 point-and-click category cards requiring no CLI knowledge                              |
| SSE Streaming Feed    | Live progress checklist with step badges, formatted metrics, tables, and image artifacts |
| Guided Action Modals  | Glassmorphism popups with dropdowns, selectors, and form buttons                         |
| Schema Explorer       | Interactive column type and dtype viewer                                                 |
| Paginated Data Viewer | Scrollable sample records table with page navigation                                     |
| Quality Health Gauge  | Circular dataset quality score indicator                                                 |
| Dry-Run Diff Cards    | Preview-before-apply diff view with Apply and Skip confirmation buttons                  |
| Model Leaderboard     | Live performance table with per-model row actions after training                         |
| Single-Row Predictor  | Dynamic feature input form with real-time class probability breakdown                    |
| Session Backup        | JSON export and import for full session persistence                                      |

---

## 9. Technology Stack

| Layer               | Library or Tool      | Minimum Version |
|:--------------------|:---------------------|:----------------|
| Language            | Python               | 3.11            |
| CLI framework       | Typer                | 0.15.1          |
| Terminal UI         | Rich                 | 13.9.4          |
| REPL / Autocomplete | prompt-toolkit       | 3.0.0           |
| Data manipulation   | pandas               | 2.0.0           |
| Machine learning    | scikit-learn         | 1.3.0           |
| Visualization       | matplotlib           | 3.7.0           |
| Excel support       | openpyxl             | 3.1.0           |
| AI / LLM            | google-generativeai  | 0.8.0           |
| Web API             | FastAPI              | 0.100.0         |
| ASGI server         | uvicorn              | 0.20.0          |
| File uploads        | python-multipart     | 0.0.9           |
| Build backend       | hatchling            | latest          |
| Package manager     | uv                   | latest          |
| Test runner         | pytest               | 8.3.4           |
| HTTP test client    | httpx                | 0.28.0          |

---

## 10. Setup and Installation

### Prerequisites

- Python 3.11 or higher
- `uv` package manager — see https://github.com/astral-sh/uv

### Install and Run the CLI Agent

```bash
# 1. Clone the repository
git clone <repo-url>
cd datascience-agent

# 2. Install all dependencies
uv sync

# 3. Configure your Gemini API key
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY=your_key_here

# 4. Start the interactive terminal agent
uv run prot start
```

### Start the Web API Server

```bash
uv run uvicorn proton_agent.api.server:app --reload --host 0.0.0.0 --port 8000
```

| URL                              | Purpose                      |
|:---------------------------------|:-----------------------------|
| `http://localhost:8000`          | Terminal-themed web UI       |
| `http://localhost:8000/docs`     | Interactive Swagger API docs |
| `http://localhost:8000/health`   | Service health check         |

### Environment Variables

| Variable               | Required | Default            | Description                |
|:-----------------------|:--------:|:-------------------|:---------------------------|
| `GEMINI_API_KEY`       | Yes      | —                  | Google AI Studio API key   |
| `PROTON_LLM_PROVIDER`  | No       | `gemini`           | LLM provider name          |
| `PROTON_GEMINI_MODEL`  | No       | `gemini-1.5-flash` | Gemini model name          |
| `TEMPERATURE`          | No       | `0.2`              | LLM sampling temperature   |

---

## 11. Testing

The project includes **82 unit tests** covering all major tool categories.

```bash
# Run all tests
uv run --no-sync pytest

# Run with verbose output
uv run --no-sync pytest -v

# Run a specific test file
uv run --no-sync pytest tests/test_eda.py
```

**Expected result:**

```
82 passed in ~13 seconds
```

**Test coverage areas:**

| Test File        | What It Covers                                                     |
|:-----------------|:-------------------------------------------------------------------|
| `test_eda.py`    | Outlier detection, distribution analysis, data quality audit tools |
| Other test files | Ingestion, preprocessing, modeling, visualization, API endpoints   |

---

## 12. Deployment

### Docker — Single Container

```bash
# Build the image
docker build -t proton-agent .

# Run the container
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here proton-agent
```

### Docker Compose

```bash
# Build and start all services
docker compose up --build

# Run in detached mode
docker compose up -d --build
```

### Production Notes

- The FastAPI server runs via `uvicorn` on port `8000`
- Generated artifacts (plots, reports, models, predictions) are written to `outputs/`
- Mount `outputs/` as a Docker volume to persist artifacts across container restarts
- CORS is open by default (`allow_origins=["*"]`); restrict to your frontend domain in production

---

## 13. Limitations and Roadmap

### Current Limitations

| Area                      | Current State                                                                                 |
|:--------------------------|:----------------------------------------------------------------------------------------------|
| **Authentication**        | No auth, no multi-user support. Any client can access any session.                            |
| **Session persistence**   | Sessions are in-memory only. Server restart = all sessions lost.                              |
| **Session export**        | Exports history, transformations, and target column to JSON. Trained models are **not** included in JSON export; they must be separately exported via `export model`. No database backend. |
| **Model families**        | Classification only: Random Forest, Decision Tree, Gradient Boosting, Logistic Regression. No regression task type, no XGBoost/LightGBM/CatBoost. |
| **CORS**                  | Wide open by default (`allow_origins=["*"]`). Must be restricted for production deployments.  |
| **LLM grounding**         | Gemini is used for explanation and narrative only; it never computes metrics or mutates data. However, Gemini-generated text (diagnostic messages, Q&A answers) is not automatically verified against actual tool output. |
| **Scalability**           | Single-process `uvicorn` server. Not suitable for high concurrency without an ASGI worker pool. |

### Roadmap

- [ ] Regression task support (RMSE, MAE, R², residual plots, predicted-vs-actual scatter)
- [ ] XGBoost, LightGBM, CatBoost model families
- [ ] Optuna-based hyperparameter tuning (replacing RandomizedSearchCV)
- [ ] Per-row SHAP explanation for single-row predictions ("Why?" button in web UI)
- [ ] Data drift detection (training distribution vs. production batch)
- [ ] LLM-suggested feature engineering (opt-in, same guardrail pattern as cleaning)
- [ ] Narrated executive session summaries from Gemini, grounded in actual tool output
- [ ] Auth and session isolation for multi-user deployments

---

## 14. Troubleshooting

### `uv` command not found

`uv` is not on your PATH. Install it via one of:

```bash
# Option 1: Official installer (requires astral.sh to be reachable)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Option 2: winget (Windows — no astral.sh required)
winget install astral-sh.uv

# Option 3: pip fallback (always works if Python is installed)
pip install uv
```

After installation, restart your terminal so the PATH update takes effect.

---

### BOM character breaking `tomllib` on Windows

If you created or edited `pyproject.toml` using PowerShell `Set-Content` without specifying
`-Encoding utf8NoBOM`, it may have written a UTF-8 BOM that breaks Python’s `tomllib` parser
with a `TOMLDecodeError`. Fix it:

```powershell
$content = Get-Content -Path pyproject.toml -Raw
[IO.File]::WriteAllText(
    (Resolve-Path pyproject.toml).Path,
    $content,
    [Text.UTF8Encoding]::new($false)
)
```

---

### `hatchling build` fails: `README.md not found`

`pyproject.toml` declares `readme = "README.md"`. The file must exist at the project root.
Create a minimal one if it is missing:

```bash
echo "# proton-agent" > README.md
```

---

### `astral.sh` unreachable (corporate proxy / firewall)

Use `winget` or `pip` to install `uv` instead (see above). Then proceed normally:

```bash
pip install uv
uv sync
uv run prot start
```

---

### Gemini API key not recognized

- Ensure `.env` exists at the project root (next to `pyproject.toml`), not inside `src/`.
- The key must not start with `your_` — the loader skips placeholder values.
- Keys from Google AI Studio (`aistudio.google.com`) are supported.
- Run `config show` to confirm the masked key is loaded.
- If no key is set, PROTON runs in **HEURISTIC MODE** using `RuleLLMClient`. All tool results
  still work; only the LLM Q&A fallback is disabled.

---

### `ModuleNotFoundError: No module named proton_agent`

Run commands with `uv run` (not plain `python`), which activates the virtual environment
and the `src/` path automatically:

```bash
# Correct
uv run prot start

# Incorrect (will fail unless venv is manually activated)
python -m proton_agent
```
