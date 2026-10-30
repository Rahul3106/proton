# Proton Data Science Agent

Proton is a data science assistant that can load datasets, inspect and clean
data, create visualizations, train machine-learning models, generate reports,
and expose the same functionality through a local web API.

The application source is in [`datascience-agent/`](datascience-agent/).

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- A Gemini API key for natural-language agent features

## Run locally with uv

From the repository root:

```bash
cd datascience-agent
uv sync
cp .env.example .env
```

Open `.env` and replace `my_api_key` with your Gemini API key:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
```

Check that the CLI is installed:

```bash
uv run proton --help
uv run proton version
```

### Start the interactive terminal session

```bash
uv run proton start
```

Example commands inside Proton:

```text
proton > load customer.csv
proton > analyze it
proton > build a model to predict churn
```

### Start the local web API and dashboard

```bash
uv run proton serve --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000> in a browser. The `api` command is an alias:

```bash
uv run proton api --host 127.0.0.1 --port 8000
```

Generated plots, reports, models, and uploads are stored in
`datascience-agent/outputs/`.

## Run with Docker

From the repository root:

```bash
cd datascience-agent
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env`, then build and start the service:

```bash
docker compose up --build
```

The dashboard and API are available at <http://127.0.0.1:8000>. Stop the
service with:

```bash
docker compose down
```

## Run tests

```bash
cd datascience-agent
uv run pytest
```

## Project layout

```text
datascience-agent/
├── src/proton_agent/   # Application and CLI source
├── tests/              # Test suite
├── outputs/            # Generated artifacts (created at runtime)
├── pyproject.toml      # Project and dependency configuration
└── docker-compose.yml  # Local Docker service
```
