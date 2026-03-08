# Operation_Flag

CoreWeave Demo Wizard Hackathon MVP.

This is a single-file Gradio app that generates a structured, **hypothetical** GPU cluster demo plan from:
- workload type
- requested GPU count
- business/technical priorities

It uses OpenAI Chat Completions (`OPENAI_MODEL`, default: `gpt-4o-mini`) and returns:
- on-screen markdown output
- optional downloadable `.md` report

## Why this matters for the hackathon

This prototype is built to support fast, credible demo prep for CoreWeave-style customer conversations.

It emphasizes:
- speed to first draft
- consistent response structure
- explicit assumptions and hypothetical framing for high-stakes scenarios
- guardrails against fabricated compliance or factual claims

## Quick start

### 1. Create and activate Python 3.12 virtual environment

```bash
cd /Users/lwells/Desktop/Hackathon/Operation_Flag
python3.12 -m venv .venv312
source .venv312/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API key

Option A (recommended): put key in `secrets/.env`

```bash
mkdir -p secrets
echo 'OPENAI_API_KEY="sk-..."' > secrets/.env
```

Option B: export in shell

```bash
export OPENAI_API_KEY="sk-..."
```

Optional model override:

```bash
export OPENAI_MODEL="gpt-4o-mini"
```

You can also provide a fallback key in the app UI.

### 4. Run the app

```bash
python demo_wizard.py
```

Open in browser:
- `http://localhost:7860`

If port `7860` is busy:

```bash
PORT=7861 python demo_wizard.py
```

## Project files

- `demo_wizard.py` - app, prompt logic, OpenAI call, and markdown export
- `requirements.txt` - Python dependencies
- `.gitignore` - Python/venv/editor ignores
