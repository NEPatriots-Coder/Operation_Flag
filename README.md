# CoreWeave Atlas

**A CoreWeave demo wizard that generates structured, hypothetical GPU cluster demo plans in seconds.**

## Overview

Prepping a credible GPU cluster demo on the fly is tedious: you need the right cluster shape, realistic config snippets, defensible performance framing, and clear next steps -- all without overclaiming. CoreWeave Atlas automates that first draft.

It is a single-file Python/Gradio app backed by an LLM (Anthropic Claude or OpenAI). You select a workload type, pick a GPU count, describe your priorities, and the wizard returns a structured markdown plan with cluster recommendations, a deployment snippet, production next steps, and explicit assumptions. The output is designed to be shown in a live customer conversation or saved as a follow-up artifact.

Built for sales engineers, solutions architects, demo engineers, and hackathon collaborators working in CoreWeave-style GPU cloud environments.

## Key Features

- **Structured demo plan generation** from workload type + GPU count + business/technical priorities.
- **Consistent output contract** -- every plan follows the same five-section markdown structure, so your demos look uniform.
- **Anthropic and OpenAI support** with a one-line config switch (`LLM_PROVIDER`).
- **Hypothetical framing and guardrails** -- the system prompt prevents fabricated compliance claims, certifications, or production guarantees. High-stakes scenarios are explicitly labeled as hypothetical.
- **Benchmark language guardrails** -- performance references (goodput, H200 vs H100, GB200 framing) are directional and cited conservatively, used at most once per output.
- **Downloadable markdown report** -- every generated plan is available as a `.md` file for note-taking, follow-up emails, or CRM attachments.
- **Fallback API key field** -- if your env isn't configured, paste a key directly in the UI to get started immediately.
- **Configurable port** -- run on any port via the `PORT` env var.

## Architecture at a Glance

```
demo_wizard.py          <- Everything lives here
  |
  +-- SYSTEM_PROMPT     <- Output rules, safety constraints, benchmark guardrails
  +-- FEW_SHOT_EXAMPLES <- Five worked examples covering training, inference,
  |                        high-stakes hypothetical, K8s YAML, and massive-dataset scenarios
  +-- build_user_prompt <- Assembles the per-request prompt from UI inputs
  +-- generate_demo     <- Provider dispatch (Anthropic or OpenAI), response extraction,
  |                        markdown export to temp file
  +-- build_interface   <- Gradio Blocks UI (dropdown, slider, text fields, button)
  +-- main              <- Loads secrets/.env, launches Gradio server
```

The app is intentionally single-file. Prompt engineering, provider logic, response parsing, and UI are co-located so the entire behavior is visible in one read-through.

## Quick Start

### 1. Clone the repo

```bash
git clone <your-repo-url> Operation_Flag
cd Operation_Flag
```

### 2. Create and activate a Python 3.12 virtualenv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `gradio`, `openai`, `anthropic`, and `python-dotenv`.

### 4. Configure your LLM provider

Create a `secrets/.env` file (this path is gitignored):

```bash
mkdir -p secrets
```

**Option A -- Anthropic (Claude):**

```bash
cat > secrets/.env << 'EOF'
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
EOF
```

**Option B -- OpenAI:**

```bash
cat > secrets/.env << 'EOF'
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
EOF
```

### 5. Run the app

```bash
python demo_wizard.py
```

Open your browser to **http://localhost:7860**.

If port 7860 is already in use:

```bash
PORT=7861 python demo_wizard.py
```

## Configuration

| Variable | Values | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | `anthropic`, `openai`, `auto` | `auto` | Which LLM backend to use |
| `ANTHROPIC_API_KEY` | string | -- | Required when provider is `anthropic` |
| `ANTHROPIC_MODEL` | string | `claude-3-5-sonnet-latest` | Anthropic model identifier |
| `OPENAI_API_KEY` | string | -- | Required when provider is `openai` |
| `OPENAI_MODEL` | string | `gpt-4o-mini` | OpenAI model identifier |
| `PORT` | integer | `7860` | Gradio server port |

**Provider resolution (`auto` mode):** When `LLM_PROVIDER` is `auto` or unset, the app checks for `ANTHROPIC_API_KEY` first, then `OPENAI_API_KEY`. The first key found wins. If neither key exists, it falls back to OpenAI (which will fail without a key unless one is provided in the UI fallback field).

## How to Use the Demo Wizard

1. **Select a workload type** from the dropdown. Options range from standard (LLM Fine-Tuning, Batch Inference) to hypothetical high-stakes scenarios (Secure Large-Scale Operational Simulation).

2. **Set the GPU count** using the slider (1--1024).

3. **Describe your priorities** in the text box -- for example: "low latency p95, cost control, checkpoint reliability." Leave blank and the wizard will note that no explicit priorities were provided.

4. **Click "Generate Demo."** The LLM generates a five-section markdown plan:
   - **Recommended Cluster Configuration** -- GPU type, networking, storage assumptions.
   - **Performance & Business Impact** -- directional benchmark framing tied to your workload.
   - **Sample Deployment Snippet** -- a YAML or Slurm code block you can walk through on-screen.
   - **Production Next Steps** -- concrete actions including CTA (ARENA interest form, sales contact).
   - **Notes & Assumptions** -- everything the plan assumes, stated explicitly.

5. **Download the report** using the file link below the output. The `.md` file can be attached to a follow-up email or dropped into a CRM note.

**Demo-day tip:** For a live customer call, pre-fill the workload type and GPU count to match the customer's scenario, type their stated priorities, and generate while sharing your screen. The structured output gives you a credible talking framework in under 30 seconds.

## Assumptions, Limitations, and Guardrails

This section exists to build trust by being explicit about what the tool does **not** do.

- **All plans are hypothetical.** The wizard generates illustrative demo plans, not binding infrastructure commitments, quotes, or SLAs.
- **No real-time pricing.** Cluster costs are not calculated or estimated. Cost discussions should happen with sales/solutions architecture.
- **No compliance certification claims.** The system prompt explicitly prevents fabrication of FedRAMP authorization, SOC 2, HIPAA, or similar claims. For high-stakes scenarios, the output uses "pursuit alignment" language only.
- **Benchmark figures are directional.** Performance references (goodput, H200 vs H100 throughput, GB200 speedup) are framed as directional from public reports and MLPerf benchmarks, not guarantees.
- **YAML/Slurm snippets are illustrative.** They follow realistic structure (K8s-style, sbatch directives) but are not production-ready configs. Adapt to your cluster policies.
- **Single LLM call per generation.** There is no multi-turn refinement, retrieval augmentation, or tool use in this MVP.

## File and Project Layout

| Path | Purpose |
|---|---|
| `demo_wizard.py` | Application entry point -- prompt config, LLM dispatch, Gradio UI, markdown export |
| `requirements.txt` | Python dependencies (`gradio`, `openai`, `anthropic`, `python-dotenv`) |
| `secrets/.env` | Provider and API key config (gitignored, you create this) |
| `.gitignore` | Ignores virtualenvs, caches, secrets, editor files |
| `CLAUDE.md` | Project instructions for AI-assisted development |

## Development Notes and Local Iteration

### Modifying prompts

The system prompt (`SYSTEM_PROMPT`) and few-shot examples (`FEW_SHOT_EXAMPLES`) are plain strings at the top of `demo_wizard.py`. Edit them directly -- no template engine or external files to hunt for.

### Changing models

Set `ANTHROPIC_MODEL` or `OPENAI_MODEL` in `secrets/.env` to point at any model your API key has access to. For example, `OPENAI_MODEL=gpt-4o` or `ANTHROPIC_MODEL=claude-sonnet-4-20250514`.

### Adding input fields

The UI is built with Gradio Blocks in `build_interface()`. Add a new `gr.Textbox`, `gr.Slider`, or `gr.Dropdown`, wire it into the `generate_btn.click()` inputs list, and update `generate_demo()` to accept the new parameter.

### Debugging provider issues

Run with verbose logging to see what provider was resolved:

```python
provider = resolve_provider()
print(f"Using provider: {provider}")
```

Or temporarily hardcode `LLM_PROVIDER=anthropic` in your env to bypass auto-detection.

### Temperature

Generation temperature is set to `0.1` for consistent, low-variance output. Increase it in `generate_demo()` if you want more creative variation between runs.

## Troubleshooting

**"Missing API key" error in the UI**
- Check that `secrets/.env` exists and contains the correct key for your chosen provider.
- Verify there are no extra quotes or whitespace around the key value.
- As a quick workaround, paste the key in the "LLM API Key (optional fallback)" field.

**"The anthropic/openai package is not installed"**
- Run `pip install -r requirements.txt` inside your activated virtualenv.

**Port already in use**
- Run with a different port: `PORT=7861 python demo_wizard.py`
- Or find and stop whatever is using 7860: `lsof -i :7860`

**Provider mismatch (LLM_PROVIDER set but wrong key)**
- If `LLM_PROVIDER=anthropic` but only `OPENAI_API_KEY` is set, the app will fail. Either switch the provider or add the matching key.
- Set `LLM_PROVIDER=auto` to let the app pick whichever key it finds.

**Empty or garbled response**
- Verify your API key has access to the configured model.
- Check network connectivity to the provider's API endpoint.
- The debug detail in the error output will include the upstream exception message.

## Roadmap / Ideas

- Multi-step plan refinement (iterate on a generated plan with follow-up questions).
- Cost estimation hooks (integrate pricing API or lookup tables).
- Scenario library (save and reload past demo configurations).
- CoreWeave-specific presets (pre-filled workload/GPU combos for common customer profiles).
- Export to PDF or slide-ready format.
- Side-by-side comparison of two cluster configurations.
- Persistent history of generated plans with search.

## License and Credits

This project was created as a hackathon/demo tool for CoreWeave-style GPU cloud environments as part of the **More. Faster. Better. 2026** hackathon.

License: TBD -- add a `LICENSE` file to formalize.
