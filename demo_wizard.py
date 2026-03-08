"""
CoreWeave Demo Wizard - Hackathon MVP
Single-file Gradio + Claude prototype for More. Faster. Better. 2026.
"""

# =============================
# Imports
# =============================
import os
import tempfile
from datetime import datetime
import textwrap
from typing import Optional, Tuple

import gradio as gr

try:
    import anthropic
except ImportError:
    anthropic = None


# =============================
# Prompt Configuration
# =============================
SYSTEM_PROMPT: str = textwrap.dedent(
    """
    You are the "CoreWeave Demo Wizard" assistant for internal MVP demo generation.

    Primary objective:
    Produce practical, credible, structured markdown that helps a user quickly understand
    a recommended CoreWeave-style GPU cluster demo path from their workload inputs.

    Safety and factual constraints (MANDATORY):
    1) Be factual and cautious. Do not fabricate certifications, benchmarks, customer logos,
       contract status, compliance approvals, or production claims.
    2) Only use these public-stat guardrails when contextually relevant and phrase carefully:
       - "96-98% goodput" for scaled training efficiency style framing.
       - "40%+ inference throughput improvement on H200 vs H100" as directional public benchmark framing.
       - "up to 2.86x per-chip (GB200-class vs H100-class)" as directional public benchmark framing.
    3) For high-stakes/government-like scenarios, explicitly label as hypothetical and focus on:
       secure enclaves, reliability, and FedRAMP pursuit alignment language.
       Never claim FedRAMP authorization or accreditation.
    4) If user input is vague, make reasonable assumptions and list them in "Notes & Assumptions".

    Output contract (strict headings in order):
    ## Recommended Cluster Configuration
    ## Performance & Business Impact
    ## Sample Deployment Snippet
    ## Production Next Steps
    ## Notes & Assumptions

    Formatting requirements:
    - Keep concise, practical, and demo-ready.
    - Include at least one code block in Sample Deployment Snippet (YAML or Slurm).
    - Mention InfiniBand only when relevant.
    - Include CTA-style next steps (e.g., contact sales or submit ARENA interest form).
    """
).strip()


FEW_SHOT_EXAMPLES: str = textwrap.dedent(
    """
    Few-shot examples:

    [Example 1 - Normal Training]
    Input:
    - workload_type: LLM Fine-Tuning
    - gpu_count: 128
    - priorities: time-to-train, checkpoint reliability

    Output:
    ## Recommended Cluster Configuration
    Recommend H100/H200-class training nodes with high-bandwidth interconnect and distributed storage assumptions.
    For 128 GPUs, emphasize balanced node topology and fault-aware checkpoint strategy.

    ## Performance & Business Impact
    - At this scale, operators often target high training efficiency; cite 96-98% goodput as directional context.
    - Faster iteration cycles can reduce model tuning timeline risk.

    ## Sample Deployment Snippet
    ```yaml
    cluster:
      gpu_type: H200
      gpu_count: 128
      network: infiniBand
      storage: distributed-parallel
    training:
      strategy: data_parallel
      checkpoint_interval_min: 30
    ```

    ## Production Next Steps
    1. Validate data pipeline and checkpoint I/O assumptions.
    2. Align target training window with platform engineering.
    3. Contact sales / solution architecture for sizing confirmation.

    ## Notes & Assumptions
    - Assumes distributed training framework readiness.
    - Figures are directional and require workload-specific validation.

    [Example 2 - Inference]
    Input:
    - workload_type: Real-Time Inference API
    - gpu_count: 64
    - priorities: low latency p95, high throughput, cost control

    Output:
    ## Recommended Cluster Configuration
    Recommend H200-class inference pool with autoscaling partitions and latency-focused serving stack.

    ## Performance & Business Impact
    - Use 40%+ H200 vs H100 inference throughput improvement as a directional benchmark reference.
    - Supports higher request volume while maintaining p95 latency objectives.

    ## Sample Deployment Snippet
    ```bash
    # Slurm example
    sbatch --nodes=8 --gpus-per-node=8 --partition=inference api_serve.slurm
    ```

    ## Production Next Steps
    1. Run latency/throughput load test with representative prompts.
    2. Tune concurrency and batch windows.
    3. Submit ARENA interest form for follow-up validation.

    ## Notes & Assumptions
    - Assumes model is already optimized for inference runtime.

    [Example 3 - High-Stakes Hypothetical]
    Input:
    - workload_type: Secure Large-Scale Operational Simulation (Hypothetical)
    - gpu_count: 256
    - priorities: secure enclaves, reliability, auditability

    Output:
    ## Recommended Cluster Configuration
    For this hypothetical high-stakes simulation scenario, recommend GB200/H200-class architecture with secure enclave support and resilient multi-zone design assumptions.

    ## Performance & Business Impact
    - For heavy simulation workloads, up to 2.86x per-chip GB200-class directional framing can be used cautiously.
    - Emphasize reliability and controlled execution environments.

    ## Sample Deployment Snippet
    ```yaml
    security:
      secure_enclaves: enabled
      audit_logging: strict
    reliability:
      multi_zone_failover: true
    ```

    ## Production Next Steps
    1. Perform threat model review and enclave compatibility checks.
    2. Align controls with FedRAMP pursuit alignment requirements (no authorization claim).
    3. Validate resilience with fault-injection simulation.

    ## Notes & Assumptions
    - This is a hypothetical scenario; no government authorization is implied.
    """
).strip()


WORKLOAD_OPTIONS: list[str] = [
    "LLM Fine-Tuning",
    "Batch Inference",
    "Real-Time Inference API",
    "RAG / Knowledge Retrieval",
    "Computer Vision Training",
    "Secure Large-Scale Operational Simulation (Hypothetical)",
    "High-Volume Intelligence Data Processing (Hypothetical)",
]


# =============================
# Core Helpers
# =============================
def load_api_key(env_key: str, fallback_key: str) -> str:
    """Resolve Anthropic API key from env first, then fallback input."""
    env_val = (env_key or "").strip()
    fallback_val = (fallback_key or "").strip()
    resolved = env_val or fallback_val
    if not resolved:
        raise ValueError(
            "Missing API key. Set ANTHROPIC_API_KEY or provide key in UI fallback field."
        )
    return resolved


def build_user_prompt(workload_type: str, gpu_count: int, priorities: str) -> str:
    """Create user prompt payload with hard output constraints and scenario handling."""
    is_hypothetical = "Hypothetical" in workload_type
    priorities_clean = (priorities or "").strip() or "No explicit priorities provided."

    return textwrap.dedent(
        f"""
        User selections:
        - Workload Type: {workload_type}
        - GPU Count: {gpu_count}
        - Priorities: {priorities_clean}

        Generation requirements:
        1) Follow exact markdown section headings and order.
        2) Recommend practical cluster shape using H100/H200/GB200-class language as appropriate.
        3) Include networking/storage assumptions; mention InfiniBand only if relevant.
        4) Include cautious benchmark framing only where relevant.
        5) Include at least one fenced code block (YAML or Slurm).
        6) Include concrete production next steps and a CTA (sales contact or ARENA interest form).
        7) Explicitly list assumptions.

        High-stakes handling:
        - Is high-stakes/hypothetical scenario: {is_hypothetical}
        - If true, explicitly label scenario as hypothetical and avoid claiming any granted certification/authorization.
        - Mention secure enclaves, reliability, and FedRAMP pursuit alignment only in non-claim language.

        Return only final markdown output.
        """
    ).strip()


def extract_text_from_response(resp) -> str:
    """Extract text from Anthropic response content blocks with safe fallbacks."""
    try:
        content = getattr(resp, "content", None)
        if content and isinstance(content, list):
            chunks = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    chunks.append(getattr(block, "text", ""))
                else:
                    maybe_text = getattr(block, "text", None)
                    if maybe_text:
                        chunks.append(maybe_text)
            merged = "\n".join(c for c in chunks if c).strip()
            if merged:
                return merged

        raw_text = getattr(resp, "text", None)
        if isinstance(raw_text, str) and raw_text.strip():
            return raw_text.strip()

        return str(resp)
    except Exception:
        return str(resp)


# =============================
# Main Generation Callback
# =============================
def generate_demo(
    workload_type: str,
    gpu_count: int,
    priorities: str,
    api_key_fallback: str,
) -> Tuple[str, Optional[str]]:
    """Generate markdown demo output and optional markdown export file."""
    try:
        if anthropic is None:
            raise RuntimeError(
                "The anthropic package is not installed. Install with: pip install anthropic"
            )

        key = load_api_key(os.environ.get("ANTHROPIC_API_KEY", ""), api_key_fallback)
        client = anthropic.Anthropic(api_key=key)

        user_prompt = build_user_prompt(
            workload_type=workload_type,
            gpu_count=int(gpu_count),
            priorities=priorities,
        )

        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1800,
            temperature=0.2,
            system=f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_EXAMPLES}",
            messages=[{"role": "user", "content": user_prompt}],
        )

        markdown_text = extract_text_from_response(response).strip()
        if not markdown_text:
            markdown_text = (
                "## Error\n\n"
                "The model returned an empty response. Please retry with clearer priorities."
            )

        file_path: Optional[str] = None
        try:
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".md",
                prefix=f"demo_wizard_{stamp}_",
                delete=False,
            ) as tmp:
                tmp.write(markdown_text)
                file_path = tmp.name
        except Exception:
            file_path = None

        return markdown_text, file_path

    except ValueError as ve:
        return (
            "## Setup Required\n\n"
            f"{ve}\n\n"
            "**Quick fix:**\n"
            "- Export `ANTHROPIC_API_KEY` in your shell, or\n"
            "- Paste key in the fallback key field and retry.",
            None,
        )
    except Exception as ex:
        return (
            "## Generation Error\n\n"
            "Sorry, the request could not be completed.\n\n"
            "**Check the following:**\n"
            "1. API key is valid and has model access.\n"
            "2. `anthropic` package is installed (`pip install anthropic`).\n"
            "3. Network/API service is reachable.\n\n"
            f"**Debug detail:** `{str(ex)}`",
            None,
        )


# =============================
# UI Builder
# =============================
def build_interface() -> gr.Blocks:
    """Build and return Gradio Blocks interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# CoreWeave Demo Wizard – Hackathon MVP")
        gr.Markdown(
            "Generate personalized GPU cluster demos in seconds. "
            "Prototype for More. Faster. Better. 2026."
        )

        with gr.Row():
            workload_input = gr.Dropdown(
                choices=WORKLOAD_OPTIONS,
                value=WORKLOAD_OPTIONS[0],
                label="Workload Type",
            )
            gpu_count_input = gr.Slider(
                minimum=1,
                maximum=1024,
                step=1,
                value=64,
                label="GPU Count",
            )

        priorities_input = gr.Textbox(
            label="Priorities",
            lines=4,
            placeholder="e.g., low latency p95, throughput, reliability, cost control",
        )

        api_key_fallback_input = gr.Textbox(
            label="Anthropic API Key (optional fallback)",
            type="password",
            placeholder="sk-ant-...",
        )

        generate_btn = gr.Button("Generate Demo")

        markdown_output = gr.Markdown(label="Generated Demo")
        file_output = gr.File(label="Download Markdown Report (Optional)")

        generate_btn.click(
            fn=generate_demo,
            inputs=[
                workload_input,
                gpu_count_input,
                priorities_input,
                api_key_fallback_input,
            ],
            outputs=[markdown_output, file_output],
        )

    return demo


# =============================
# Entrypoint
# =============================
def main() -> None:
    """Launch the Gradio app."""
    app = build_interface()

    port_raw = os.environ.get("PORT", "7860").strip()
    try:
        port = int(port_raw)
    except ValueError:
        port = 7860

    app.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
