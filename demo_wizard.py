"""
CoreWeave Atlas - Hackathon MVP
Single-file Gradio app with OpenAI/Anthropic support for More. Faster. Better. 2026.
"""

# =============================
# Imports
# =============================
import os
import tempfile
from datetime import datetime
import textwrap
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

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
    2) Only use these benchmark guardrails when contextually relevant and phrase carefully:
       - Goodput: "CoreWeave infrastructure has demonstrated 96-98% goodput on scaled H100/H200 clusters (directional from public reports and MLPerf benchmarks)".
       - H200 training: emphasize higher memory capacity and bandwidth (141GB HBM3e per GPU) for larger models/batches.
       - H200 inference: "up to ~40% higher throughput vs H100 on models like Llama 2 70B inference (directional MLPerf)".
       - GB200: "up to 2.86x per-chip speedup vs H200 on large models like Llama 3.1 405B (directional MLPerf)".
       Use benchmark language at most once per output.
    3) For high-stakes/government-like scenarios, explicitly label as hypothetical and focus on:
       secure enclaves, reliability, and FedRAMP pursuit alignment language.
       Never claim FedRAMP authorization or accreditation.
    4) If user input is vague, make reasonable assumptions and list them in "Notes & Assumptions".
    5) Ensure YAML/Slurm snippets use valid syntax and correct spelling/casing:
       H200 (not H2O00), InfiniBand (capitalized), data_parallel, distributed_parallel.
    6) Prefer realistic snippet structure based on public CoreWeave-style examples:
       K8s-style keys such as resources and nodeSelector, or Slurm sbatch directives.
    7) Keep each section concise; avoid repeating the same benefit across sections.
    8) In Production Next Steps, include a concrete CTA when relevant:
       "Explore ARENA lab access for production-scale validation (submit interest at coreweave.com/arena)".
    9) Additional output guidelines:
       - Goodput: use "CoreWeave infrastructure has demonstrated 96-98% goodput on scaled H100/H200 clusters
         (directional from public reports and MLPerf benchmarks)" only once, when workload scale justifies it.
       - H200: for training/fine-tuning, focus on higher memory capacity (141GB HBM3e) and bandwidth for
         larger models/batches; reserve "up to ~40% higher throughput vs H100" for inference workloads.
       - GB200: use "up to 2.86x per-chip performance improvement on large models like Llama 3.1 405B
         (directional from CoreWeave MLPerf Inference v5.0 results vs H200-class)" only for relevant
         scale/workloads.
       - Next Steps: include "submit an ARENA interest form at coreweave.com/arena for production-scale
         validation" as a CTA when relevant.
       - Avoid inventing compliance keys in YAML; prefer descriptive notes such as
         "secure_enclaves: enabled (hypothetical)".
       - Keep YAML realistic and minimal; prefer simple config over complex K8s unless explicitly requested.
    10) For workloads involving very large datasets (e.g., 100TB or petabyte-scale):
       - Emphasize CoreWeave high-performance storage solutions, such as AI Object Storage with
         distributed-parallel access patterns.
       - Directionally mention throughput framing such as "up to several GB/s per GPU for data loading"
         or "aggregate reads scaling to hundreds of GiB/s at large cluster sizes" only when relevant.
         Do not fabricate exact numbers.
       - Recommend distributed-parallel storage, InfiniBand networking for multi-node efficiency,
         and async/reliable checkpointing to reduce I/O bottlenecks.
       - In Notes & Assumptions, state that actual performance requires validation with the specific
         data pipeline and workload.
       - Keep references directional and tied to public CoreWeave capabilities such as petabyte-scale
         migrations and large-scale GPU training.

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
    - CoreWeave infrastructure has demonstrated 96-98% goodput on scaled H100/H200 clusters (directional benchmark framing).
    - Faster iteration cycles can reduce model tuning timeline risk.

    ## Sample Deployment Snippet
    ```yaml
    cluster:
      gpu_type: H200
      gpu_count: 128
      network: InfiniBand
      storage: distributed_parallel
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
    - Directional figures are based on public CoreWeave benchmark framing (including MLPerf-style reporting).

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

    [Example 4 - Better YAML Style]
    Input:
    - workload_type: LLM Fine-Tuning
    - gpu_count: 64
    - priorities: cost awareness, reliable checkpoints

    Output:
    ## Recommended Cluster Configuration
    Recommend a Kubernetes batch job pattern on H200-class nodes with explicit node targeting and persistent storage for checkpoints.

    ## Performance & Business Impact
    - For training at this scale, cite 96-98% goodput as directional context only.
    - Structured checkpointing lowers recovery risk and wasted compute during retries.

    ## Sample Deployment Snippet
    ```yaml
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: llm-finetune-job
    spec:
      template:
        spec:
          nodeSelector:
            accelerator: h200
          containers:
            - name: trainer
              image: coreweave/pytorch:latest
              resources:
                limits:
                  nvidia.com/gpu: 8
              env:
                - name: CHECKPOINT_DIR
                  value: /mnt/checkpoints
          restartPolicy: Never
    ```

    ## Production Next Steps
    1. Validate checkpoint write/read throughput under failure-restart tests.
    2. Tune batch size and gradient accumulation for utilization and cost targets.
    3. Confirm cluster sizing with solution architecture.

    ## Notes & Assumptions
    - Snippet is illustrative and should be adapted to your cluster policies.

    [Example: Large-Scale ML Training with Massive Dataset]
    Input:
    - workload_type: Large-Scale ML Training with Massive Dataset
    - gpu_count: 1024
    - priorities: handle 100TB dataset efficiently, high storage I/O throughput, maximum training efficiency, checkpoint reliability

    Output:
    ## Recommended Cluster Configuration
    For training on massive datasets (e.g., 100TB+ scale), recommend GB200 or H200-class nodes with InfiniBand networking and distributed-parallel storage (for example, CoreWeave AI Object Storage) for high-throughput data access. At 1024 GPUs, use multi-node topology with fault-tolerant design.

    ## Performance & Business Impact
    - Directional: CoreWeave storage solutions support high per-GPU throughput (several GB/s range) and aggregate reads scaling significantly at large cluster sizes, enabling efficient data loading for massive datasets.
    - High goodput (96-98% directional) and reliable checkpointing minimize I/O-related stalls and progress loss.

    ## Sample Deployment Snippet
    ```yaml
    cluster:
      gpu_type: GB200
      gpu_count: 1024
      network: InfiniBand
      storage: distributed_parallel
    training:
      strategy: data_parallel
      checkpoint:
        async: enabled
        interval_minutes: 15
    data:
      throughput_priority: high
      dataset_size_tb: "100+"
    ```

    ## Production Next Steps
    1. Validate storage and loader performance with a representative 100TB+ shard layout.
    2. Run end-to-end checkpoint and restore tests under failure conditions.
    3. Submit an ARENA interest form at coreweave.com/arena for production-scale validation.

    ## Notes & Assumptions
    - Figures are directional from public benchmark framing and require workload-specific validation.
    - Actual throughput depends on data pipeline design, preprocessing, and storage access patterns.
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
    "Large-Scale ML Training with Massive Dataset",
]


# =============================
# Core Helpers
# =============================
def load_api_key(env_key: str, fallback_key: str) -> str:
    """Resolve API key from env first, then fallback input."""
    env_val = (env_key or "").strip()
    fallback_val = (fallback_key or "").strip()
    resolved = env_val or fallback_val
    if not resolved:
        raise ValueError(
            "Missing API key. Set provider key in env file or provide key in UI fallback field."
        )
    return resolved


def resolve_provider() -> str:
    """
    Resolve provider from env.
    Supported values: anthropic, openai, auto (default).
    """
    raw_provider = (os.environ.get("LLM_PROVIDER", "auto") or "auto").strip().lower()
    if raw_provider in {"anthropic", "openai"}:
        return raw_provider

    has_anthropic_key = bool((os.environ.get("ANTHROPIC_API_KEY", "") or "").strip())
    has_openai_key = bool((os.environ.get("OPENAI_API_KEY", "") or "").strip())
    if has_anthropic_key:
        return "anthropic"
    if has_openai_key:
        return "openai"
    return "openai"


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
           - For training, prefer H200 memory capacity/bandwidth framing (141GB HBM3e).
           - Reserve "~40% higher throughput vs H100" framing for inference-oriented contexts.
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
    """Extract text from provider response object with safe fallbacks."""
    try:
        # Anthropic messages API path.
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

        # Chat Completions API primary path.
        choices = getattr(resp, "choices", None)
        if choices and isinstance(choices, list):
            first = choices[0]
            message = getattr(first, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()

        # Responses API fallback path (if app migrates APIs later).
        out_text = getattr(resp, "output_text", None)
        if isinstance(out_text, str) and out_text.strip():
            return out_text.strip()

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
        provider = resolve_provider()

        user_prompt = build_user_prompt(
            workload_type=workload_type,
            gpu_count=int(gpu_count),
            priorities=priorities,
        )

        if provider == "anthropic":
            if anthropic is None:
                raise RuntimeError(
                    "The anthropic package is not installed. Install with: pip install anthropic"
                )
            key = load_api_key(
                os.environ.get("ANTHROPIC_API_KEY", ""),
                api_key_fallback,
            )
            client = anthropic.Anthropic(api_key=key)
            model = (
                os.environ.get("ANTHROPIC_MODEL", "") or "claude-3-5-sonnet-latest"
            ).strip()
            response = client.messages.create(
                model=model,
                max_tokens=1800,
                temperature=0.1,
                system=f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_EXAMPLES}",
                messages=[{"role": "user", "content": user_prompt}],
            )
        else:
            if OpenAI is None:
                raise RuntimeError(
                    "The openai package is not installed. Install with: pip install openai"
                )
            key = load_api_key(os.environ.get("OPENAI_API_KEY", ""), api_key_fallback)
            client = OpenAI(api_key=key)
            model = (os.environ.get("OPENAI_MODEL", "") or "gpt-4o-mini").strip()
            response = client.chat.completions.create(
                model=model,
                max_tokens=1800,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_EXAMPLES}"},
                    {"role": "user", "content": user_prompt},
                ],
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
            "- Set `LLM_PROVIDER` + provider API key in `secrets/.env`, or\n"
            "- Paste key in the fallback key field and retry.",
            None,
        )
    except Exception as ex:
        return (
            "## Generation Error\n\n"
            "Sorry, the request could not be completed.\n\n"
            "**Check the following:**\n"
            "1. API key is valid and has model access.\n"
            "2. Provider package is installed (`pip install openai anthropic`).\n"
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
        gr.Markdown("# CoreWeave Atlas – Hackathon MVP")
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
            label="LLM API Key (optional fallback)",
            type="password",
            placeholder="sk-... or sk-ant-...",
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
    # Load key from local secrets file if present.
    load_dotenv(Path(__file__).resolve().parent / "secrets" / ".env")

    app = build_interface()

    port_raw = os.environ.get("PORT", "7860").strip()
    try:
        port = int(port_raw)
    except ValueError:
        port = 7860

    app.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
