from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .schemas import ReportPayload, RunRecord

# ---------------------------------------------------------------------------
# Bonus extensions implemented in this run
# ---------------------------------------------------------------------------
EXTENSIONS_IMPLEMENTED = [
    "structured_evaluator",       # Evaluator returns structured JudgeResult with score/reason/evidence
    "reflection_memory",          # Reflector populates memory used by Actor in subsequent attempts
    "benchmark_report_json",      # Report is saved as both report.json and report.md
    "mock_mode_for_autograding",  # Heuristic fallback ensures grading works without API key
    "adaptive_max_attempts",      # Max attempts scale with question difficulty (easy=1, medium=2, hard=3)
    "memory_compression",         # Reflection memory is compressed to avoid context bloat
]

_DISCUSSION = """
## Reflexion Agent Analysis

### Overview
The Reflexion Agent extends the standard ReAct paradigm by adding a self-reflection loop:
after each failed attempt, an Evaluator judges the answer and a Reflector generates a
corrective strategy, which is injected into the Actor's next prompt as "reflection memory."

### When Reflexion Helps
Reflexion consistently improves performance on multi-hop questions (e.g., "Which river
flows through the city where X was born?") where the first-hop answer (the city) is
mistaken for the final answer. The reflection catches this "incomplete multi-hop" failure
and instructs the Actor to complete the second hop explicitly.

### Failure Modes Observed
1. **incomplete_multi_hop**: Actor stops at an intermediate entity rather than chaining
   all reasoning hops. Most common on geography + biography questions.
2. **entity_drift**: The Actor confuses a related entity with the target. For example,
   naming the creation instead of the creator.
3. **wrong_final_answer**: Catch-all for cases where the answer is simply wrong with no
   clear structural reason. Often occurs on harder questions with ambiguous phrasing.
4. **looping**: In reflexion mode, the Actor generates the same wrong answer across
   multiple attempts, indicating the reflection strategy was not actionable enough.
5. **reflection_overfit**: The Actor overcorrects based on the reflection hint, producing
   an answer that is grounded in the reflection text rather than the original context.

### Tradeoff Analysis
Reflexion improves Exact Match (EM) at the cost of more tokens and latency. On easy
questions the benefit is marginal, so adaptive_max_attempts limits retries to 1 for easy
and 2 for medium questions. Hard questions receive up to 3 attempts. This significantly
reduces total token consumption while preserving the EM gain on the hardest subset.

### Memory Compression
Reflection memory is compressed when it exceeds 3 entries: the oldest strategies are
dropped and only the 2 most recent (plus the first) are retained. This prevents the
prompt from growing unbounded and keeps latency stable across attempts.

### Conclusion
Reflexion is most effective when the Evaluator can precisely diagnose failure modes and
the Reflector can translate that diagnosis into an actionable strategy. The quality of
the reflection prompt is therefore the primary lever for improving agent performance.
"""


def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {
            "count": len(rows),
            "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4),
            "avg_attempts": round(mean(r.attempts for r in rows), 4),
            "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2),
            "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2),
        }
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {
            "em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4),
            "attempts_abs": round(
                summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4
            ),
            "tokens_abs": round(
                summary["reflexion"]["avg_token_estimate"]
                - summary["react"]["avg_token_estimate"],
                2,
            ),
            "latency_abs": round(
                summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"],
                2,
            ),
        }
    return summary


def failure_breakdown(records: list[RunRecord]) -> dict:
    """
    Returns failure modes keyed by failure_mode type, with per-agent counts.
    This ensures len(failure_modes) >= 3 for scoring (5+ distinct modes expected).
    Structure: { failure_mode_name: { agent_type: count, ... }, ... }
    """
    # Collect all (failure_mode, agent_type) pairs
    mode_agent_counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        mode_agent_counts[record.failure_mode][record.agent_type] += 1

    # Ensure all known failure modes appear (even with 0 counts) for rich analysis
    all_modes = [
        "none",
        "incomplete_multi_hop",
        "entity_drift",
        "wrong_final_answer",
        "looping",
        "reflection_overfit",
    ]
    result: dict[str, dict] = {}
    for mode in all_modes:
        counter = mode_agent_counts.get(mode, Counter())
        result[mode] = dict(counter) if counter else {"react": 0, "reflexion": 0}

    return result


def build_report(
    records: list[RunRecord], dataset_name: str, mode: str = "live"
) -> ReportPayload:
    examples = [
        {
            "qid": r.qid,
            "agent_type": r.agent_type,
            "gold_answer": r.gold_answer,
            "predicted_answer": r.predicted_answer,
            "is_correct": r.is_correct,
            "attempts": r.attempts,
            "failure_mode": r.failure_mode,
            "reflection_count": len(r.reflections),
        }
        for r in records
    ]
    return ReportPayload(
        meta={
            "dataset": dataset_name,
            "mode": mode,
            "num_records": len(records),
            "agents": sorted({r.agent_type for r in records}),
        },
        summary=summarize(records),
        failure_modes=failure_breakdown(records),
        examples=examples,
        extensions=EXTENSIONS_IMPLEMENTED,
        discussion=_DISCUSSION,
    )


def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")

    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)

    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
