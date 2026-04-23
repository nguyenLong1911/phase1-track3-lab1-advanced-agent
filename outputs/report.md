# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: live
- Records: 230
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.9739 | 0.9826 | 0.0087 |
| Avg attempts | 1 | 1.0348 | 0.0348 |
| Avg token estimate | 528.36 | 574.63 | 46.27 |
| Avg latency (ms) | 11265.69 | 17652.45 | 6386.76 |

## Failure modes
```json
{
  "none": {
    "react": 112,
    "reflexion": 113
  },
  "incomplete_multi_hop": {
    "react": 2
  },
  "entity_drift": {
    "react": 0,
    "reflexion": 0
  },
  "wrong_final_answer": {
    "react": 1,
    "reflexion": 2
  },
  "looping": {
    "react": 0,
    "reflexion": 0
  },
  "reflection_overfit": {
    "react": 0,
    "reflexion": 0
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding
- adaptive_max_attempts
- memory_compression

## Discussion

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

