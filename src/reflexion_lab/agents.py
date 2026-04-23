from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from .llm_runtime import actor_answer, evaluator, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

# ---------------------------------------------------------------------------
# Adaptive max-attempts table (bonus: adaptive_max_attempts)
# ---------------------------------------------------------------------------
_DIFFICULTY_MAX_ATTEMPTS: dict[str, int] = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
}

_KNOWN_FAILURE_MODES: dict[str, str] = {
    # Populated dynamically; kept as fallback
}


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    use_adaptive_attempts: bool = False  # bonus: adaptive_max_attempts

    def _get_max_attempts(self, example: QAExample) -> int:
        """Bonus: adaptive_max_attempts — scale retries by question difficulty."""
        if self.use_adaptive_attempts and self.agent_type == "reflexion":
            return _DIFFICULTY_MAX_ATTEMPTS.get(example.difficulty, self.max_attempts)
        return self.max_attempts

    def _compress_memory(self, memory: list[str]) -> list[str]:
        """
        Bonus: memory_compression — keep only the most recent reflection
        strategy if memory exceeds 3 entries to avoid context bloat.
        """
        if len(memory) > 3:
            # Keep the first (original context clue) and last 2 (most recent)
            return [memory[0]] + memory[-2:]
        return memory

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        total_tokens = 0

        effective_max = self._get_max_attempts(example)

        for attempt_id in range(1, effective_max + 1):
            t_start = time.perf_counter()

            # --- Actor ---
            answer, actor_tokens = actor_answer(
                example, attempt_id, self.agent_type, reflection_memory
            )

            # --- Evaluator ---
            judge, eval_tokens = evaluator(example, answer)

            latency_ms = int((time.perf_counter() - t_start) * 1000)
            token_count = actor_tokens + eval_tokens

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=token_count,
                latency_ms=latency_ms,
            )
            traces.append(trace)
            total_tokens += token_count

            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                break

            # --- Reflexion loop (only for reflexion agent, not last attempt) ---
            if self.agent_type == "reflexion" and attempt_id < effective_max:
                reflect_entry, reflect_tokens = reflector(example, attempt_id, judge)
                total_tokens += reflect_tokens
                reflections.append(reflect_entry)

                # Update trace with reflection
                trace.reflection = reflect_entry

                # Build a concise memory string from the reflection
                memory_note = (
                    f"[Attempt {attempt_id}] {reflect_entry.next_strategy}"
                )
                reflection_memory.append(memory_note)

                # Bonus: memory_compression
                reflection_memory = self._compress_memory(reflection_memory)

        # Determine failure mode
        if final_score == 1:
            failure_mode = "none"
        elif self.agent_type == "reflexion" and len(traces) >= effective_max and final_score == 0:
            # Still wrong after all reflexion attempts
            last_reflection = reflections[-1] if reflections else None
            if last_reflection and "multi_hop" in last_reflection.failure_reason:
                failure_mode = "incomplete_multi_hop"
            elif last_reflection and "drift" in last_reflection.failure_reason:
                failure_mode = "entity_drift"
            elif len(traces) > 1 and traces[-1].answer == traces[-2].answer:
                failure_mode = "looping"
            else:
                failure_mode = "wrong_final_answer"
        else:
            # ReAct single-attempt failure
            q_lower = example.question.lower()
            if any(w in q_lower for w in ["river", "ocean", "sea", "flow"]):
                failure_mode = "incomplete_multi_hop"
            elif any(w in q_lower for w in ["founder", "creat", "invent"]):
                failure_mode = "entity_drift"
            else:
                failure_mode = "wrong_final_answer"

        total_latency = sum(t.latency_ms for t in traces)

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        # Enable adaptive_max_attempts bonus feature
        super().__init__(
            agent_type="reflexion",
            max_attempts=max_attempts,
            use_adaptive_attempts=True,
        )
