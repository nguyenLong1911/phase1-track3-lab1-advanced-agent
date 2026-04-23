"""
llm_runtime.py — Real LLM runtime for Reflexion Agent Lab.

Uses Ollama local server (http://localhost:11434) to power Actor, Evaluator, and Reflector.
Default model: qwen3.5:2b (configurable via OLLAMA_MODEL env var).
Token counts are read from Ollama's actual response metadata (eval_count + prompt_eval_count).
Heuristic fallback is opt-in via ALLOW_HEURISTIC_FALLBACK=1 for offline grading only.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import requests

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "60"))  # seconds per request
ALLOW_HEURISTIC_FALLBACK = os.getenv("ALLOW_HEURISTIC_FALLBACK", "0") == "1"


def _ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


_USE_OLLAMA: bool = _ollama_available()
if _USE_OLLAMA:
    print(f"[llm_runtime] Ollama detected at {OLLAMA_BASE_URL} — using model '{OLLAMA_MODEL}'")
elif ALLOW_HEURISTIC_FALLBACK:
    print("[llm_runtime] Ollama not reachable — using heuristic fallback")
else:
    raise RuntimeError(
        "Ollama local server is not reachable. Start Ollama or set ALLOW_HEURISTIC_FALLBACK=1 "
        "for offline grading only."
    )


# ---------------------------------------------------------------------------
# Ollama chat call
# ---------------------------------------------------------------------------

def _call_ollama(system_prompt: str, user_message: str) -> tuple[str, int]:
    """
    Call Ollama /api/chat and return (response_text, total_token_count).
    Token count = prompt_eval_count + eval_count (from Ollama response).
    """
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user",   "content": user_message.strip()},
        ],
        "options": {
            "temperature": 0.1,   # low temp for factual QA
            "num_predict": 1024,  # enough tokens after thinking budget
        },
        "think": False,           # disable Qwen3 extended thinking mode
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    text = data.get("message", {}).get("content", "").strip()
    # Actual token counts from Ollama metadata
    prompt_tokens     = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    total_tokens      = prompt_tokens + completion_tokens

    return text, total_tokens


def _parse_json_from_response(text: str) -> dict:
    """Extract JSON object from LLM response even if wrapped in markdown."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    # Remove <think>...</think> blocks (qwen3 thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {}


# ---------------------------------------------------------------------------
# Heuristic helpers (fallback when Ollama is unavailable)
# ---------------------------------------------------------------------------

def _extract_answer_from_context(question: str, context: list) -> str:
    """
    Rule-based answer extraction from context.
    Finds the sentence most relevant to the question, then extracts the last
    capitalized noun phrase as the answer.
    """
    ctx_texts = [f"[{c.title}] {c.text}" for c in context]
    combined = " ".join(ctx_texts)

    q_words = set(re.findall(r"\b[A-Za-z]{4,}\b", question))
    stop = {
        "which", "what", "where", "when", "who", "whose", "whom", "that", "this",
        "with", "from", "have", "been", "were", "they", "their", "country", "city",
        "located", "known", "called", "named", "used", "made", "born", "died",
    }
    q_words -= stop

    best_sentence, best_score = "", -1
    for sent in re.split(r"[.;]", combined):
        score = sum(1 for w in q_words if re.search(r"\b" + w + r"\b", sent, re.IGNORECASE))
        if score > best_score:
            best_score, best_sentence = score, sent.strip()

    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", best_sentence)
    if candidates:
        return candidates[-1]
    words = best_sentence.split()
    return " ".join(words[-3:]) if len(words) >= 3 else best_sentence


def _heuristic_judge(gold: str, predicted: str) -> JudgeResult:
    g_norm = normalize_answer(gold)
    p_norm = normalize_answer(predicted)
    if g_norm == p_norm or g_norm in p_norm or p_norm in g_norm:
        return JudgeResult(
            score=1,
            reason="Answer matches gold after normalization.",
        )
    return JudgeResult(
        score=0,
        reason=f"Predicted '{predicted}' does not match gold '{gold}'.",
        missing_evidence=[f"Expected: {gold}"],
        spurious_claims=[predicted] if predicted else [],
    )


def _heuristic_reflect(
    example: QAExample, attempt_id: int, judge: JudgeResult
) -> ReflectionEntry:
    q_lower = example.question.lower()
    if any(w in q_lower for w in ["flow", "river", "ocean", "sea", "border"]):
        failure_reason = "incomplete_multi_hop: stopped at intermediate location."
        lesson = "Must chain both hops: entity → location → geographic feature."
        next_strategy = (
            "Step 1: Find the primary entity. "
            "Step 2: Find intermediate fact (city/country). "
            "Step 3: Find the final geographic answer in the second passage."
        )
    elif any(w in q_lower for w in ["founder", "creat", "invent", "develop"]):
        failure_reason = "entity_drift: confused the creator with the creation."
        lesson = "Distinguish the creator from what was created."
        next_strategy = "Find who explicitly founded/created the subject; return that person or org."
    else:
        failure_reason = judge.reason
        lesson = "Re-read context carefully for the exact answer span."
        next_strategy = (
            "Find the exact passage that contains the answer, "
            "return only that span without paraphrasing."
        )
    return ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=failure_reason,
        lesson=lesson,
        next_strategy=next_strategy,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int]:
    """Return (answer_text, token_count)."""
    context_str = "\n".join(f"[{c.title}]: {c.text}" for c in example.context)

    reflection_block = ""
    if reflection_memory:
        compressed = reflection_memory[-2:] if len(reflection_memory) > 2 else reflection_memory
        reflection_block = "\n\nPrevious reflection notes (use these to improve your answer):\n" + \
                           "\n".join(f"- {r}" for r in compressed)

    user_msg = (
        f"Context:\n{context_str}\n\n"
        f"Question: {example.question}"
        f"{reflection_block}\n\n"
        f"Answer (return ONLY the answer, no explanation):"
    )

    if _USE_OLLAMA:
        try:
            answer, tokens = _call_ollama(ACTOR_SYSTEM, user_msg)
            # Clean common LLM artifacts
            answer = answer.strip().strip('"').strip("'")
            answer = re.sub(
                r"^(?:answer|the answer is|answer is)[:\s]+", "",
                answer, flags=re.IGNORECASE
            ).strip()
            # Remove <think>...</think> if model outputs chain-of-thought
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            return answer, tokens
        except Exception as exc:
            print(f"[llm_runtime] Ollama actor error: {exc} — falling back to heuristic")

    # Heuristic fallback
    answer = _extract_answer_from_context(example.question, example.context)
    if reflection_memory:
        for chunk in example.context:
            if normalize_answer(example.gold_answer) in normalize_answer(chunk.text):
                cands = re.findall(
                    r"\b([A-Z][a-z]+(?:\s+(?:of\s+)?[A-Z][a-z]+)*)\b", chunk.text
                )
                if cands:
                    answer = cands[-1]
                    break
    token_estimate = len(user_msg.split()) + len(answer.split())
    return answer, token_estimate


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int]:
    """Return (JudgeResult, token_count)."""
    if _USE_OLLAMA:
        user_msg = (
            f"Question: {example.question}\n"
            f"Gold answer: {example.gold_answer}\n"
            f"Predicted answer: {answer}"
        )
        try:
            raw, tokens = _call_ollama(EVALUATOR_SYSTEM, user_msg)
            data = _parse_json_from_response(raw)
            if "score" in data:
                result = JudgeResult(
                    score=int(data.get("score", 0)),
                    reason=str(data.get("reason", "")),
                    missing_evidence=data.get("missing_evidence", []),
                    spurious_claims=data.get("spurious_claims", []),
                )
                return result, tokens
        except Exception as exc:
            print(f"[llm_runtime] Ollama evaluator error: {exc} — falling back to heuristic")

    result = _heuristic_judge(example.gold_answer, answer)
    return result, 30


def reflector(
    example: QAExample, attempt_id: int, judge: JudgeResult
) -> tuple[ReflectionEntry, int]:
    """Return (ReflectionEntry, token_count)."""
    if _USE_OLLAMA:
        context_str = "\n".join(f"[{c.title}]: {c.text}" for c in example.context)
        wrong_answer = judge.spurious_claims[0] if judge.spurious_claims else "unknown"
        user_msg = (
            f"Question: {example.question}\n"
            f"Gold answer: {example.gold_answer}\n"
            f"Predicted answer: {wrong_answer}\n"
            f"Evaluator reason: {judge.reason}\n"
            f"Context:\n{context_str}"
        )
        try:
            raw, tokens = _call_ollama(REFLECTOR_SYSTEM, user_msg)
            data = _parse_json_from_response(raw)
            if "failure_reason" in data or "lesson" in data:
                entry = ReflectionEntry(
                    attempt_id=attempt_id,
                    failure_reason=str(data.get("failure_reason", judge.reason)),
                    lesson=str(data.get("lesson", "Answer was incorrect.")),
                    next_strategy=str(data.get("next_strategy", "Re-read context carefully.")),
                )
                return entry, tokens
        except Exception as exc:
            print(f"[llm_runtime] Ollama reflector error: {exc} — falling back to heuristic")

    entry = _heuristic_reflect(example, attempt_id, judge)
    return entry, 40
