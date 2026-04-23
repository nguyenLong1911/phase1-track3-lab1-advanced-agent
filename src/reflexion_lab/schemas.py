from __future__ import annotations
from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    title: str
    text: str


class QAExample(BaseModel):
    qid: str
    difficulty: Literal["easy", "medium", "hard"]
    question: str
    gold_answer: str
    context: list[ContextChunk]


class JudgeResult(BaseModel):
    score: int = Field(..., description="1 if correct, 0 if incorrect")
    reason: str = Field(..., description="Explanation of the evaluation result")
    missing_evidence: list[str] = Field(default_factory=list, description="Evidence needed but missing from the answer")
    spurious_claims: list[str] = Field(default_factory=list, description="Wrong claims made by the answer")


class ReflectionEntry(BaseModel):
    attempt_id: int = Field(..., description="Which attempt this reflection is from")
    failure_reason: str = Field(..., description="Why this attempt failed")
    lesson: str = Field(..., description="Lesson learned from this failure")
    next_strategy: str = Field(..., description="Strategy to use in the next attempt")


class AttemptTrace(BaseModel):
    attempt_id: int
    answer: str
    score: int
    reason: str
    reflection: Optional[ReflectionEntry] = None
    token_estimate: int = 0
    latency_ms: int = 0


class RunRecord(BaseModel):
    qid: str
    question: str
    gold_answer: str
    agent_type: Literal["react", "reflexion"]
    predicted_answer: str
    is_correct: bool
    attempts: int
    token_estimate: int
    latency_ms: int
    failure_mode: Literal["none", "entity_drift", "incomplete_multi_hop", "wrong_final_answer", "looping", "reflection_overfit"]
    reflections: list[ReflectionEntry] = Field(default_factory=list)
    traces: list[AttemptTrace] = Field(default_factory=list)


class ReportPayload(BaseModel):
    meta: dict
    summary: dict
    failure_modes: dict
    examples: list[dict]
    extensions: list[str]
    discussion: str


class ReflexionState(TypedDict):
    question: str
    context: list[str]
    trajectory: list[str]
    reflection_memory: list[str]
    attempt_count: int
    success: bool
    final_answer: str
