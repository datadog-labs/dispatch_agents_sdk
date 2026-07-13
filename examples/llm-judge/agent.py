"""Generic LLM-as-judge agent.

Designed to be wired up as an *invoker scorer* on the experiment runner
(see the evals-experiments PR). Receives a single
:class:`dispatch_agents.EvalItem` per case and returns a binary score
plus a one-sentence reason and confidence.

Because it takes ``EvalItem`` directly, you can attach it as an invoker
scorer without setting any custom ``input_mapping`` — the runner's
default ``{"input": "input", "expected": "expected", "output": "output"}``
mapping already matches this shape.

Response shape::

    {"score": 0 | 1, "reason": "one-sentence justification", "confidence": 0.0-1.0}

The experiments runner splits this dict into one column per top-level
key: ``score`` (binary ✓/✗), ``confidence`` (continuous), ``reason``
(text).
"""

import json
import logging
import re

from dispatch_agents import BasePayload, EvalItem, fn, llm

logger = logging.getLogger(__name__)


def _to_text(value: object) -> str:
    """Render whatever the experiment passed us as a string the LLM can read.

    Cases can carry scalars (when the dataset's input is a plain string)
    or dicts (when the agent took a structured payload). We accept both
    so the judge works without the user knowing the dataset's schema.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


class JudgeResponse(BasePayload):
    """Verdict + justification + confidence.

    The experiments runner decomposes this dict into one metric column
    per top-level key — see the InvokerConfig docstring in
    backend/models/evals.py. ``score`` renders as a binary chip,
    ``confidence`` as a continuous number, ``reason`` as text.
    """

    score: int
    reason: str
    confidence: float = 0.5


PROMPT = """\
You are evaluating whether an AI agent's answer correctly responds to a
question. Compare it against the expected answer if one is provided.

Scoring rubric:
- 1 = The agent's answer is factually correct and responsive to the
  question. Minor phrasing differences from the expected answer are fine
  as long as the core fact is right.
- 0 = The agent's answer is incorrect, irrelevant, refuses to answer,
  or contains material that contradicts the expected answer.

Respond ONLY with a JSON object on a single line:
{"score": 0 or 1, "confidence": 0.0-1.0, "reason": "one sentence justifying the score"}
"""


def _extract_json(text: str) -> dict | None:
    """Pull the first {...} block from the response, tolerating prose."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


@fn()
async def judge(case: EvalItem) -> JudgeResponse:
    """Score a single experiment case.

    ``case.input`` is the agent's input payload, ``case.expected`` is the
    dataset's ground truth (may be missing), ``case.output`` is what the
    agent actually returned.
    """
    input_text = _to_text(case.input)
    expected_text = _to_text(case.expected)
    output_text = _to_text(case.output)
    user_msg = (
        f"Question / input: {input_text or '(none)'}\n\n"
        f"Expected answer: {expected_text or '(none provided)'}\n\n"
        f"Agent's answer: {output_text or '(no answer)'}"
    )

    response = await llm.inference(
        [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": user_msg},
        ]
    )
    raw = (response.content or "").strip()
    parsed = _extract_json(raw)
    if parsed is None:
        logger.warning(
            "LLM judge: failed to parse JSON, defaulting to 0", extra={"raw": raw[:500]}
        )
        return JudgeResponse(
            score=0, reason=f"could not parse judge output: {raw[:200]}"
        )

    score_raw = parsed.get("score")
    if isinstance(score_raw, bool):
        score = 1 if score_raw else 0
    elif isinstance(score_raw, (int, float)):
        score = 1 if score_raw >= 0.5 else 0
    elif isinstance(score_raw, str):
        score = 1 if score_raw.strip() in {"1", "true", "True"} else 0
    else:
        score = 0
    reason = str(parsed.get("reason", "")).strip() or "(no reason)"
    confidence_raw = parsed.get("confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = 0.5
    return JudgeResponse(score=score, reason=reason, confidence=confidence)
