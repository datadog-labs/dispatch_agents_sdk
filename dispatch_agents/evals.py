"""Standard payload types for eval-related agent functions.

When an agent function is wired up as an *invoker scorer* on the
experiments runner, the runner sends one HTTP request per case with the
case's input / expected / output joined onto the payload. The shape it
sends by default — when no custom ``input_mapping`` is configured —
is exactly :class:`EvalItem`.

Use this as your scorer function's input type when you don't want to
deal with mapping configuration. Subclass it to add extra fields (e.g.,
your judge needs a rubric) and the experiments UI's "extra_payload"
panel will let you populate them statically.
"""

from typing import Any

from .models import BasePayload


class EvalItem(BasePayload):
    """A single experiment case as sent to an invoker scorer.

    Fields mirror what the experiments runner tracks per case:

    - ``id`` — stable identifier the runner assigns per (experiment,
      item). Most single-item scorers can ignore it; batch scorers use
      it to map their dict-keyed return value back to specific rows.
    - ``input`` — payload the agent was invoked with.
    - ``expected`` — the human-supplied ground truth (may be ``None``).
    - ``output`` — whatever the agent returned.

    All non-id fields are typed ``Any`` because the runner doesn't
    enforce a schema on dataset cases — your agent decides the shape.
    JSON strings are auto-decoded before send, so dict fields land as
    ``dict``.

    Example::

        from dispatch_agents import EvalItem, fn

        class JudgeVerdict(BasePayload):
            score: int
            reason: str

        @fn()
        async def judge(case: EvalItem) -> JudgeVerdict:
            ...

    For batch scoring, see :class:`EvalBatch`.
    """

    id: str = ""
    input: Any = None
    expected: Any = None
    output: Any = None


class EvalBatch(BasePayload):
    """A batch of experiment cases for an invoker scorer.

    Use this when one scorer invocation should evaluate multiple cases
    at once (e.g., an LLM judge that scores N answers in a single
    prompt to amortize overhead). The runner groups cases into batches
    of ``InvokerConfig.batch_size`` before calling the scorer.

    The scorer is expected to return a ``dict[str, ...]`` keyed by
    :attr:`EvalItem.id` so the runner can map results back to rows.
    Missing keys are recorded as scorer errors for the affected
    cases; extra keys are dropped.

    Example::

        from dispatch_agents import EvalBatch, fn

        @fn()
        async def judge_batch(batch: EvalBatch) -> dict[str, dict]:
            results = {}
            for item in batch.items:
                results[item.id] = {"score": grade(item)}
            return results
    """

    items: list[EvalItem]
