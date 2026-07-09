"""Grounded answer generation: numbered sources, mandatory [n] citations,
explicit refusal when the retrieved chunks don't support an answer.

On financial data a hallucinated figure is disqualifying, so the system prompt
forbids answering outside the sources and demands verbatim numbers; the evals
(Jalon 6) check citation validity and numeric exactness against the golden
dataset.
"""

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from secrag.config import get_settings
from secrag.retrieval.search import RetrievedChunk

REFUSAL_SENTENCE = "I cannot answer this from the provided filings."

SYSTEM_PROMPT = (
    "You answer questions about SEC 10-K annual reports using ONLY the numbered"
    " sources provided in the user message.\n\n"
    "Rules:\n"
    "- Every factual claim must cite its source with a [n] marker.\n"
    "- Copy figures exactly as written in the sources: no rounding, no unit conversion.\n"
    "- If the sources do not contain the information needed to answer,"
    f' reply exactly: "{REFUSAL_SENTENCE}"\n'
    "- Never use outside knowledge, even when you are confident.\n"
    "- Be concise."
)

# USD per million tokens (input, output); used for the cost-per-query metric.
PRICING_USD_PER_MTOK = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def source_label(chunk: RetrievedChunk) -> str:
    item = f"Item {chunk.item.upper()}" if chunk.item else "Cover"
    return f"{chunk.ticker} 10-K FY{chunk.fiscal_year}, {item}"


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    sources = "\n\n---\n\n".join(
        f"[{i}] ({source_label(c)})\n{c.content}" for i, c in enumerate(chunks, 1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICING_USD_PER_MTOK:
        return None
    in_rate, out_rate = PRICING_USD_PER_MTOK[model]
    return round((input_tokens * in_rate + output_tokens * out_rate) / 1_000_000, 6)


async def generate_answer(question: str, chunks: list[RetrievedChunk]) -> AsyncIterator[dict]:
    """Yield {"type": "token", ...} events then a final {"type": "done", ...}."""
    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    async with client.messages.stream(
        model=settings.generation_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(question, chunks)}],
    ) as stream:
        async for text in stream.text_stream:
            yield {"type": "token", "text": text}
        final = await stream.get_final_message()
    usage = final.usage
    yield {
        "type": "done",
        "model": final.model,
        "stop_reason": final.stop_reason,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": estimate_cost_usd(
            settings.generation_model, usage.input_tokens, usage.output_tokens
        ),
    }
