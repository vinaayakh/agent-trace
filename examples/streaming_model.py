"""Direct model call with streaming — auto-traced by agent_trace.

No agent()/step()/tool() wrappers needed. The httpx interceptor captures
the full SSE stream and emits a span with token counts and completion text.

Run:
    AGENT_TRACE_EXPORTER=console python examples/streaming_model.py --provider anthropic
    AGENT_TRACE_EXPORTER=console python examples/streaming_model.py --provider openai
"""
from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv

import agent_trace

load_dotenv()


async def stream_anthropic() -> None:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    async with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": "Count to 5, one word per line."}],
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
    print()


async def stream_openai() -> None:
    import openai

    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count to 5, one word per line."}],
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        text = chunk.choices[0].delta.content or "" if chunk.choices else ""
        print(text, end="", flush=True)
    print()


async def main(provider: str) -> None:
    agent_trace.init(service_name="streaming-demo")
    if provider == "anthropic":
        await stream_anthropic()
    else:
        await stream_openai()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    args = parser.parse_args()
    asyncio.run(main(args.provider))
