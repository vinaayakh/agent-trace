"""Same ReAct agent, same tracer — but using the raw OpenAI SDK.

Demonstrates that agent_trace requires zero SDK-specific code.
Swap the client; everything else, including the tracer, is identical.

Run:
    cp sample.env .env  # add OPENAI_API_KEY
    python examples/openai_agent.py
"""
from __future__ import annotations

import asyncio
import os
import re

import openai
from dotenv import load_dotenv

import agent_trace

load_dotenv()

SYSTEM_PROMPT = """\
You are a research assistant. Answer questions by reasoning step-by-step.
When you need to look something up, use this exact format:
  Action: search("your query")
When you have enough information, answer with:
  Final Answer: <your answer>
"""

TOOLS = {
    "search": lambda query: (
        f"Search results for '{query}': "
        "Wikipedia says this is a well-known topic with documented history dating back to 1970."
    ),
}


def run_tool(action_str: str) -> str:
    match = re.match(r'(\w+)\("([^"]*)"\)', action_str.strip())
    if not match:
        return "Tool call format not recognized."
    tool_name, tool_input = match.group(1), match.group(2)
    fn = TOOLS.get(tool_name)
    if fn is None:
        return f"Unknown tool: {tool_name}"
    return fn(tool_input)


async def run_agent(question: str, max_steps: int = 5) -> str:
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    async with agent_trace.agent("ReActAgent"):
        for _ in range(max_steps):
            async with agent_trace.step("think"):
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=512,
                    messages=messages,
                )
                assistant_text = response.choices[0].message.content
                messages.append({"role": "assistant", "content": assistant_text})

            if "Final Answer:" in assistant_text:
                answer = assistant_text.split("Final Answer:")[-1].strip()
                print(f"\nAnswer: {answer}")
                return answer

            action_match = re.search(r"Action:\s*(.+)", assistant_text)
            if action_match:
                action_str = action_match.group(1).strip()
                async with agent_trace.tool("search", input=action_str):
                    observation = run_tool(action_str)

                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                })

        return "Max steps reached without a final answer."


if __name__ == "__main__":
    agent_trace.init(service_name="openai-agent-demo")
    question = "What is the history of the Python programming language?"
    print(f"Question: {question}\n")
    asyncio.run(run_agent(question))
