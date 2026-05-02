"""
Basic Usage Demo — Quick overview of ToolAgent capabilities.

Run:    python -m demo.basic_usage
"""

import asyncio
import json

from agent.core.agent import AgentConfig, ToolAgent
from agent.tools.registry import create_default_registry


async def main() -> None:
    print("=" * 60)
    print("ToolAgent — Basic Usage Demo")
    print("=" * 60)

    # Create the agent with all default tools
    registry = create_default_registry(demo_mode=True)
    agent = ToolAgent(registry=registry)

    print(f"\n📦 Available tools ({len(registry)}):")
    for name in registry.tool_names:
        print(f"   🔧 {name}")
    print(f"\n🤖 LLM Provider: {agent.llm.provider_name}")

    # ── Example 1: Calculator ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("📐 Example 1: Calculator")
    print("   Prompt: What is (15 * 3) / 5 + 2**10?")
    print("─" * 60)

    session = await agent.run("What is (15 * 3) / 5 + 2**10?")
    print(f"   Status: {session.status.value}")
    print(f"   Steps: {session.step_count}")
    print(f"   Response: {session.final_response[:200]}")

    # Reset for next example
    agent.reset()

    # ── Example 2: Weather + Database ──────────────────────────────
    print("\n" + "─" * 60)
    print("🌤️  Example 2: Weather + Database Query")
    print("   Prompt: What's the weather in Paris? And show me all products.")
    print("─" * 60)

    session = await agent.run("What's the weather in Paris? And show me all products.")
    print(f"   Status: {session.status.value}")
    print(f"   Steps: {session.step_count}")
    for step in session.steps:
        if step.tool_name:
            print(f"   Step {step.step_number}: {step.tool_name} ({step.duration_ms:.0f}ms)")
    print(f"   Response: {session.final_response[:300]}")

    agent.reset()

    # ── Example 3: Code Execution ──────────────────────────────────
    print("\n" + "─" * 60)
    print("💻 Example 3: Code Execution")
    print("   Prompt: Run Python code to print the first 10 Fibonacci numbers")
    print("─" * 60)

    session = await agent.run("Run Python code to print the first 10 Fibonacci numbers")
    print(f"   Status: {session.status.value}")
    print(f"   Steps: {session.step_count}")
    if session.final_response:
        print(f"   Response: {session.final_response[:300]}")

    agent.reset()

    # ── Example 4: Multi-Step Workflow ─────────────────────────────
    print("\n" + "─" * 60)
    print("🔗 Example 4: Multi-Step Workflow")
    print("   Prompt: Get all electronics products from the database and calculate their average price")
    print("─" * 60)

    session = await agent.run("Get all electronics products from the database and calculate their average price")
    print(f"   Status: {session.status.value}")
    print(f"   Steps: {session.step_count}")
    for step in session.steps:
        if step.tool_name:
            print(f"   Step {step.step_number}: {step.tool_name}")
    if session.final_response:
        print(f"   Response: {session.final_response[:300]}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
