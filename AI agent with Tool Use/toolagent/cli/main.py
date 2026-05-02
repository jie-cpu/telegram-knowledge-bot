"""CLI entry point for ToolAgent interactive sessions.

Usage:
    toolagent                    # Interactive mode
    toolagent --prompt "..."     # Single query mode
    toolagent --api              # Start API server
    toolagent --list-tools       # List available tools
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

# Rich is optional — fall back to plain print if not installed
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Console = None
    Markdown = None
    Panel = None
    Table = None
    Tree = None

from agent.core.agent import AgentConfig, ToolAgent
from agent.guardrails.input_guardrails import InputContentGuardrail, InputLengthGuardrail
from agent.guardrails.output_guardrails import ToolCallValidationGuardrail
from agent.tools.registry import create_default_registry

console = Console() if HAS_RICH else None


def _print(*args, **kwargs) -> None:
    """Print with or without rich formatting."""
    if HAS_RICH and console:
        console.print(*args, **kwargs)
    else:
        # Strip basic rich markup when rich is not available
        text = " ".join(str(a) for a in args)
        import re
        text = re.sub(r"\[/?\w+(?: \w+=[^]]*)?\]", "", text)
        print(text, **kwargs)


def _input(prompt: str = "") -> str:
    """Input with or without rich styling."""
    if HAS_RICH and console:
        return console.input(prompt)
    # Strip rich markup
    import re
    clean = re.sub(r"\[/?\w+(?: \w+=[^]]*)?\]", "", prompt)
    return input(clean)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ToolAgent — An extensible AI agent with tool use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  toolagent                          Start interactive session
  toolagent -p "What's the weather in Tokyo?"  Single query
  toolagent --list-tools             List all available tools
  toolagent --api                    Start the API server
  toolagent --api --port 8080        Start API on custom port
        """,
    )

    parser.add_argument("-p", "--prompt", type=str, help="Single query to run (non-interactive)")
    parser.add_argument("--list-tools", action="store_true", help="List all available tools")
    parser.add_argument("--api", action="store_true", help="Start the API server")
    parser.add_argument("--port", type=int, default=8000, help="API server port (default: 8000)")
    parser.add_argument("--max-steps", type=int, default=15, help="Maximum agent steps (default: 15)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format (non-interactive)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    return parser


def list_tools() -> None:
    """Display all available tools with their schemas."""
    registry = create_default_registry(demo_mode=True)

    _print("[bold cyan]ToolAgent — Available Tools[/bold cyan]\n")

    if HAS_RICH:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Tool Name", style="cyan")
        table.add_column("Description")
        table.add_column("Parameters")

        for tool in registry.all_tools:
            params = tool.parameters
            required = params.get("required", [])
            props = params.get("properties", {})
            param_desc = ", ".join(
                f"{k} ({'required' if k in required else 'optional'})"
                for k in props.keys()
            )
            table.add_row(tool.name, tool.description, param_desc)

        _print(table)
    else:
        # Plain text fallback
        for tool in registry.all_tools:
            params = tool.parameters
            print(f"  {tool.name}: {tool.description}")
            for p_name, p_info in params.get("properties", {}).items():
                print(f"    - {p_name}: {p_info.get('description', '')}")
        print()

    _print(f"\n[green]Total: {len(registry)} tools[/green]")


async def run_single_query(prompt: str, max_steps: int = 15, json_output: bool = False) -> None:
    """Run a single query and display results."""
    registry = create_default_registry(demo_mode=True)
    config = AgentConfig(max_steps=max_steps, guardrails_enabled=True)
    agent = ToolAgent(registry=registry, config=config)

    input_guardrails = [InputLengthGuardrail(), InputContentGuardrail()]
    output_guardrails = [ToolCallValidationGuardrail()]
    agent.set_guardrails(input_guardrails, output_guardrails)

    if not json_output:
        print(f"\nQuery: {prompt}")
        print("Processing...\n")

    start = time.time()
    session = await agent.run(prompt)
    elapsed = time.time() - start

    if json_output:
        print(json.dumps(session.to_dict(), indent=2, default=str))
        return

    # Display steps
    if session.steps and HAS_RICH:
        tree = Tree("[bold cyan]Agent Reasoning Trace[/bold cyan]")
        for step in session.steps:
            if step.tool_name:
                branch = tree.add(
                    f"[yellow]Step {step.step_number}:[/yellow] "
                    f"[green]{step.tool_name}[/green]"
                    f" [dim]({step.duration_ms:.0f}ms)[/dim]"
                )
                if step.tool_input:
                    branch.add(f"Input: {json.dumps(step.tool_input, default=str)[:200]}")
                if step.error:
                    branch.add(f"[red]Error: {step.error}[/red]")
                elif step.tool_result:
                    output_str = str(step.tool_result.output)[:300]
                    branch.add(f"Result: {output_str}")
            elif step.llm_response and step.llm_response.content:
                tree.add(f"[blue]Step {step.step_number}:[/blue] {step.llm_response.content[:200]}...")
        _print(tree)
    elif session.steps:
        for step in session.steps:
            if step.tool_name:
                status = "✗" if step.error else "✓"
                print(f"  Step {step.step_number} [{status}] {step.tool_name} ({step.duration_ms:.0f}ms)")

    # Display final response
    if HAS_RICH:
        _print("\n" + "─" * 60)
        _print("[bold cyan]Final Response:[/bold cyan]")
        _print(Panel(Markdown(session.final_response or "No response"), border_style="green"))
    else:
        print("\n" + "-" * 60)
        print("Final Response:")
        print(session.final_response or "No response")

    # Display metrics
    print(f"\nStatus: {session.status.value}")
    print(f"Steps: {session.step_count}")
    print(f"Cost: ${session.total_cost_usd:.6f}")
    print(f"Tokens: {session.total_tokens}")
    print(f"Time: {elapsed:.2f}s")


async def run_interactive(max_steps: int = 15) -> None:
    """Run interactive chat session with the agent."""
    registry = create_default_registry(demo_mode=True)
    config = AgentConfig(max_steps=max_steps, guardrails_enabled=True)
    agent = ToolAgent(registry=registry, config=config)

    input_guardrails = [InputLengthGuardrail(), InputContentGuardrail()]
    output_guardrails = [ToolCallValidationGuardrail()]
    agent.set_guardrails(input_guardrails, output_guardrails)

    print("ToolAgent Interactive")
    print("Type 'exit', 'quit', or Ctrl+C to quit")
    print("Type '/tools' to list tools")
    print("Type '/clear' to reset conversation")

    while True:
        try:
            prompt = _input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in ("exit", "quit", "/exit", "/quit"):
            print("Goodbye!")
            break

        if prompt.strip() == "/tools":
            list_tools()
            continue

        if prompt.strip() == "/clear":
            agent.reset()
            print("Conversation reset.")
            continue

        if prompt.strip() == "":
            continue

        start = time.time()
        session = await agent.run(prompt)
        elapsed = time.time() - start

        # Show step summary
        if session.steps:
            step_summary = []
            for step in session.steps:
                if step.tool_name:
                    status = "✗" if step.error else "✓"
                    step_summary.append(f"[{status}] {step.tool_name}")
            if step_summary:
                print("  " + " → ".join(step_summary) + f" ({elapsed:.1f}s)")

        # Show final response
        if HAS_RICH:
            _print(Panel(
                Markdown(session.final_response or "No response"),
                border_style="cyan",
                title="[bold]Agent[/bold]",
                title_align="left",
            ))
        else:
            print(f"\nAgent: {session.final_response}")


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.list_tools:
        list_tools()
        return

    if args.api:
        from api.main import run_server
        print(f"Starting ToolAgent API on port {args.port}...")
        run_server(port=args.port)
        return

    if args.prompt:
        asyncio.run(run_single_query(args.prompt, args.max_steps, args.json))
        return

    # Interactive mode
    asyncio.run(run_interactive(args.max_steps))


if __name__ == "__main__":
    main()
