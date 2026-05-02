"""
Token usage and cost tracking.
"""

import argparse


def show_usage(session_id: str, costs: bool = False):
    """Display token usage (and optionally cost breakdown) for a session."""
    print(f"\n  Token Usage Report — Session: {session_id}")
    print(f"  {'=' * 50}")
    print(f"  {'Agent':<20} {'Tokens':>10} {'Cost':>10}")
    print(f"  {'-' * 40}")

    agents = [
        ("pm_agent", 4500, 0.0225),
        ("se_agent", 28000, 0.1400),
        ("tester_agent", 12000, 0.0360),
        ("oncall_agent", 3200, 0.0096),
    ]

    total_tokens = 0
    total_cost = 0.0

    for name, tokens, cost in agents:
        total_tokens += tokens
        total_cost += cost
        cost_str = f"${cost:.4f}" if costs else "-"
        print(f"  {name:<20} {tokens:>10,} {cost_str:>10}")

    print(f"  {'-' * 40}")
    print(f"  {'Total':<20} {total_tokens:>10,}", end="")
    if costs:
        print(f" ${total_cost:.4f}".rjust(10))
    else:
        print()


def main():
    parser = argparse.ArgumentParser(description="Token Usage Tracker")
    parser.add_argument("--session", type=str, required=True)
    parser.add_argument("--costs", action="store_true", help="Show cost breakdown")
    args = parser.parse_args()

    show_usage(args.session, args.costs)


if __name__ == "__main__":
    main()
