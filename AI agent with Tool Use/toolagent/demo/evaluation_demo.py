"""
Evaluation Demo — Run the evaluation harness against the golden dataset.

Run:    python -m demo.evaluation_demo
"""

import asyncio

from tests.evaluation.dataset import create_golden_dataset
from tests.evaluation.evaluator import Evaluator


async def main() -> None:
    print("=" * 60)
    print("ToolAgent — Evaluation Harness Demo")
    print("=" * 60)

    # Load the golden dataset
    dataset = create_golden_dataset()
    print(f"\n📊 Dataset: {dataset.name}")
    print(f"   Total examples: {len(dataset.examples)}")
    print(f"   Tool coverage: {dataset.tool_coverage}")

    # Create evaluator and run
    evaluator = Evaluator(dataset)
    print("\n🔍 Running evaluation...\n")

    result = await evaluator.run(verbose=True)

    # Print report
    report = evaluator.print_report(result)
    print(f"\n{report}")

    # Summary line
    print("\n" + "=" * 60)
    print(f"Pass Rate: {result.pass_rate * 100:.1f}% | "
          f"Selection Accuracy: {result.avg_tool_selection_score * 100:.1f}% | "
          f"Avg Steps: {result.avg_steps:.1f} | "
          f"Cost: ${result.total_cost_usd:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
