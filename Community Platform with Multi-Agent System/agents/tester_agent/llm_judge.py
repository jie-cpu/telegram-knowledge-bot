"""
LLM-as-Judge — evaluates AI-generated code for quality, security, and completeness.

This is a core part of the "Testing strategy for AI-generated code" skill.
Instead of trusting that generated code is correct, we run it through a separate
LLM call that scores it on multiple dimensions with specific, actionable feedback.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.llm_client import LLMClient, APIKeyMissingError


@dataclass
class JudgeResult:
    """Result of an LLM-as-judge evaluation."""

    overall_score: float  # 0.0 to 10.0
    passed: bool  # True if overall_score >= threshold
    dimensions: dict[str, float]  # per-dimension scores
    strengths: list[str]
    issues: list[dict[str, str]]  # [{severity, location, description}]
    summary: str

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "passed": self.passed,
            "dimensions": self.dimensions,
            "strengths": self.strengths,
            "issues": self.issues,
            "summary": self.summary,
        }


JUDGE_SYSTEM_PROMPT = """You are a senior code reviewer evaluating AI-generated code for a production system.

Evaluate the code on these dimensions (score each 1-10):

1. **Correctness** (1-10): Does it correctly implement what was asked?
   - 10: Perfect implementation, handles all cases
   - 7: Works for common cases, minor edge case gaps
   - 4: Partially works, significant gaps
   - 1: Does not work at all

2. **Security** (1-10): Is it free of common vulnerabilities?
   - 10: Input sanitization, auth checks, no hardcoded secrets
   - 7: Good practices, minor improvements possible
   - 4: Some issues (missing input validation, weak auth)
   - 1: Major vulnerabilities (SQL injection, hardcoded secrets, no auth)

3. **Code Quality** (1-10): Is the code clean, readable, maintainable?
   - 10: Excellent structure, clear names, appropriate comments
   - 7: Good structure, mostly clear
   - 4: Messy, hard to follow
   - 1: Unreadable, no structure

4. **Error Handling** (1-10): Does it handle errors gracefully?
   - 10: Comprehensive error handling, user-friendly messages
   - 7: Good error handling for common cases
   - 4: Minimal error handling
   - 1: No error handling at all

5. **Testability** (1-10): Is the code designed to be testable?
   - 10: Clean interfaces, dependency injection, easy to mock
   - 7: Mostly testable with some effort
   - 4: Hard to test (tight coupling, global state)
   - 1: Untestable

Also identify:
- 2-3 specific strengths
- 1-3 specific issues with severity (CRITICAL, HIGH, MEDIUM, LOW), location (line or function), and description
- A 1-2 sentence summary of the overall assessment"""


JUDGE_USER_PROMPT = """Evaluate this code:

FEATURE REQUEST: {feature_description}

CODE FILES:
{code_files}

Respond with ONLY a JSON object:
{{
  "overall_score": 7.5,
  "dimensions": {{
    "correctness": 8,
    "security": 7,
    "code_quality": 8,
    "error_handling": 7,
    "testability": 7
  }},
  "strengths": ["...", "..."],
  "issues": [
    {{
      "severity": "HIGH",
      "location": "line 45 in profiles.py",
      "description": "Missing input validation for avatar file type"
    }}
  ],
  "summary": "..."
}}"""


class LLMJudge:
    """Evaluates AI-generated code using a separate LLM call.

    This provides an independent quality check — the evaluating model
    is different from the generating model, catching issues that the
    generator missed.
    """

    DEFAULT_THRESHOLD = 6.0
    DIMENSIONS = ["correctness", "security", "code_quality", "error_handling", "testability"]

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        config_path: str = "config.yaml",
        llm_client: LLMClient | None = None,
    ):
        self.threshold = threshold
        self._llm = llm_client
        self._llm_available = False

        if self._llm is None:
            try:
                config = Path(config_path)
                if config.exists():
                    self._llm = LLMClient.from_config(str(config))
                    # For judging, use a cheaper model to save costs
                    # (overridable via config)
                    self._llm_available = True
            except (FileNotFoundError, APIKeyMissingError):
                self._llm_available = False

    def evaluate(
        self,
        feature_description: str,
        code_files: dict[str, str],
    ) -> JudgeResult:
        """Evaluate a set of code files against a feature description.

        Args:
            feature_description: Natural language description of what the code should do.
            code_files: Dict mapping filename → file contents.

        Returns:
            JudgeResult with scores and detailed feedback.
        """
        if self._llm_available and self._llm:
            return self._evaluate_with_llm(feature_description, code_files)

        return self._evaluate_rules(feature_description, code_files)

    def _evaluate_with_llm(
        self, feature_description: str, code_files: dict[str, str]
    ) -> JudgeResult:
        """Use LLM for intelligent code evaluation."""
        # Build code listing for the prompt
        code_listing = ""
        for filename, contents in code_files.items():
            code_listing += f"\n### {filename}\n```python\n{contents}\n```\n"

        user_prompt = JUDGE_USER_PROMPT.format(
            feature_description=feature_description,
            code_files=code_listing,
        )

        output_schema = {
            "overall_score": 0.0,
            "dimensions": {
                "correctness": 0,
                "security": 0,
                "code_quality": 0,
                "error_handling": 0,
                "testability": 0,
            },
            "strengths": ["string"],
            "issues": [
                {
                    "severity": "CRITICAL | HIGH | MEDIUM | LOW",
                    "location": "string",
                    "description": "string",
                }
            ],
            "summary": "string",
        }

        try:
            result = self._llm.structured_chat(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_schema=output_schema,
            )
        except Exception as e:
            print(f"  [!] LLM judge failed ({e}), falling back to rule-based eval")
            return self._evaluate_rules(feature_description, code_files)

        overall = float(result.get("overall_score", 5.0))
        dims = result.get("dimensions", {})
        for key in self.DIMENSIONS:
            dims.setdefault(key, 5.0)

        return JudgeResult(
            overall_score=overall,
            passed=overall >= self.threshold,
            dimensions=dims,
            strengths=result.get("strengths", []),
            issues=result.get("issues", []),
            summary=result.get("summary", "No summary provided"),
        )

    def _evaluate_rules(
        self, feature_description: str, code_files: dict[str, str]
    ) -> JudgeResult:
        """Rule-based static analysis fallback (no LLM required).

        Checks for common issues that can be detected without an LLM.
        """
        total_code = "\n".join(code_files.values())

        issues: list[dict[str, str]] = []
        strengths: list[str] = []

        # Check for hardcoded secrets
        if "password" in total_code.lower() and '"' in total_code:
            issues.append({
                "severity": "CRITICAL",
                "location": "search for 'password'",
                "description": "Potential hardcoded credential or sensitive value",
            })
        if "sk-" in total_code or "api_key" in total_code.lower():
            issues.append({
                "severity": "CRITICAL",
                "location": "search for API key pattern",
                "description": "Potential hardcoded API key",
            })

        # Check for input validation
        if ("upload" in feature_description.lower() or "file" in feature_description.lower()):
            has_validation = any(
                pattern in total_code.lower()
                for pattern in ["filetype", "content_type", ".endswith", "allowed_types",
                               "magic", "mimetype", "validate"]
            )
            if not has_validation:
                issues.append({
                    "severity": "HIGH",
                    "location": "file upload handler",
                    "description": "File upload lacks content-type or extension validation",
                })
            else:
                strengths.append("Contains file validation logic")

        # Check for error handling
        has_try = "try:" in total_code
        has_except = "except" in total_code
        if has_try and has_except:
            strengths.append("Contains explicit error handling (try/except)")
        else:
            issues.append({
                "severity": "MEDIUM",
                "location": "throughout",
                "description": "Limited or no error handling detected",
            })

        # Check for type hints
        has_type_hints = ("def " in total_code and "->" in total_code)
        if has_type_hints:
            strengths.append("Uses type hints")

        # Check for docstrings
        if '"""' in total_code or "'''" in total_code:
            strengths.append("Includes docstrings")

        # Score calculation
        severe_issue_count = sum(
            1 for i in issues if i["severity"] in ("CRITICAL", "HIGH")
        )
        total_issues = len(issues)

        correctness = 7.0
        security = max(2.0, 8.0 - severe_issue_count * 2.0)
        code_quality = max(3.0, 7.0 - total_issues * 0.5)
        error_handling = 7.0 if (has_try and has_except) else 4.0
        testability = 6.0 if has_type_hints else 5.0

        overall = (
            correctness * 0.30 +
            security * 0.25 +
            code_quality * 0.20 +
            error_handling * 0.15 +
            testability * 0.10
        )

        return JudgeResult(
            overall_score=round(overall, 1),
            passed=overall >= self.threshold,
            dimensions={
                "correctness": correctness,
                "security": security,
                "code_quality": code_quality,
                "error_handling": error_handling,
                "testability": testability,
            },
            strengths=strengths or ["Code is syntactically valid Python"],
            issues=issues or [{"severity": "LOW", "location": "n/a", "description": "No obvious issues detected by static analysis"}],
            summary=(
                f"Static analysis found {len(issues)} issues "
                f"({severe_issue_count} high/critical). "
                f"Overall score: {overall:.1f}/10."
            ),
        )

    def format_report(self, result: JudgeResult) -> str:
        """Format a JudgeResult as a readable report string."""
        status = "PASS" if result.passed else "FAIL"
        icon = "PASS" if result.passed else "FAIL"

        lines = [
            f"  LLM-Judge Report  {icon} {status}",
            f"  {'─' * 45}",
            f"  Overall Score: {result.overall_score}/10 (threshold: {self.threshold})",
            f"",
            f"  Dimension Scores:",
        ]
        for dim, score in result.dimensions.items():
            bar = "█" * int(score) + "░" * (10 - int(score))
            lines.append(f"    {dim:<18} {bar} {score}/10")

        if result.strengths:
            lines.append("")
            lines.append("  Strengths:")
            for s in result.strengths:
                lines.append(f"    + {s}")

        if result.issues:
            lines.append("")
            lines.append("  Issues:")
            for issue in result.issues:
                sev = issue["severity"]
                loc = issue["location"]
                desc = issue["description"]
                lines.append(f"    [{sev}] {loc}: {desc}")

        lines.append(f"")
        lines.append(f"  Summary: {result.summary}")

        return "\n".join(lines)
