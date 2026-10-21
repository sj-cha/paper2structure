from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CheckResult(BaseModel):
    name: str
    status: Literal["pass", "warn", "fail"]
    detail: str = ""


class ValidationReport(BaseModel):
    """Deterministic geometric/chemical checks of one structure. Experimental agreement is not
    assessed here: no observable is simulated."""
    structure_id: str
    passed: bool
    checks: list[CheckResult]
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    observables_evaluated: list[str] = Field(default_factory=list)

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "fail"]

    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "warn"]

    def with_check(self, check: CheckResult) -> ValidationReport:
        """A copy with `check` added (a failing check fails the report)."""
        return self.model_copy(update={"checks": [*self.checks, check],
                                       "passed": self.passed and check.status != "fail"})

    def summary(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        bad = [f"{c.name}: {c.detail}" for c in self.checks if c.status != "pass"]
        return head + (" - " + "; ".join(bad) if bad else "")
