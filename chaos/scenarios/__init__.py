"""Scenario registry. scenarios.yaml holds the facts; each module holds the mutation."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from chaos.mutation import MutationContext

SCENARIOS_YAML = Path(__file__).resolve().parent.parent / "scenarios.yaml"


@dataclass(frozen=True)
class Scenario:
    number: int
    key: str
    title: str
    mutation: str
    dbt_args: tuple[str, ...]
    category: str
    fix_kind: str
    expected_failing: tuple[str, ...]
    error_contains: str
    apply: Callable[[MutationContext], None]

    @property
    def sandbox_name(self) -> str:
        return self.key


def load_scenarios(path: Path = SCENARIOS_YAML) -> dict[int, Scenario]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenarios: dict[int, Scenario] = {}
    for entry in raw["scenarios"]:
        module = importlib.import_module(f"chaos.scenarios.{entry['key']}")
        number = int(entry["number"])
        scenarios[number] = Scenario(
            number=number,
            key=entry["key"],
            title=entry["title"],
            mutation=entry["mutation"],
            dbt_args=tuple(entry["dbt_args"]),
            category=entry["category"],
            fix_kind=entry["fix_kind"],
            expected_failing=tuple(entry["expected_failing"]),
            error_contains=entry["error_contains"],
            apply=module.apply,
        )
    return dict(sorted(scenarios.items()))


def get_scenario(number: int, path: Path = SCENARIOS_YAML) -> Scenario:
    scenarios = load_scenarios(path)
    if number not in scenarios:
        raise KeyError(f"no scenario {number}; have {sorted(scenarios)}")
    return scenarios[number]
