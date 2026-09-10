"""All paths and settings resolve here. Nothing else reads the environment.

The target pipeline is the marketing-attribution repo one directory up. Every
path below defaults to that layout and can be overridden from .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class PipelineTarget:
    """Where the pipeline under investigation lives."""

    repo: Path = field(
        default_factory=lambda: _env_path(
            "PIPELINE_REPO", PROJECT_ROOT.parent / "marketing-attribution"
        )
    )

    @property
    def dbt_dir(self) -> Path:
        return self.repo / "dbt_marketing"

    @property
    def manifest(self) -> Path:
        return self.dbt_dir / "target" / "manifest.json"

    @property
    def run_results(self) -> Path:
        return self.dbt_dir / "target" / "run_results.json"

    @property
    def duckdb(self) -> Path:
        return _env_path("PIPELINE_DUCKDB", self.repo / "data" / "marketing.duckdb")

    @property
    def dbt_exe(self) -> Path:
        """dbt from the pipeline's own venv, so sandbox runs use its exact version."""
        return self.repo / ".venv" / "Scripts" / "dbt.exe"


@dataclass(frozen=True)
class LLMSettings:
    provider: str = field(default_factory=lambda: os.environ.get("MEDIC_LLM_PROVIDER", "gemini"))
    model: str = field(default_factory=lambda: os.environ.get("MEDIC_MODEL", "gemini-3.8-flash"))
    # Seconds between model calls. The free Gemini tier allows 10 requests a
    # minute; spacing calls costs less than the retries a burst provokes.
    min_interval_s: float = field(
        default_factory=lambda: float(os.environ.get("MEDIC_LLM_MIN_INTERVAL_S", "6.5"))
    )


@dataclass(frozen=True)
class Budget:
    """Hard per-incident limits. Exceeding any of them routes to escalate_report."""

    max_tool_calls: int = 25
    max_fix_iterations: int = 3
    max_tokens: int = 200_000
    sql_row_limit: int = 200
    sql_timeout_s: int = 10


TARGET = PipelineTarget()
LLM = LLMSettings()
BUDGET = Budget()
# Sandboxes hold a full warehouse copy each (about 40 MB), so this can be moved
# to a bigger drive with MEDIC_SANDBOX_ROOT.
SANDBOX_ROOT = _env_path("MEDIC_SANDBOX_ROOT", PROJECT_ROOT / "sandbox")
CHECKPOINT_DB = PROJECT_ROOT / "checkpoints.sqlite"
