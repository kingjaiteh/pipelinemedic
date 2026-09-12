"""Validation selection and the pull request tool (dry-run, and push against a local bare repo)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import httpx
import pytest

from medic.config import GitHubSettings
from medic.graph.state import (
    ApplyResult,
    Evidence,
    FailingNode,
    FixAttempt,
    Hypothesis,
    Incident,
    ProposedFix,
    PullRequest,
    Validation,
)
from medic.tools.dbt_artifacts import Manifest
from medic.tools.github import PullRequestError, pr_body, pr_title, push_and_open, write_dry_run
from medic.tools.validate import selection_for

PKG = "marketing_attribution"


def incident(*nodes: tuple[str, str], args: list[str] | None = None) -> Incident:
    return Incident(
        source="chaos",
        run_id="chaos-x",
        dbt_args=args or ["build"],
        artifact="a",
        manifest="m",
        sandbox_dir="s",
        duckdb="d",
        repo="r",
        failing_nodes=[
            FailingNode(unique_id=uid, name=name, resource_type=uid.split(".")[0], status="error")
            for uid, name in nodes
        ],
    )


def test_selection_covers_failing_models_tested_models_and_edited_files(manifest_path: Path):
    manifest = Manifest.load(manifest_path)
    inc = incident((f"model.{PKG}.dim_channels", "dim_channels"))
    assert selection_for(inc, manifest, []) == ["build", "--select", "dim_channels+"]
    # a failing test resolves to the model it tests
    inc = incident(
        (f"test.{PKG}.not_null_stg_touchpoints_user_id.abc123", "not_null_stg_touchpoints_user_id")
    )
    assert selection_for(inc, manifest, []) == ["build", "--select", "stg_touchpoints+"]
    # an edited model file is added; a yml edit adds nothing
    sel = selection_for(
        inc,
        manifest,
        [
            "dbt_marketing/models/marts/dim_channels.sql",
            "dbt_marketing/models/staging/_sources.yml",
        ],
    )
    assert sel == ["build", "--select", "stg_touchpoints+ dim_channels+"]


def test_selection_for_freshness_and_for_nothing(manifest_path: Path):
    manifest = Manifest.load(manifest_path)
    inc = incident((f"source.{PKG}.raw.touchpoints", "touchpoints"), args=["source", "freshness"])
    assert selection_for(inc, manifest, []) == ["source", "freshness"]
    inc = incident((f"source.{PKG}.raw.touchpoints", "touchpoints"))
    assert selection_for(inc, manifest, []) == ["build"]


ATTEMPT = FixAttempt(
    iteration=1,
    proposal=ProposedFix(
        kind="code_patch",
        explanation="Revert c.touch_cnt to c.touch_count. The CTE defines touch_count.",
    ),
    apply=ApplyResult(
        ok=True, diff="--- a/x\n+++ b/x\n-c.touch_cnt\n+c.touch_count\n", edited_paths=["x"]
    ),
    validation=Validation(
        passed=True, dbt_args=["build", "--select", "dim_channels+"], status_counts={"success": 1}
    ),
)
TOP = Hypothesis(
    rank=1,
    root_cause="commit ab4cfdb renamed touch_count to touch_cnt",
    category="code_error",
    evidence_ids=["E1", "E9"],
    confidence=0.95,
    fix_kind="code_patch",
    recommended_action="revert",
)
EV = [
    Evidence(
        id="E1",
        claim="the diff shows the rename",
        tool="get_recent_changes",
        tool_call_id="T4",
        excerpt="-    c.touch_count,",
    )
]


def test_title_and_body_carry_the_evidence_diff_and_validation():
    inc = incident((f"model.{PKG}.dim_channels", "dim_channels"))
    title = pr_title(inc, ATTEMPT.proposal.explanation)
    assert title == "medic: fix dim_channels: Revert c.touch_cnt to c.touch_count"
    body = pr_body(inc, TOP, EV, ATTEMPT, ATTEMPT.validation, "fine by me")
    assert "## Root cause" in body and "commit ab4cfdb" in body
    assert "- the diff shows the rename" in body and "`-    c.touch_count,`" in body
    assert "```diff\n--- a/x" in body
    assert "dbt build --select dim_channels+ exit 0 (success=1)" in body
    assert "Approved by a human reviewer. fine by me" in body
    assert "E9" not in body  # a citation with no surviving evidence is simply absent


def test_dry_run_writes_markdown(tmp_path: Path):
    pr = PullRequest(
        mode="dry-run", title="medic: fix x", branch="medic/x", body="## Incident\n\nbody"
    )
    path = write_dry_run(tmp_path / "incidents" / "x.pr.md", pr)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# medic: fix x\n\nbranch: `medic/x`\n\n## Incident")


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True, env=env
    ).stdout.strip()


@pytest.fixture
def worktree_with_origin(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-q", str(origin))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "dbt_marketing").mkdir()
    (repo / "dbt_marketing" / "m.sql").write_text("select 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "checkout", "-q", "-b", "medic/x")
    (repo / "dbt_marketing" / "m.sql").write_text("select 2\n", encoding="utf-8")
    return repo, origin


def test_push_commits_pushes_and_posts_the_pull_request(worktree_with_origin, monkeypatch):
    repo, _origin = worktree_with_origin
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@example.com")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["json"] = request.read()
        return httpx.Response(201, json={"html_url": "https://github.com/o/r/pull/7", "number": 7})

    settings = GitHubSettings(token="ghp_test", repo="o/r", base_branch="main", pr_mode="push")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    pr = push_and_open(repo, "medic/x", "medic: fix m", "body text", settings, client=client)

    assert pr.mode == "push" and pr.url == "https://github.com/o/r/pull/7"
    assert seen["url"] == "https://api.github.com/repos/o/r/pulls"
    assert seen["auth"] == "Bearer ghp_test"
    assert b'"head":"medic/x"' in seen["json"].replace(b" ", b"") and b'"base":"main"' in seen[
        "json"
    ].replace(b" ", b"")
    # the branch on the origin carries the fix commit
    assert _git(repo, "log", "-1", "--format=%s", "origin/medic/x") == "medic: fix m"
    assert _git(repo, "rev-parse", "--short", "HEAD") == pr.commit
    assert _git(repo, "status", "--porcelain") == ""


def test_push_without_token_or_with_api_error_raises(worktree_with_origin):
    repo, _ = worktree_with_origin
    with pytest.raises(PullRequestError, match="GITHUB_TOKEN"):
        push_and_open(repo, "medic/x", "t", "b", GitHubSettings(token=None, repo="o/r"))
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(422, text="nope")))
    with pytest.raises(PullRequestError, match="422"):
        push_and_open(
            repo, "medic/x", "t", "b", GitHubSettings(token="x", repo="o/r"), client=client
        )
