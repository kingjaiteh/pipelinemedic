"""open_pull_request: after approval only. Dry-run by default, push on request.

Dry-run writes the pull request as markdown next to the incident records.
Push commits the worktree's dbt project on its `medic/<name>` branch, pushes
that branch, and opens the pull request against the base branch through the
GitHub REST API with the fine-grained token. Nothing here touches main.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import httpx

from medic.config import GITHUB, GitHubSettings
from medic.graph.state import (
    Evidence,
    FixAttempt,
    Hypothesis,
    Incident,
    PullRequest,
    Validation,
)

API = "https://api.github.com"


class PullRequestError(RuntimeError):
    pass


def pr_title(incident: Incident, explanation: str) -> str:
    nodes = ", ".join(n.name for n in incident.failing_nodes[:2]) or incident.run_id
    first = explanation.strip().split(". ")[0].rstrip(".")
    title = f"medic: fix {nodes}: {first}"
    return title if len(title) <= 100 else title[:97] + "..."


def pr_body(
    incident: Incident,
    top: Hypothesis | None,
    evidence: list[Evidence],
    attempt: FixAttempt,
    validation: Validation | None,
    reviewer_note: str,
) -> str:
    by_id = {e.id: e for e in evidence}
    lines = ["## Incident", ""]
    lines.append(f"`dbt {' '.join(incident.dbt_args)}` failed ({incident.run_id}).")
    for n in incident.failing_nodes:
        first = (n.message or "").strip().splitlines()
        lines.append(f"- `{n.name}` {n.status}: {first[0][:300] if first else ''}".rstrip(": "))
    lines += ["", "## Root cause", ""]
    if top:
        lines.append(f"{top.root_cause} (confidence {top.confidence:.2f}, {top.category})")
        lines += ["", "Evidence, quoted from tool results:", ""]
        for eid in top.evidence_ids:
            e = by_id.get(eid)
            if e:
                lines.append(f"- {e.claim}")
                lines.append(f"  `{e.excerpt[:200]}`")
    lines += ["", "## Fix", "", attempt.proposal.explanation, ""]
    if attempt.apply and attempt.apply.diff:
        lines += ["```diff", attempt.apply.diff.rstrip(), "```", ""]
    lines += ["## Validation", ""]
    if validation:
        lines.append(validation.summary())
    else:
        lines.append("not validated")
    lines += ["", "## Review", ""]
    lines.append(f"Approved by a human reviewer. {reviewer_note}".strip())
    lines.append("")
    lines.append("Opened by PipelineMedic after human approval; nothing is merged automatically.")
    return "\n".join(lines)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise PullRequestError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def write_dry_run(path: Path, pr: PullRequest) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {pr.title}\n\nbranch: `{pr.branch}`\n\n{pr.body}\n", encoding="utf-8")
    return path


def push_and_open(
    repo: Path,
    branch: str,
    title: str,
    body: str,
    settings: GitHubSettings = GITHUB,
    client: httpx.Client | None = None,
) -> PullRequest:
    if not settings.token:
        raise PullRequestError("GITHUB_TOKEN is not set; cannot push")
    _git(repo, "add", "-A", "--", "dbt_marketing")
    if _git(repo, "status", "--porcelain", "--", "dbt_marketing"):
        _git(
            repo,
            "commit",
            "-m",
            title,
            "-m",
            "Proposed by PipelineMedic and approved by a reviewer.",
        )
    commit = _git(repo, "rev-parse", "--short", "HEAD")
    _git(repo, "push", "-u", "origin", f"{branch}:{branch}")
    own = client is None
    client = client or httpx.Client(timeout=30)
    try:
        r = client.post(
            f"{API}/repos/{settings.repo}/pulls",
            headers={
                "Authorization": f"Bearer {settings.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"title": title, "head": branch, "base": settings.base_branch, "body": body},
        )
    finally:
        if own:
            client.close()
    if r.status_code >= 300:
        raise PullRequestError(f"GitHub returned {r.status_code}: {r.text[:500]}")
    return PullRequest(
        mode="push",
        title=title,
        branch=branch,
        body=body,
        url=r.json().get("html_url"),
        commit=commit,
    )
