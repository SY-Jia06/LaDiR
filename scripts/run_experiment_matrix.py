#!/usr/bin/env python3
"""Expand and optionally execute comparison/ablation command matrices.

Dry-run is the default.  Every executed job is appended to a JSONL manifest so
results remain tied to a concrete command and git revision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Iterator

from omegaconf import OmegaConf


@dataclass(frozen=True)
class Job:
    group: str
    name: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix", default="configs/reasoner_experiments.yaml"
    )
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--manifest", default="results/experiment_manifest.jsonl"
    )
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def render(template: str, values: dict[str, Any]) -> str:
    return template.format(**values)


def expand_grid(group_name: str, group: Any) -> Iterator[Job]:
    command = shlex.split(str(group.command))
    fixed_args = [str(value) for value in group.get("fixed_args", [])]
    dimensions = dict(group.get("dimensions", {}))
    aliases = list(dimensions)
    value_lists = [list(dimensions[alias].get("values", [])) for alias in aliases]
    for combination in itertools.product(*value_lists):
        values = dict(zip(aliases, combination))
        name = render(str(group.name_template), values)
        arguments = list(fixed_args)
        for alias, value in values.items():
            spec = dimensions[alias]
            for template in spec.get("arguments", []):
                arguments.append(render(str(template), {**values, "value": value}))
        for template in group.get("derived_args", []):
            arguments.append(render(str(template), {**values, "name": name}))
        yield Job(group_name, name, command + arguments)


def expand_group(group_name: str, group: Any) -> Iterator[Job]:
    for explicit in group.get("jobs", []):
        yield Job(
            group_name,
            str(explicit.name),
            shlex.split(str(explicit.command)),
        )
    if group.get("command"):
        yield from expand_grid(group_name, group)


def load_jobs(path: str, selected: set[str]) -> list[Job]:
    cfg = OmegaConf.load(path)
    jobs: list[Job] = []
    for group_name, group in cfg.groups.items():
        if selected and group_name not in selected:
            continue
        jobs.extend(expand_group(group_name, group))
    names = [job.name for job in jobs]
    if len(names) != len(set(names)):
        raise ValueError("experiment names must be globally unique")
    return jobs


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    jobs = load_jobs(args.matrix, set(args.group))
    if not jobs:
        raise ValueError("selected experiment matrix contains no jobs")
    for job in jobs:
        print(f"[{job.group}] {job.name}")
        print("  " + shlex.join(job.command))
    if not args.execute:
        print(f"dry-run: {len(jobs)} jobs; pass --execute to run them")
        return

    manifest = Path(args.manifest)
    revision = git_revision()
    for job in jobs:
        start = utc_now()
        completed = subprocess.run(job.command, check=False)
        record = {
            "group": job.group,
            "name": job.name,
            "command": job.command,
            "command_shell": shlex.join(job.command),
            "git_revision": revision,
            "started_at": start,
            "finished_at": utc_now(),
            "returncode": completed.returncode,
            "status": "completed" if completed.returncode == 0 else "failed",
        }
        append_manifest(manifest, record)
        if completed.returncode and not args.keep_going:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
