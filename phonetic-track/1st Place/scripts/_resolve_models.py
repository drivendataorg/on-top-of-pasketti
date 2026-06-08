#!/usr/bin/env python3
"""Resolve a ``models.txt``-style file into one absolute model directory per
line. Skips comment lines (``#``) and blank lines.

A line ``foo.bar.eval`` is resolved to::

    <repo>/working/online/<RUN_VERSION>/<foo.bar.eval>/

falling back to ``working/offline/<RUN_VERSION>/<...>/0/`` if the online
directory is absent. Absolute paths are passed through unchanged.

Usage:
    python scripts/_resolve_models.py path/to/models.txt
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def _read_run_version(repo: Path) -> str:
  cfg = repo / 'src' / 'config.py'
  if not cfg.exists():
    return '17'
  for line in cfg.read_text().splitlines():
    m = re.match(r"\s*RUN_VERSION\s*=\s*['\"]([^'\"]+)['\"]", line)
    if m:
      return m.group(1)
  return '17'


def _resolve(name: str, repo: Path, run_v: str) -> str:
  if name.startswith('/') or '/' in name:
    return str(Path(name).resolve())
  for sub in (f'working/online/{run_v}/{name}',
              f'working/offline/{run_v}/{name}/0'):
    p = repo / sub
    if p.is_dir():
      return str(p.resolve())
  raise FileNotFoundError(f'Cannot resolve model: {name} (looked under working/{{online,offline}}/{run_v}/)')


def main() -> None:
  if len(sys.argv) != 2:
    print('usage: _resolve_models.py <models_file>', file=sys.stderr)
    sys.exit(2)
  models_file = Path(sys.argv[1])
  repo = Path(__file__).resolve().parent.parent
  run_v = _read_run_version(repo)
  for line in models_file.read_text().splitlines():
    s = line.strip()
    if not s or s.startswith('#'):
      continue
    print(_resolve(s, repo, run_v))


if __name__ == '__main__':
  main()
