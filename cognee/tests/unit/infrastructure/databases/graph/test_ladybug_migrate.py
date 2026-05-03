"""Tests for Ladybug migration helpers."""

import os
import subprocess
import tempfile

from cognee.infrastructure.databases.graph.ladybug.ladybug_migrate import run_migration_step


def test_run_migration_step_isolates_legacy_import_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.getcwd())
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(
        "cognee.infrastructure.databases.graph.ladybug.ladybug_migrate.subprocess.run",
        fake_run,
    )

    run_migration_step("/tmp/legacy/bin/python", "kuzu", "relative_db", "MATCH (n) RETURN n")

    args, kwargs = calls[0]
    assert args[:2] == ["/tmp/legacy/bin/python", "-c"]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["cwd"] == tempfile.gettempdir()
    assert "PYTHONPATH" not in kwargs["env"]
    assert f"Database({os.path.abspath('relative_db')!r})" in args[2]
    assert "conn.execute('MATCH (n) RETURN n')" in args[2]
