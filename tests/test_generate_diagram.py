from __future__ import annotations

import os
import subprocess
import sys


def test_generate_diagram_runs_without_error(tmp_path, monkeypatch):
    """generate_diagram.py should create the architecture PNG without raising."""
    monkeypatch.chdir(tmp_path)
    result = subprocess.run(
        [sys.executable, str(
            os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_diagram.py")
        )],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert os.path.exists(tmp_path / "screenshots" / "architecture.png")
