from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys
import pytest
from scripts import run_pipeline


def test_pipeline_stops_before_summary_when_a_stage_fails(monkeypatch):
    monkeypatch.setattr(sys,'argv',['run_pipeline.py'])
    calls=[]
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(run_pipeline.subprocess,'run',fail)
    monkeypatch.setattr(run_pipeline,'summarize',lambda *a: pytest.fail('No summary after failure'))
    with pytest.raises(subprocess.CalledProcessError):run_pipeline.main()
    assert len(calls)==1 and calls[0][1].endswith('generate_signals.py')


def test_dry_run_never_runs_commands_or_writes_summary(monkeypatch):
    monkeypatch.setattr(sys,'argv',['run_pipeline.py','--dry-run'])
    monkeypatch.setattr(run_pipeline.subprocess,'run',lambda *a,**k:pytest.fail('dry run executes nothing'))
    monkeypatch.setattr(run_pipeline,'summarize',lambda *a:pytest.fail('dry run writes nothing'))
    run_pipeline.main()
