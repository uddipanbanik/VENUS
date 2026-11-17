"""Restarted grid runs can omit one or both endpoint projections."""
import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "tools/plot_einstein_vlasov_outputs.py"
SPEC = importlib.util.spec_from_file_location("optional_grid_phase_plots", SOURCE)
PLOTTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTTER)


@pytest.mark.parametrize("saved", [(), ("initial",), ("final",)])
def test_incomplete_endpoint_pair_is_optional(tmp_path, saved, capsys):
    for name in saved:
        (tmp_path / f"phase_space_{name}.csv").write_text("unused optional projection")
    PLOTTER.plot_phase(tmp_path, tmp_path)
    assert "both projections are required" in capsys.readouterr().out
    assert not (tmp_path / "phase_space_initial_final.png").exists()


def test_present_invalid_projections_still_report_errors(tmp_path, monkeypatch):
    for name in ("initial", "final"):
        (tmp_path / f"phase_space_{name}.csv").write_text("invalid projection")

    def invalid_projection(path):
        raise ValueError("invalid phase-space data")

    monkeypatch.setattr(PLOTTER, "load_phase", invalid_projection)
    with pytest.raises(ValueError, match="invalid phase-space data"):
        PLOTTER.plot_phase(tmp_path, tmp_path)
