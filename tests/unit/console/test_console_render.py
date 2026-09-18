"""Console rendering over a real fake-system bundle (ch37 mandatory views, SS-04 fail-closed)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from console_bundle_fixture import TRUTH_MARKER, make_bundle

from conrad.console import IntegrityFailure, build_console, open_bundle
from conrad.runtime.event_log import read_events


@pytest.fixture(scope="module")
def page(bundle_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> str:
    out = build_console(bundle_dir, tmp_path_factory.mktemp("html") / "console.html")
    return out.read_text(encoding="utf-8")


def _section(page: str, sid: str) -> str:
    m = re.search(rf'<section id="{sid}".*?</section>', page, re.S)
    assert m, sid
    return m.group(0)


def test_belief_status_timeline_shows_property_level_status(page: str) -> None:
    sec = _section(page, "belief-status-timeline")
    assert "2T technical" in sec and "claim <b>severity</b>" in sec
    # rev 0 left the far side unseen (UNKNOWN), rev 1 observed it
    assert 'badge ks-UNKNOWN">UNKNOWN' in sec and 'badge ks-OBSERVED">OBSERVED' in sec
    assert "rev 0" in sec and "rev 1" in sec and "data-seq=" in sec


def test_uncertainty_channels_are_all_four_and_charted(page: str) -> None:
    sec = _section(page, "uncertainty-channels")
    for label in ("U_A aleatoric", "U_E epistemic", "U_C contradiction", "U_O observational"):
        assert label in sec
    assert sec.count('<polyline class="series') >= 4 and "confidence" not in sec.lower().replace(
        "no single confidence scalar", ""
    )


def test_provenance_chain_runs_from_last_command_to_an_observation(page: str, bundle_dir: Path) -> None:
    events = list(read_events(bundle_dir / "events.jsonl"))
    last_cmd = [e for e in events if e.event_type.value == "COMMAND_SENT"][-1]
    obs_ids = {e.payload["observation_id"] for e in events if e.event_type.value == "OBSERVATION_RECEIVED"}
    chain = re.search(r'<div id="provenance-last-command" class="tree">(.*?)</div></section>', page, re.S)
    assert chain
    text = chain.group(1)
    assert last_cmd.payload["provenance_root"] in text.split("</div>")[0]  # the root line is the command
    for kind in ("COMMAND", "PLAN", "DECISION", "DIRECT_OBSERVATION"):
        assert kind in text
    raw = re.findall(r"RAW OBSERVATION ([0-9a-f-]{36})", text)
    assert raw and set(raw) <= obs_ids
    prov = json.loads(re.search(r'id="prov-data">(.*?)</script>', page, re.S).group(1))
    assert last_cmd.payload["provenance_root"] in prov["nodes"]


def test_command_outcomes_and_mcbr_rejections(page: str) -> None:
    cmd = _section(page, "command-outcomes")
    assert 'badge out-ACCEPTED">ACCEPTED' in cmd and "106 commands" in cmd
    assert (
        "[raw actuator payload not shown]" not in cmd and "thruster_commands" not in page.split("<script")[0]
    )
    mcbr = _section(page, "mcbr")
    assert "NOT_VISIBLE_FROM_LANE" in mcbr and "selected view" in mcbr
    assert "EXTEND_COVERAGE" in page or "REQUEST_INFORMATION" in mcbr


def test_default_page_contains_no_truth_value(page: str) -> None:
    """ch37 UI-08: without evaluation mode no truth-derived value appears anywhere in the HTML."""
    assert str(TRUTH_MARKER) not in page
    assert "TRUTH (evaluation only)" not in page and "EVALUATION MODE" not in page
    assert "c-truth" not in page.split("<style>")[0] + page.split("</style>")[1]
    truth = _section(page, "truth-evaluation-only")
    assert "evaluation-only truth record exists" in truth and "include_truth=True" in truth
    assert "hidden outside evaluation mode" in _section(page, "pose-track")


def test_evaluation_mode_labels_truth_and_confines_it(bundle_dir: Path, tmp_path: Path) -> None:
    html = build_console(bundle_dir, tmp_path / "eval.html", include_truth=True).read_text(encoding="utf-8")
    assert "<title>EVALUATION MODE" in html and 'id="evaluation-mode"' in html
    truth = _section(html, "truth-evaluation-only")
    assert 'class="truth"' in truth and "TRUTH (evaluation only)" in truth and str(TRUTH_MARKER) in truth
    assert str(TRUTH_MARKER) not in html.replace(truth, "")
    assert "TRUTH (evaluation only)" in _section(html, "pose-track")
    for sid in ("belief-status-timeline", "uncertainty-channels", "provenance-explorer"):
        assert "TRUTH" not in _section(html, sid)


def test_page_is_self_contained(page: str) -> None:
    assert not re.search(r"https?://", page)
    assert not re.search(r"""(?:src|href|action)\s*=\s*["'](?!#)""", page)
    assert "<link" not in page and "@import" not in page and "fetch(" not in page
    assert "localStorage" not in page and "XMLHttpRequest" not in page
    assert page.count("<script") == 3  # two inert JSON blocks + one inline script


def test_tampered_event_log_renders_integrity_page_only(bundle_copy: Path, tmp_path: Path) -> None:
    with (bundle_copy / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    html = build_console(bundle_copy, tmp_path / "bad.html").read_text(encoding="utf-8")
    assert 'id="integrity-error"' in html and "events.jsonl" in html and "corrupt" in html
    assert "belief-status-timeline" not in html and "<script" not in html


def test_missing_database_digest_fails_closed(bundle_copy: Path) -> None:
    (bundle_copy / "run.sqlite").unlink()
    loaded = open_bundle(bundle_copy)
    assert isinstance(loaded, IntegrityFailure)
    assert any("run.sqlite" in p and "missing" in p for p in loaded.problems)


def test_missing_manifest_fails_closed(bundle_copy: Path) -> None:
    (bundle_copy / "bundle_manifest.json").unlink()
    assert isinstance(open_bundle(bundle_copy), IntegrityFailure)


def test_optional_parts_absent_show_not_present(tmp_path: Path) -> None:
    run = make_bundle(tmp_path / "bare", seed=5, extras=False)
    html = build_console(run, tmp_path / "bare.html").read_text(encoding="utf-8")
    truth = _section(html, "truth-evaluation-only")
    assert "truth/: not present in this bundle" in truth and "exists" not in truth
    assert "reports/metrics.json: not present" in _section(html, "metrics")
    assert "no truth/ pose record present" in _section(html, "pose-track")
    assert "badge ks-" in _section(html, "belief-status-timeline")
    assert list(run.glob("*.sqlite-*")) == []  # the console never wrote WAL/SHM files into the run dir
