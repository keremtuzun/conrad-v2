"""Truth-removal falsification of the inference and decision planes (CC-10 follow-up).

For every bundle given on the command line:

1. copy it to a scratch directory and MOVE its truth and simulator-side entries out of the copy
   (``truth/``, ``reports/``, ``unity/``);
2. re-run the unchanged ``MissionRuntime`` from the captured observations only
   (``conrad.sim.mission.capture.replay_inference``) in a fresh Python process with an audit hook
   (``sys.addaudithook``) that records every ``open`` event, then list every file that process opened and every
   truth-side module it imported;
3. compare events, SQLite tables (belief revisions, evidence, commands...), ``mission/*`` and commands with the
   original bundle, byte for byte;
4. controls: (A) truth files corrupted in place -> output must NOT change; (B) one captured depth sample altered
   and (C) one captured payload observation dropped -> output MUST change; (D) the altered tape without resealing
   the manifest is refused by the digest check;
5. only after the audited phase: put the truth back next to the replayed outputs, re-run the truth-side evaluation
   and compare the headline numbers with the original ``reports/metrics.json``;
6. write ``artifacts/falsification/<bundle>/report.json`` (small JSON).

Bundles recorded before observation capture existed (no ``capture/``) cannot be replayed from observations; the
report says so and still records which files the existing CC-10 replay opens (``verify_bundle``).

Usage: python -m uv run python scripts/falsify_truth_removal.py <bundle> [<bundle> ...] [--scratch DIR]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TRUTH_ENTRIES = ("truth", "reports", "unity")
FORBIDDEN_MODULES = (
    "conrad.twins",
    "conrad.schemas.truth",
    "conrad.sim.kernel",
    "conrad.sim.unity",
    "conrad.sim.mission.world",
    "conrad.sim.mission.unity_world",
    "conrad.sim.mission.truth",
    "conrad.sim.mission.sensing",
    "conrad.sim.mission.structure",
    "conrad.sim.mission.run",
    "conrad.sim.mission.unity_run",
    "conrad.adapters.unity",
    "conrad.evaluation",
    "conrad.orchestration.evaluation",
)
CODE_SUFFIXES = (".py", ".pyc", ".pyd", ".dll", ".so", ".pth", ".typed", ".zip")


# ================================================================================================= child process
def _child(bundle: str, out: str, result: str, forbidden_roots: list[str]) -> int:
    """Audited playback. The hook is installed before anything from ``conrad`` is imported."""
    opened: list[str] = []

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and args and isinstance(args[0], str | bytes | os.PathLike):
            p = args[0]
            opened.append(os.fsdecode(p) if not isinstance(p, str) else p)

    sys.addaudithook(hook)
    from conrad.sim.mission.capture import replay_inference

    report: dict[str, Any]
    try:
        report = replay_inference(Path(bundle), Path(out))
    except Exception as exc:  # reported, never swallowed: the parent turns it into a finding
        report = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
    mods = sorted(m for m in sys.modules if m.startswith(FORBIDDEN_MODULES))
    Path(result).write_text(
        json.dumps({"playback": report, "opened": opened, "forbidden_modules_imported": mods}, default=str),
        encoding="utf-8",
    )
    return 0


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _under(p: str, root: Path) -> str | None:
    r = _norm(str(root))
    return os.path.relpath(p, r).replace("\\", "/") if p == r or p.startswith(r + os.sep) else None


def classify_opens(opened: list[str], bundle: Path, out: Path, forbidden_roots: list[Path]) -> dict[str, Any]:
    prefixes = {_norm(sys.prefix), _norm(sys.base_prefix), _norm(sys.exec_prefix)}
    seen: dict[str, None] = {}
    for raw in opened:
        seen.setdefault(_norm(raw), None)
    bundle_files: set[str] = set()
    output_files: set[str] = set()
    repo_other: set[str] = set()
    other: set[str] = set()
    code = 0
    truth_hits: list[str] = []
    for p in seen:
        for root in forbidden_roots:
            if _under(p, root) is not None:
                truth_hits.append(p)
        name = os.path.basename(p)
        if "truth" in name.lower() and not p.endswith(CODE_SUFFIXES) and p not in truth_hits:
            truth_hits.append(p)
        rel_b, rel_o = _under(p, bundle), _under(p, out)
        if rel_b is not None:
            bundle_files.add(rel_b)
            if rel_b.split("/", 1)[0] in TRUTH_ENTRIES and p not in truth_hits:
                truth_hits.append(p)
        elif rel_o is not None:
            output_files.add(rel_o)
        elif p.endswith(CODE_SUFFIXES) or any(p.startswith(x + os.sep) for x in prefixes):
            code += 1
        elif (rel_r := _under(p, REPO)) is not None:
            repo_other.add(rel_r)
        else:
            other.add(p)

    def summarize(paths: set[str]) -> dict[str, Any]:
        groups: dict[str, int] = {}
        for x in paths:
            head = x.split("/", 1)[0] if "/" in x else x
            if head in ("objects",):
                head = "objects/sha256/*"
            groups[head] = groups.get(head, 0) + 1
        return {"distinct": len(paths), "by_entry": dict(sorted(groups.items()))}

    return {
        "distinct_paths": len(seen),
        "python_code_and_libraries": code,
        "bundle": summarize(bundle_files),
        "bundle_files_except_objects": sorted(x for x in bundle_files if not x.startswith("objects/")),
        "playback_output": summarize(output_files),
        "repo_non_code": sorted(repo_other),
        "outside_repo_non_code": sorted(other),
        "truth_paths_opened": sorted(truth_hits),
    }


def run_child(bundle: Path, out: Path, forbidden_roots: list[Path], tag: str) -> dict[str, Any]:
    result = out.parent / f"{tag}.child.json"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(bundle), str(out), str(result)]
    cmd += [str(r) for r in forbidden_roots]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not result.exists():
        return {"playback": {"ran": False, "error": proc.stderr[-2000:]}, "opens": None, "forbidden": None}
    data = json.loads(result.read_text(encoding="utf-8"))
    return {
        "playback": data["playback"],
        "opens": classify_opens(data["opened"], bundle, out, forbidden_roots),
        "forbidden": data["forbidden_modules_imported"],
    }


# ================================================================================================= tape edits
def _rewrite_tape(bundle: Path, edit: Any, reseal: bool = True) -> dict[str, Any]:
    from conrad.sim.mission.capture import TAPE

    path = bundle / TAPE
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = [ln for ln in handle.read().splitlines() if ln.strip()]
    info = edit(lines)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(ln + "\n" for ln in lines))
    if reseal:
        manifest_path = bundle / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][TAPE] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return dict(info)


def _alter_depth(lines: list[str]) -> dict[str, Any]:
    idx = [i for i, ln in enumerate(lines) if '"n":"get_depth"' in ln and '"__m__"' in ln]
    if not idx:
        raise RuntimeError("tape holds no depth sample to alter")
    i = idx[len(idx) // 2]
    rec = json.loads(lines[i])
    before = rec["r"]["v"]["depth_m"]
    rec["r"]["v"]["depth_m"] = before + 0.5
    rec.pop("p", None)  # the exact (pickled) copy would otherwise be served instead of the edit
    lines[i] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    return {"tape_record": i, "field": "get_depth.depth_m", "before": before, "after": before + 0.5}


def _drop_payload(lines: list[str]) -> dict[str, Any]:
    for i, ln in enumerate(lines):
        if '"n":"get_payload_observations"' not in ln:
            continue
        rec = json.loads(ln)
        if isinstance(rec["r"], list) and rec["r"]:
            dropped = rec["r"].pop(0)
            rec.pop("p", None)
            lines[i] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
            v = dropped.get("v", {})
            return {
                "tape_record": i,
                "dropped_observation_id": v.get("observation_id"),
                "modality": v.get("modality"),
                "remaining_in_call": len(rec["r"]),
            }
    raise RuntimeError("tape holds no payload observation to drop")


def _corrupt_truth(bundle: Path) -> list[str]:
    changed = []
    for entry in TRUTH_ENTRIES:
        root = bundle / entry
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                p.write_text(
                    json.dumps({"corrupted_by": "falsify_truth_removal", "was": p.name}), encoding="utf-8"
                )
                changed.append(p.relative_to(bundle).as_posix())
    return changed


# ================================================================================================= helpers
def _legacy_replay_opens(bundle: Path) -> dict[str, Any]:
    """Which files the EXISTING CC-10 replay touches before re-execution (its digest check)."""
    from conrad.persistence.object_store import ObjectStore
    from conrad.persistence.replay_store import ReplayIntegrityError, verify_bundle

    opened: list[str] = []
    active = [True]

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if active[0] and event == "open" and args and isinstance(args[0], str | os.PathLike):
            opened.append(_norm(os.fspath(args[0])))

    sys.addaudithook(hook)
    try:
        verify_bundle(bundle, ObjectStore(bundle / "objects"))
        ok, problems = True, []
    except ReplayIntegrityError as exc:
        ok, problems = False, exc.problems
    finally:
        active[0] = False
    truth = sorted(
        {
            rel
            for p in opened
            if (rel := _under(p, bundle)) is not None and rel.split("/", 1)[0] in TRUTH_ENTRIES
        }
    )
    return {
        "verify_bundle_ok": ok,
        "problems": problems[:5],
        "truth_files_opened_by_verify_bundle": truth,
        "note": "replay_run / replay_unity_run then re-simulate the Twin world (MissionWorld / UnityMissionWorld)",
    }


def _outputs_equal(original: Path, replayed: Path) -> dict[str, Any]:
    from conrad.sim.mission.capture import compare_outputs

    if not (replayed / "conrad.sqlite").exists():
        return {"equal": False, "note": "no playback output"}
    return compare_outputs(original, replayed)


def _headline(original: Path, replayed: Path, stash: Path) -> dict[str, Any]:
    """Post-hoc, NOT audited: truth is put back next to the replayed outputs and the evaluation re-run."""
    from conrad.orchestration.evaluation import evaluate_run_dir

    orig_path = stash / "reports" / "metrics.json"
    if not orig_path.exists() or not (stash / "truth").exists():
        return {"available": False}
    shutil.copytree(stash / "truth", replayed / "truth", dirs_exist_ok=True)
    try:
        got = evaluate_run_dir(replayed)
    except (OSError, KeyError, ValueError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    want = json.loads(orig_path.read_text(encoding="utf-8"))
    keys = sorted(k for k in got if k in want)
    same = {
        k: json.dumps(got[k], sort_keys=True, default=str) == json.dumps(want[k], sort_keys=True, default=str)
        for k in keys
    }
    out: dict[str, Any] = {"available": True, "keys_compared": len(keys), "all_equal": all(same.values())}
    out["differing_keys"] = sorted(k for k, v in same.items() if not v)
    after_o, after_r = want.get("target_after") or {}, got.get("target_after") or {}
    err_o, err_r = want.get("target_error_after") or {}, got.get("target_error_after") or {}
    if after_o:
        out["crack_length_estimate_m"] = {
            "original": after_o.get("crack_length_m"),
            "replayed": after_r.get("crack_length_m"),
        }
        out["crack_length_abs_error_m"] = {
            "original": err_o.get("crack_length_m.abs_error"),
            "replayed": err_r.get("crack_length_m.abs_error"),
        }
        out["corrosion_depth_estimate_m"] = {
            "original": after_o.get("corrosion_depth_m"),
            "replayed": after_r.get("corrosion_depth_m"),
        }
    return out


def _stable_playback(p: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in p.items() if k != "absent_manifest_files"} | {
        "absent_manifest_files": len(p.get("absent_manifest_files", []))
    }


# ================================================================================================= per bundle
def falsify(bundle: Path, scratch: Path, out_root: Path) -> dict[str, Any]:
    from conrad.sim.mission.capture import TAPE, inference_fingerprint

    given = bundle.as_posix()
    bundle = bundle.resolve()
    name = f"{bundle.parent.name}__{bundle.name}"  # parent keeps same-named bundles (recorded vs fresh) apart
    work = scratch / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    report: dict[str, Any] = {
        "bundle": bundle.relative_to(REPO).as_posix() if bundle.is_relative_to(REPO) else given
    }
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    ri = manifest["replay_inputs"]
    report["provenance"] = {
        "scenario_id": ri.get("scenario_id"),
        "seed": ri.get("scenario_seed"),
        "backend": ri.get("backend", "python"),
        "config_digest": ri.get("config_digest"),
        "git_commit": ri.get("git_commit"),
        "manifest_files": len(manifest["files"]),
    }
    report["truth_layout"] = {
        e: sorted(k for k in manifest["files"] if k.split("/", 1)[0] == e) for e in TRUTH_ENTRIES
    }
    report["legacy_cc10_replay"] = _legacy_replay_opens(bundle)
    report["has_observation_capture"] = TAPE in manifest["files"]

    # --- 1. strip truth and replay from observations only
    stripped = work / "stripped" / name
    shutil.copytree(bundle, stripped)
    stash = work / "removed_truth"
    stash.mkdir()
    moved = []
    for entry in TRUTH_ENTRIES:
        if (stripped / entry).exists():
            shutil.move(str(stripped / entry), str(stash / entry))
            moved.append(entry)
    report["removed_entries"] = moved
    forbidden = [bundle / e for e in TRUTH_ENTRIES] + [stash]
    main = run_child(stripped, work / "play_stripped", forbidden, "stripped")
    report["truth_removed_replay"] = {
        "playback": _stable_playback(main["playback"]),
        "opened": main["opens"],
        "forbidden_modules_imported": main["forbidden"],
        "comparison_with_original": _outputs_equal(bundle, work / "play_stripped"),
    }
    if not main["playback"].get("ran"):
        report["verdict"] = "NOT_REPLAYABLE_FROM_OBSERVATIONS"
        return _write(report, out_root / name)
    base_fp = inference_fingerprint(work / "play_stripped")

    # --- A. truth corrupted in place: must not change anything
    corrupt = work / "corrupt" / name
    shutil.copytree(bundle, corrupt)
    changed = _corrupt_truth(corrupt)
    a = run_child(corrupt, work / "play_corrupt", [], "corrupt")
    ok_a = a["playback"].get("ran") and inference_fingerprint(work / "play_corrupt") == base_fp
    report["control_truth_corrupted"] = {
        "files_corrupted": changed,
        "playback": _stable_playback(a["playback"]),
        "truth_paths_opened": None if a["opens"] is None else a["opens"]["truth_paths_opened"],
        "output_identical_to_truth_removed_replay": bool(ok_a),
    }

    # --- B/C. one captured observation altered: output must change
    controls: dict[str, dict[str, Any]] = {}
    for tag, edit in (("depth_sample_altered", _alter_depth), ("payload_observation_dropped", _drop_payload)):
        alt = work / tag / name
        shutil.copytree(stripped, alt)
        try:
            what = _rewrite_tape(alt, edit)
        except RuntimeError as exc:
            controls[tag] = {"skipped": str(exc)}
            continue
        c = run_child(alt, work / f"play_{tag}", [], tag)
        cmp = _outputs_equal(bundle, work / f"play_{tag}")
        controls[tag] = {
            "edit": what,
            "tape_divergence": c["playback"].get("tape_divergence"),
            "command_mismatches": c["playback"].get("command_mismatches"),
            "events": cmp.get("events"),
            "sqlite_tables_changed": sorted(k for k, v in cmp.get("sqlite_tables", {}).items() if not v),
            "mission_files_changed": sorted(k for k, v in cmp.get("mission_files", {}).items() if not v),
            "output_changed": not cmp.get("equal", False),
        }
    report["control_observation_altered"] = controls

    # --- D. altered tape without resealing: the digest check must refuse it
    unsealed = work / "unsealed" / name
    shutil.copytree(stripped, unsealed)
    _rewrite_tape(unsealed, _alter_depth, reseal=False)
    from conrad.sim.mission.capture import verify_capture

    report["control_unsealed_tape_refused"] = verify_capture(unsealed)["problems"]

    # --- headline numbers (post-hoc evaluation with truth restored; outside the audited phase)
    report["headline_after_truth_restored"] = _headline(bundle, work / "play_stripped", stash)

    t = report["truth_removed_replay"]
    identical = bool(t["comparison_with_original"].get("equal"))
    clean = not (t["opened"] or {}).get("truth_paths_opened") and not t["forbidden_modules_imported"]
    detects = all(v.get("output_changed") for v in controls.values() if "skipped" not in v)
    same_code = bool(main["playback"].get("same_code"))
    passed = identical and clean and ok_a and detects and bool(report["control_unsealed_tape_refused"])
    report["verdict"] = "PASS" if passed else ("INCONCLUSIVE_CODE_CHANGED" if not same_code else "FAIL")
    report["checks"] = {
        "same_code_as_recording": same_code,
        "identical_without_truth": identical,
        "no_truth_path_opened_no_truth_module_imported": clean,
        "truth_corruption_has_no_effect": bool(ok_a),
        "observation_change_is_detected": detects,
        "unsealed_edit_refused": bool(report["control_unsealed_tape_refused"]),
    }
    return _write(report, out_root / name)


def _write(report: dict[str, Any], dest: Path) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--child":
        return _child(argv[1], argv[2], argv[3], argv[4:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundles", nargs="+", type=Path)
    ap.add_argument("--out-root", type=Path, default=REPO / "artifacts" / "falsification")
    ap.add_argument("--scratch", type=Path, default=None)
    args = ap.parse_args(argv)
    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="conrad-falsify-"))
    verdicts = {}
    for b in args.bundles:
        rep = falsify(b, scratch, args.out_root)
        verdicts[b.name] = rep["verdict"]
        print(json.dumps({"bundle": b.name, "verdict": rep["verdict"], "checks": rep.get("checks")}))
    print(f"scratch: {scratch}")
    return 0 if all(v == "PASS" for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
