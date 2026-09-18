from __future__ import annotations

from conrad.evaluation.ecological_experiments import e001_field, e002_turbidity, e003_coupling


def test_experiments_run_end_to_end_tiny(tmp_path):
    base = {
        "n_steps": 3,
        "eval_every_steps": 1,
        "model2e": {"entity": {"cover_process_sd_per_sqrt_day": 0.35}},
    }
    r1 = e001_field.run({**base, "sensor_counts": [1, 2]}, [7], tmp_path)
    assert set(r1["per_seed"]["7"]) == {"cefd", "field_only", "static_field"}
    r2 = e002_turbidity.run({**base, "turbidity_levels_ntu": [2.0]}, [7], tmp_path)
    assert "mean_UA" in r2["per_seed"]["7"]["turbidity_2"]["cefd"]
    r3 = e003_coupling.run(base, [7], tmp_path)
    worlds = r3["per_seed"]["7"]
    assert set(worlds) == {"base", "hot_counterfactual", "confounded_stable", "confounded_disturbed"}
    for w in worlds.values():
        assert w["cefd"]["observed_damage_claims"] == 0.0 and w["cefd"]["UEI"] == 0.0
    assert (tmp_path / "2E-E003.json").exists()
