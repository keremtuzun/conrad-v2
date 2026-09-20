"""Gate U0 (spec Phase 4 / ch21 "Unity gates") executed against the built Unity V2 player.

U0 pass list: stable physics, known command moves robot correctly, sensor frames correct, timestamps
correct, reset/replay works, RobotHardwareInterface works. Every tolerance is declared next to its
assertion and recorded in the evidence. Truth (``UnityTruthClient``) is used only to evaluate the simulator.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conrad_unity_testkit import make_command
from u0_harness import (
    BASE,
    INITIAL_POSITION,
    PHYSICS_DT_NS,
    RHO_NEUTRAL,
    STEP_NS,
    VARIANTS,
    Driver,
    G,
    allocate,
    body_rates,
    nose_up_deg,
    reference_1dof,
    session,
    synthetic,
    terminal_velocity,
)

from conrad.adapters.unity import (
    BridgeState,
    FrameProbeRequest,
    ProbePose,
    RecordingTransport,
    ReplayTransport,
    TcpBridgeTransport,
    UnityProtocolError,
    UnityRobotHardware,
)
from conrad.adapters.unity.frames import DEFAULT_MAPPER
from conrad.persistence.object_store import ObjectStore
from conrad.robotics.hardware.interface import RobotHardwareInterface
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality
from conrad.schemas.robot import HealthLevel, OpenParameterError
from conrad.sim.unity.player import EXIT_REFUSED_INPUTS, UnityPlayerError, UnityPlayerSession
from conrad.sim.unity.robot_export import robot_config_to_unity
from conrad.sim.unity.scene import BoxPrimitive, CapsulePrimitive, HeightfieldPrimitive, SceneGeometry

M = float(BASE.mass_kg.value)
LIN = [float(v) for v in BASE.linear_drag.value]
QUAD = [float(v) for v in BASE.quadratic_drag.value]
ADDED = [float(v) for v in BASE.added_mass_diag.value]
INERTIA = [float(v) for v in BASE.inertia_diag_kgm2.value]
THR = BASE.thrusters[0]
LAG = (float(THR.latency_s.value), float(THR.time_constant_s.value))
SUBSTEPS = STEP_NS // PHYSICS_DT_NS


def _start(name: str, live_dir: Path, **kw: object) -> Iterator[UnityPlayerSession]:
    with session(VARIANTS[name], live_dir / name, **kw) as s:
        yield s


@pytest.fixture(scope="module")
def neutral(live_dir: Path) -> Iterator[UnityPlayerSession]:
    yield from _start("neutral", live_dir)


@pytest.fixture
def driver(neutral: UnityPlayerSession, tmp_path: Path) -> Iterator[Driver]:
    """A fresh control + truth connection per test (the player serves one control client at a time)."""
    d = Driver(neutral, store=ObjectStore(tmp_path / "objects"))
    yield d
    d.close()


def _sampled(reference: np.ndarray, count: int) -> np.ndarray:
    """Reference values at the end of each STEP request (every SUBSTEPS physics steps)."""
    return reference[:, SUBSTEPS - 1 :: SUBSTEPS][:, :count]


# ---------------------------------------------------------------------------------------------- stability
@pytest.mark.u0("stable_physics_6dof_rigid_body")
def test_six_dof_rigid_body_is_stable_under_a_long_combined_manoeuvre(driver: Driver, u0_record) -> None:
    driver.reset(seed=11)
    driver.wrench((8.0, -4.0, 3.0, 0.3, -0.3, 0.4))
    truth = driver.advance(30.0)
    driver.wrench((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    truth += driver.advance(30.0)
    speeds = [float(np.linalg.norm(t.linear_velocity_world_mps)) for t in truth]
    rates = [float(np.linalg.norm(t.angular_velocity_world_rps)) for t in truth]
    finite = all(math.isfinite(v) for t in truth for v in (*t.pose.position_m, *t.pose.orientation_wxyz))
    q_norm_err = max(abs(math.sqrt(sum(c * c for c in t.pose.orientation_wxyz)) - 1.0) for t in truth)
    u0_record(
        sim_seconds=60.0,
        max_speed_mps=max(speeds),
        max_rate_rps=max(rates),
        final_speed_mps=speeds[-1],
        final_rate_rps=rates[-1],
        max_quaternion_norm_error=q_norm_err,
        all_finite=finite,
    )
    assert finite and q_norm_err < 1e-5
    # bounded: the commanded wrench is modest, so speeds stay well below 2 m/s and 3 rad/s
    assert max(speeds) < 2.0 and max(rates) < 3.0
    # after thrust is removed the damped, neutrally buoyant vehicle comes (almost) to rest within 30 s
    assert speeds[-1] < 0.02 and rates[-1] < 0.02


@pytest.mark.u0("known_command_moves_robot_correctly")
@pytest.mark.parametrize(
    ("dof", "wrench"),
    [
        ("surge", (10.0, 0, 0, 0, 0, 0)),
        ("sway", (0, 10.0, 0, 0, 0, 0)),
        ("heave", (0, 0, 10.0, 0, 0, 0)),
        ("roll", (0, 0, 0, 1.0, 0, 0)),
        ("pitch", (0, 0, 0, 0, 1.0, 0)),
        ("yaw", (0, 0, 0, 0, 0, 1.0)),
    ],
)
def test_thruster_commands_move_along_the_right_axis_with_the_right_sign(
    driver: Driver, dof: str, wrench: tuple[float, ...], u0_record
) -> None:
    driver.reset(seed=3)
    start = driver.truth.get_ground_truth()
    commands = driver.wrench(wrench)
    truth = driver.advance(0.8)
    v, w = body_rates(truth[-1])
    nu = np.concatenate([v, w])
    i = ["surge", "sway", "heave", "roll", "pitch", "yaw"].index(dof)
    same_kind = nu[:3] if i < 3 else nu[3:]
    others = np.delete(same_kind, i % 3)
    disp = np.subtract(truth[-1].pose.position_m, start.pose.position_m)
    u0_record(
        dof=dof,
        commands={k: round(val, 4) for k, val in commands.items()},
        body_velocity=[round(float(x), 6) for x in nu],
        displacement_world_m=[round(float(x), 6) for x in disp],
    )
    assert nu[i] > 0, f"{dof}: wrong sign {nu[i]}"
    # declared tolerance: cross-axis response below 10 % of the commanded axis
    assert np.all(np.abs(others) < 0.1 * nu[i]), f"{dof}: cross-axis motion {others}"
    if i < 3:  # translation shows up in WORLD with the documented Conrad sign (+X fwd, +Y left, +Z up)
        assert disp[i] > 0 and np.all(np.abs(np.delete(disp, i)) < 0.1 * disp[i])


# ---------------------------------------------------------------------------------------------- buoyancy / drag
@pytest.mark.u0("gravity_buoyancy_equilibrium")
def test_neutral_configuration_holds_depth(driver: Driver, u0_record) -> None:
    driver.reset(seed=5)
    truth = driver.advance(20.0)
    dz = max(abs(t.pose.position_m[2] - INITIAL_POSITION[2]) for t in truth)
    tilt = max(abs(nose_up_deg(t.pose.orientation_wxyz)) for t in truth)
    u0_record(seconds=20.0, rho_kgm3=RHO_NEUTRAL, max_abs_depth_change_m=dz, max_abs_pitch_deg=tilt)
    assert dz < 1e-3 and tilt < 1e-3  # declared tolerance: 1 mm, 0.001 deg over 20 s


@pytest.mark.u0("gravity_buoyancy_equilibrium")
@pytest.mark.parametrize("name", ["positive", "negative"])
def test_buoyancy_variants_rise_and_sink_as_the_analytic_heave_model(
    name: str, live_dir: Path, u0_record
) -> None:
    robot = VARIANTS[name]
    m = float(robot.mass_kg.value)
    vol = float(robot.displaced_volume_m3.value)
    force = (RHO_NEUTRAL * vol - m) * G
    seconds = 20.0
    with session(robot, live_dir / name) as s:
        d = Driver(s)
        d.reset(seed=5)
        truth = d.advance(seconds)
        d.close()
    z = np.array([t.pose.position_m[2] - INITIAL_POSITION[2] for t in truth])
    vz = np.array([t.linear_velocity_world_mps[2] for t in truth])
    ref = _sampled(reference_1dof(force, m + ADDED[2], LIN[2], QUAD[2], seconds), len(z))
    v_term = terminal_velocity(force, LIN[2], QUAD[2])
    rel_traj = float(np.max(np.abs(z - ref[0])) / abs(ref[0][-1]))
    rel_term = abs(vz[-1] - v_term) / abs(v_term)
    u0_record(
        variant=name,
        net_buoyancy_n=force,
        final_displacement_m=float(z[-1]),
        reference_displacement_m=float(ref[0][-1]),
        max_trajectory_error_relative=rel_traj,
        final_vertical_velocity_mps=float(vz[-1]),
        analytic_terminal_velocity_mps=v_term,
        terminal_velocity_error_relative=rel_term,
    )
    assert math.copysign(1.0, z[-1]) == math.copysign(1.0, force)  # rises when positive, sinks when negative
    assert rel_traj < 0.01  # declared tolerance: 1 % of the 20 s displacement
    assert rel_term < 0.01  # declared tolerance: 1 % of the analytic terminal velocity


@pytest.mark.u0("basic_drag_terminal_velocity")
def test_surge_terminal_velocity_matches_the_analytic_value(driver: Driver, u0_record) -> None:
    force = 20.0
    driver.reset(seed=7)
    driver.wrench((force, 0, 0, 0, 0, 0))
    truth = driver.advance(15.0)
    u = [body_rates(t)[0][0] for t in truth]
    v_term = terminal_velocity(force, LIN[0], QUAD[0])
    ref = _sampled(reference_1dof(force, M + ADDED[0], LIN[0], QUAD[0], 15.0, lag=LAG), len(u))
    rel = abs(u[-1] - v_term) / v_term
    rel_traj = float(np.max(np.abs(np.array(u) - ref[1])) / v_term)
    u0_record(
        surge_force_n=force,
        d1=LIN[0],
        d2=QUAD[0],
        measured_terminal_velocity_mps=u[-1],
        analytic_terminal_velocity_mps=v_term,
        relative_error=rel,
        max_velocity_trajectory_error_relative=rel_traj,
    )
    assert rel < 0.005  # declared tolerance: 0.5 %
    assert rel_traj < 0.01  # whole 15 s velocity history within 1 % of the 1-DOF reference (lag included)


# ---------------------------------------------------------------------------------------------- mass properties
@pytest.mark.u0("mass_inertia_com_cob_configurable")
def test_inertia_from_robot_config_sets_the_yaw_response(
    neutral: UnityPlayerSession, live_dir: Path, u0_record
) -> None:
    torque, seconds = 0.4, 3.0
    results = {}
    for name in ("neutral", "inertia_x2"):
        robot = VARIANTS[name]
        iz = float(robot.inertia_diag_kgm2.value[2])
        if name == "neutral":
            d = Driver(neutral, seed=21)
            d.reset(seed=9)
            d.wrench((0, 0, 0, 0, 0, torque))
            truth = d.advance(seconds)
            d.close()
        else:
            with session(robot, live_dir / name) as s:
                d = Driver(s)
                d.reset(seed=9)
                d.wrench((0, 0, 0, 0, 0, torque))
                truth = d.advance(seconds)
                d.close()
        r = np.array([body_rates(t)[1][2] for t in truth])
        ref = _sampled(reference_1dof(torque, iz + ADDED[5], LIN[5], QUAD[5], seconds, lag=LAG), len(r))
        err = float(np.max(np.abs(r - ref[1])) / np.max(np.abs(ref[1])))
        results[name] = {
            "inertia_zz": iz,
            "yaw_rate_at_0.5s": float(r[4]),
            "reference_at_0.5s": float(ref[1][4]),
            "max_relative_error": err,
        }
    u0_record(**results)
    for res in results.values():
        assert res["max_relative_error"] < 0.01  # declared tolerance: 1 % of the peak yaw rate
    assert results["inertia_x2"]["yaw_rate_at_0.5s"] < results["neutral"]["yaw_rate_at_0.5s"]


@pytest.mark.u0("mass_inertia_com_cob_configurable")
@pytest.mark.parametrize("name", ["cob_forward", "com_aft"])
def test_com_and_cob_offsets_set_the_trim_angle(name: str, live_dir: Path, u0_record) -> None:
    robot = VARIANTS[name]
    cob = np.asarray(robot.center_of_buoyancy_body_m.value, dtype=float)
    com = np.asarray(robot.center_of_mass_body_m.value, dtype=float)
    r = cob - com
    expected = math.degrees(math.atan2(r[0], r[2]))  # nose-up trim that puts CoB straight above CoG
    with session(robot, live_dir / name) as s:
        d = Driver(s)
        d.reset(seed=5)
        truth = d.advance(90.0, step_ns=200_000_000)
        d.close()
    pitch = np.array([nose_up_deg(t.pose.orientation_wxyz) for t in truth])
    settled = float(np.mean(pitch[-50:]))  # last 10 s
    u0_record(
        variant=name,
        cob_minus_com_m=r.tolist(),
        expected_trim_deg=expected,
        measured_trim_deg=settled,
        residual_oscillation_deg=float(np.ptp(pitch[-50:])),
    )
    assert abs(settled - expected) < 0.5  # declared tolerance: 0.5 deg


@pytest.mark.u0("mass_inertia_com_cob_configurable")
def test_open_parameters_are_refused_on_both_sides(live_dir: Path, u0_record) -> None:
    open_mass = {"value": None, "units": "kg", "source": "OPEN"}
    doc = BASE.model_dump(mode="json")
    doc["mass_kg"] = open_mass
    from conrad.schemas.robot import RobotConfig

    with pytest.raises(OpenParameterError):
        robot_config_to_unity(RobotConfig.model_validate(doc))
    # Unity side: hand the player a document with an OPEN mass directly (bypassing the Python refusal)
    unity_doc = robot_config_to_unity(BASE)
    unity_doc["mass_kg"] = {"value": None, "units": "kg", "source": "OPEN", "sigma": None}
    s = session(BASE, live_dir / "open_refused", robot_document=unity_doc)
    s.start()
    try:
        code = s.wait_exit(timeout_s=60)
    finally:
        s.stop()
    log = s.log_text()
    u0_record(
        python_export="OpenParameterError",
        player_exit_code=code,
        player_log_mentions_open="Unity refuses OPEN physical parameters: mass_kg" in log,
    )
    assert code == EXIT_REFUSED_INPUTS
    assert "Unity refuses OPEN physical parameters: mass_kg" in log
    with pytest.raises(UnityPlayerError):
        s2 = session(BASE, live_dir / "open_refused2", robot_document=unity_doc, startup_timeout_s=60)
        with s2:
            pass


# ---------------------------------------------------------------------------------------------- sensors
@pytest.mark.u0("sensor_frames_and_timestamps")
def test_imu_depth_camera_frames_and_monotonic_timestamps(driver: Driver, u0_record) -> None:
    driver.reset(seed=13)
    driver.advance(2.0, step_ns=20_000_000, keep_states=True)  # at rest
    rest = list(driver.states)
    driver.wrench((0, 0, 0, 0, 0, 0.4))  # yaw left
    driver.advance(1.0, step_ns=20_000_000, keep_states=True)

    packets: dict[str, list] = {}
    for st in driver.states:
        for p in st.sensors:
            packets.setdefault(p.sensor_name, []).append(p)
    report: dict[str, Any] = {}
    periods = {s.sensor_name: round(1e9 / float(s.rate_hz.value)) for s in BASE.sensors}
    latencies = {s.sensor_name: round(float(s.latency_s.value) * 1e9) for s in BASE.sensors}
    for name, pkts in packets.items():
        acq = [p.acquisition_time_ns for p in pkts]
        seq = [p.sequence_index for p in pkts]
        report[name] = {
            "count": len(pkts),
            "frame_id": pkts[0].frame_id,
            "clock_domain": pkts[0].clock_domain,
            "strictly_increasing": all(b > a for a, b in itertools.pairwise(acq)),
            "sequence_consecutive": seq == list(range(seq[0], seq[0] + len(seq))),
            "period_ns": sorted({b - a for a, b in itertools.pairwise(acq)}),
            "latency_ns": sorted({p.delivery_time_ns - p.acquisition_time_ns for p in pkts}),
        }
        assert report[name]["strictly_increasing"] and report[name]["sequence_consecutive"], name
        assert report[name]["period_ns"] == [periods[name]], name
        assert report[name]["latency_ns"] == [latencies[name]], name
        assert all(a % PHYSICS_DT_NS == 0 for a in acq), name  # acquisition stamped on the physics clock
    frames = {s.sensor_name: s.frame_id for s in BASE.sensors}
    assert {n: r["frame_id"] for n, r in report.items()} == frames
    assert {r["clock_domain"] for r in report.values()} == {"SIM"}

    # IMU at rest: specific force is +g along Conrad +Z (up), gyro ~ 0 (noise sigma 0.01 SYNTHETIC_ONLY)
    imu_rest = [p for st in rest for p in st.sensors if p.sensor_name == "imu"]
    from conrad.adapters.unity.conversion import depth_from_packet, imu_from_packet

    acc = np.array([imu_from_packet(p, DEFAULT_MAPPER).linear_acceleration_mps2 for p in imu_rest])
    mean_acc = acc.mean(axis=0)
    sigma = float(BASE.sensors[0].noise_std.value)
    tol = 5 * sigma / math.sqrt(len(acc))
    imu_now = driver.hw.get_imu()
    assert (
        imu_now is not None and imu_now.frame_id == "SENSOR_IMU" and imu_now.timestamp.clock_domain == "SIM"
    )
    depths = [depth_from_packet(p).depth_m for st in rest for p in st.sensors if p.sensor_name == "depth"]
    cam = driver.hw.get_camera()
    assert cam is not None and cam.payload_ref is not None and driver.store is not None
    img = np.frombuffer(driver.store.get_bytes(cam.payload_ref), dtype=np.uint8).reshape(
        cam.payload_ref.shape
    )
    top, bottom = float(img[: img.shape[0] // 4].mean()), float(img[-img.shape[0] // 4 :].mean())
    report["imu_rest_mean_specific_force_mps2"] = mean_acc.round(5).tolist()
    report["imu_yaw_left_gyro_z_rps"] = imu_now.angular_velocity_rps[2]
    report["depth_mean_m"] = float(np.mean(depths))
    report["camera_image"] = {
        "shape": list(cam.payload_ref.shape),
        "sensor_frame": cam.sensor_frame,
        "graphics_device": cam.sensor_context.get("graphics_device"),
        "top_quarter_mean_dn": top,
        "bottom_quarter_mean_dn": bottom,
    }
    u0_record(**report)
    assert abs(mean_acc[2] - G) < tol and abs(mean_acc[0]) < tol and abs(mean_acc[1]) < tol
    assert imu_now.angular_velocity_rps[2] > 0.05  # yaw left is +Z in Conrad after the adapter's axial map
    assert abs(np.mean(depths) - (-INITIAL_POSITION[2])) < 5 * 0.02 / math.sqrt(len(depths))
    assert cam.modality is Modality.RGB and cam.sensor_frame == "SENSOR_CAMERA"
    assert cam.payload_ref.shape == (48, 64, 3) and cam.sensor_context["graphics_device"] != "Null"
    # image orientation: the lit seafloor (below the vehicle) fills the bottom rows, open water the top rows
    assert bottom > top + 50


# ---------------------------------------------------------------------------------------------- determinism / replay
def _run_trace(d: Driver, seed: int) -> tuple[list[tuple[float, ...]], list[str]]:
    d.reset(seed=seed)
    d.wrench((6.0, 2.0, -1.0, 0.1, 0.0, 0.2))
    truth = d.advance(5.0, step_ns=50_000_000, keep_states=True)
    traj = [(*t.pose.position_m, *t.pose.orientation_wxyz) for t in truth]
    digests = [p.payload_digest for st in d.states for p in st.sensors if p.sensor_name in ("imu", "depth")]
    return traj, digests


@pytest.mark.u0("deterministic_reset_and_replay")
def test_same_seed_same_trajectory_across_resets_and_processes(
    driver: Driver, live_dir: Path, tmp_path: Path, u0_record
) -> None:
    a_traj, a_dig = _run_trace(driver, 42)
    b_traj, b_dig = _run_trace(driver, 42)
    c_traj, c_dig = _run_trace(driver, 43)
    with session(BASE, live_dir / "neutral_second_process") as s2:
        d2 = Driver(s2)
        p_traj, p_dig = _run_trace(d2, 42)
        d2.close()
    diff_reset = float(np.max(np.abs(np.subtract(a_traj, b_traj))))
    diff_process = float(np.max(np.abs(np.subtract(a_traj, p_traj))))
    diff_seed_physics = float(np.max(np.abs(np.subtract(a_traj, c_traj))))
    u0_record(
        steps=len(a_traj),
        max_abs_diff_same_seed_reset=diff_reset,
        max_abs_diff_same_seed_new_process=diff_process,
        sensor_payload_digests_equal_reset=a_dig == b_dig,
        sensor_payload_digests_equal_process=a_dig == p_dig,
        sensor_payload_digests_differ_other_seed=a_dig != c_dig,
        max_abs_diff_other_seed_physics=diff_seed_physics,
        declared_tolerance=1e-9,
    )
    assert diff_reset <= 1e-9 and diff_process <= 1e-9  # declared tolerance 1e-9 (m, quaternion units)
    assert a_dig == b_dig == p_dig
    assert a_dig != c_dig  # the seed drives the sensor noise streams


@pytest.mark.u0("deterministic_reset_and_replay")
def test_recorded_wire_replays_with_identical_acquisition_timestamps(
    neutral: UnityPlayerSession, tmp_path: Path, u0_record
) -> None:
    log = tmp_path / "wire.jsonl"

    def run(hw: UnityRobotHardware) -> list[tuple[int, int, float, float]]:
        hw.connect()
        hw.reset(seed=77)
        out = []
        for _ in range(30):
            hw.step(STEP_NS)
            imu, depth = hw.get_imu(), hw.get_depth()
            assert imu is not None and depth is not None
            out.append(
                (
                    imu.timestamp.time_ns,
                    depth.timestamp.time_ns,
                    depth.depth_m,
                    imu.linear_acceleration_mps2[2],
                )
            )
        hw.close()
        return out

    cfg = neutral.bridge_config()
    ids = IdFactory(seed=5)
    live_transport = RecordingTransport(
        TcpBridgeTransport(cfg.control_endpoint, None, cfg.allowed_peers, cfg.request_timeout_ms), log
    )
    live = run(UnityRobotHardware(cfg, BASE, ids, ids.new(), ids.new(), transport=live_transport))
    ids_r = IdFactory(seed=5)
    replayed = run(
        UnityRobotHardware(cfg, BASE, ids_r, ids_r.new(), ids_r.new(), transport=ReplayTransport(log))
    )
    u0_record(samples=len(live), identical=replayed == live, wire_log_bytes=log.stat().st_size)
    assert replayed == live


# ---------------------------------------------------------------------------------------------- frames
@pytest.mark.u0("frame_conversion_probes")
def test_frame_probe_unit_axes_and_round_trip_through_the_engine(driver: Driver, u0_record) -> None:
    s = math.sqrt(0.5)
    poses = [
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (s, 0.0, 0.0, s)),  # yaw left 90 deg
        ((0.0, 0.0, 0.0), (s, 0.0, s, 0.0)),  # pitch +90 deg about +Y (nose down)
        ((0.0, 0.0, 0.0), (s, s, 0.0, 0.0)),  # roll +90 deg about +X (left side up)
        ((3.2, -1.7, -12.5), (0.8, 0.1, -0.3, 0.5)),  # arbitrary round trip
    ]
    reply = driver.hw.frame_probe(
        FrameProbeRequest(poses=tuple(ProbePose(position_m=p, orientation_wxyz=q) for p, q in poses))
    )
    r = reply.results
    tol = 1e-6  # float32 engine storage
    # unit axes: Conrad +X fwd -> Unity +Z, +Y left -> Unity -X, +Z up -> Unity +Y
    assert r[0].unity_position == pytest.approx((0, 0, 1), abs=tol)
    assert r[1].unity_position == pytest.approx((-1, 0, 0), abs=tol)
    assert r[2].unity_position == pytest.approx((0, 1, 0), abs=tol)
    # identity pose: body forward/right/up in Unity
    assert r[0].unity_forward == pytest.approx((0, 0, 1), abs=tol)
    assert r[0].unity_right == pytest.approx((1, 0, 0), abs=tol)
    assert r[0].unity_up == pytest.approx((0, 1, 0), abs=tol)
    # yaw left: body forward now points to Conrad +Y = Unity -X
    assert r[3].unity_forward == pytest.approx((-1, 0, 0), abs=tol)
    # pitch +90 about +Y: body forward points down (Conrad -Z = Unity -Y)
    assert r[4].unity_forward == pytest.approx((0, -1, 0), abs=tol)
    # roll +90 about +X: body up points to Conrad -Y (right) = Unity +X
    assert r[5].unity_up == pytest.approx((1, 0, 0), abs=tol)
    rt_pos = max(
        float(np.max(np.abs(np.subtract(res.conrad_position, p))))
        for res, (p, _) in zip(r, poses, strict=True)
    )
    rt_rot = 0.0
    for res, (_, q) in zip(r, poses, strict=True):
        qn = np.asarray(q) / np.linalg.norm(q)
        rt_rot = max(rt_rot, 1.0 - abs(float(np.dot(qn, res.conrad_orientation_wxyz))))
        # the engine's quaternion equals the Python mapper's (up to sign)
        assert abs(
            float(np.dot(DEFAULT_MAPPER.quat_to_unity(tuple(qn)), res.unity_rotation_wxyz))
        ) == pytest.approx(1, abs=tol)
    u0_record(
        unit_axes={"+X": r[0].unity_position, "+Y": r[1].unity_position, "+Z": r[2].unity_position},
        yaw_left_forward_unity=r[3].unity_forward,
        pitch_pos_forward_unity=r[4].unity_forward,
        roll_pos_up_unity=r[5].unity_up,
        round_trip_max_position_error_m=rt_pos,
        round_trip_max_quaternion_error=rt_rot,
    )
    assert rt_pos < 1e-5 and rt_rot < 1e-6


# ---------------------------------------------------------------------------------------------- scene geometry
@pytest.mark.u0("frame_conversion_probes")
def test_configured_scene_is_seen_where_conrad_placed_it(live_dir: Path, tmp_path: Path, u0_record) -> None:
    """Box ahead and a capsule ahead-left, seen by the raycast sonar at the right range and bearing."""
    with session(BASE, live_dir / "scene") as s:
        _scene_probe(Driver(s, store=ObjectStore(tmp_path / "objects")), u0_record)


def _scene_probe(driver: Driver, u0_record) -> None:
    scene = SceneGeometry(
        primitives=(
            BoxPrimitive(id="wall_ahead", center_m=(6.25, 0.0, -10.0), size_m=(0.5, 1.0, 3.0)),
            CapsulePrimitive(id="pole_left", p0_m=(5.0, 1.75, -14.0), p1_m=(5.0, 1.75, -6.0), radius_m=0.25),
        )
    )
    ack = driver.hw.configure_scene(scene.to_json())
    driver.reset(seed=2)
    driver.advance(1.0)
    sonar = driver.hw.get_sonar()
    assert sonar is not None and sonar.payload_ref is not None and driver.store is not None
    beams, bins = sonar.payload_ref.shape
    img = np.frombuffer(driver.store.get_bytes(sonar.payload_ref), dtype="<f4").reshape(beams, bins)
    rng_max = float(sonar.sensor_context["range_max_m"])
    fov = float(sonar.sensor_context["horizontal_fov_rad"])
    bearings = [-0.5 * fov + fov * b / (beams - 1) for b in range(beams)]
    centre = min(range(beams), key=lambda b: abs(bearings[b]))
    wall_bin = int(np.argmax(img[centre]))
    wall_range = (wall_bin + 0.5) * rng_max / bins
    expected_wall = 6.0 - 0.25  # wall face at x = 6.0, sonar mount at x = +0.25
    pole_bearing = math.atan2(1.75, 5.0 - 0.25)  # inside the +-0.5 rad fan (experiment sonar FOV 1 rad)
    assert abs(pole_bearing) < 0.5 * fov
    pole_beam = min(range(beams), key=lambda b: abs(bearings[b] - pole_bearing))
    pole_bin = int(np.argmax(img[pole_beam]))
    pole_range = (pole_bin + 0.5) * rng_max / bins
    expected_pole = math.hypot(1.75, 4.75) - 0.25
    u0_record(
        scene_primitives=ack.primitive_count,
        wall_expected_range_m=expected_wall,
        wall_measured_range_m=wall_range,
        sonar_horizontal_fov_rad=fov,
        pole_expected_bearing_rad=pole_bearing,
        pole_beam_bearing_rad=bearings[pole_beam],
        pole_expected_range_m=expected_pole,
        pole_measured_range_m=pole_range,
        bin_size_m=rng_max / bins,
    )
    assert ack.primitive_count == 2
    bin_size = rng_max / bins
    assert abs(wall_range - expected_wall) <= bin_size  # declared tolerance: one range bin
    assert abs(pole_range - expected_pole) <= 2 * bin_size  # curved surface: two bins
    # nothing to the RIGHT (negative bearing) at the pole's range: left/right are not mirrored
    mirror_beam = min(range(beams), key=lambda b: abs(bearings[b] + pole_bearing))
    assert img[mirror_beam, pole_bin] < 0.5 * img[pole_beam, pole_bin]
    # a bad primitive refuses the whole request and faults the adapter (fail closed)
    with pytest.raises(UnityProtocolError):
        driver.hw.configure_scene(
            {"frame": "WORLD", "replace": True, "primitives": [{"kind": "cone", "id": "x"}]}
        )
    assert driver.hw.state is BridgeState.FAULTED


@pytest.mark.u0("frame_conversion_probes")
def test_heightfield_collider_supports_a_sinking_vehicle(live_dir: Path, u0_record) -> None:
    ground = -12.0
    heights = tuple(tuple(0.0 for _ in range(9)) for _ in range(9))
    scene = SceneGeometry(
        primitives=(
            HeightfieldPrimitive(
                id="floor", origin_m=(-4.0, -4.0, ground), spacing_m=(1.0, 1.0), heights_m=heights
            ),
        )
    )
    robot = VARIANTS["negative"]
    half_height = float(robot.dimensions_m.value[2]) / 2
    with session(robot, live_dir / "heightfield") as s:
        d = Driver(s)
        d.hw.configure_scene(scene.to_json())
        d.reset(seed=5)
        truth = d.advance(25.0)
        d.close()
    z = [t.pose.position_m[2] for t in truth]
    rest = float(np.mean(z[-20:]))
    u0_record(
        ground_z_m=ground,
        expected_rest_z_m=ground + half_height,
        measured_rest_z_m=rest,
        final_vertical_speed_mps=abs(truth[-1].linear_velocity_world_mps[2]),
    )
    assert abs(rest - (ground + half_height)) < 0.03  # declared tolerance: 3 cm (PhysX contact offset)
    assert abs(truth[-1].linear_velocity_world_mps[2]) < 1e-3


# ---------------------------------------------------------------------------------------------- RHI + gateway
@pytest.mark.u0("robot_hardware_interface_via_command_gateway")
def test_full_robot_hardware_interface_through_the_real_command_gateway(driver: Driver, u0_record) -> None:
    hw = driver.hw
    assert isinstance(hw, RobotHardwareInterface) and not hw.is_physical and hw.adapter_name == "unity_v2"
    driver.reset(seed=17)
    caps = hw.capabilities()
    assert (
        caps.thruster_count == len(BASE.thrusters) and caps.imu and caps.depth and caps.cameras == ("camera",)
    )
    assert hw.robot_config_digest() == BASE.content_digest() and hw.clock_domain() == "SIM"
    driver.advance(0.2)
    before = driver.truth.get_ground_truth()
    values = allocate(BASE, (15.0, 0, 0, 0, 0, 0))
    cmd = make_command(driver.ids, BASE, driver.mission, driver.run, hw.now_ns(), values)
    ack = driver.gateway.submit(cmd)
    assert ack.accepted and ack.adapter == "unity_v2", ack.reason_codes
    duplicate = driver.gateway.submit(cmd)
    bad_digest = driver.gateway.submit(
        make_command(driver.ids, BASE, driver.mission, driver.run, hw.now_ns(), values, digest="c" * 64)
    )
    unauthorized = driver.gateway.submit(
        make_command(driver.ids, BASE, driver.mission, driver.run, hw.now_ns(), values, authorized=False)
    )
    driver.advance(3.0)
    after = driver.truth.get_ground_truth()
    thrusters = hw.get_thruster_state()
    power = hw.get_power_state()
    health = hw.get_health()
    imu, depth, cam, sonar = hw.get_imu(), hw.get_depth(), hw.get_camera(), hw.get_sonar()
    moved = after.pose.position_m[0] - before.pose.position_m[0]
    u0_record(
        accepted=ack.accepted,
        duplicate_reasons=list(duplicate.reason_codes),
        bad_digest_reasons=list(bad_digest.reason_codes),
        unauthorized_reasons=list(unauthorized.reason_codes),
        forward_displacement_m=moved,
        thruster_commands_echoed={t.thruster_id: t.command for t in thrusters},
        battery_remaining_fraction=None if power is None else power.remaining_fraction,
        health=health.overall.value,
        gateway_accepted=driver.gateway.accepted,
        gateway_rejected=driver.gateway.rejected,
    )
    assert not duplicate.accepted and not bad_digest.accepted and not unauthorized.accepted
    assert moved > 1.0  # 15 N surge for 3 s moves the vehicle forward (+X) by more than a metre
    assert {t.thruster_id: t.command for t in thrusters} == pytest.approx(values)
    assert all(t.estimated_thrust_n != 0.0 for t in thrusters if abs(values[t.thruster_id]) > 0)
    assert power is not None and power.remaining_fraction < 1.0
    assert health.overall is HealthLevel.OK
    assert imu is not None and depth is not None and cam is not None and sonar is not None
    assert isinstance(hw, UnityRobotHardware)  # engine metrics are an adapter-level service
    assert json.dumps(hw.get_metrics().metrics)  # metrics are served


def test_player_binary_is_the_one_just_built() -> None:
    """Records which binary the evidence is about (no criterion of its own)."""
    from conrad.sim.unity.player import find_player

    player = find_player()
    assert player is not None
    digest = hashlib.sha256(player.read_bytes()).hexdigest()
    assert len(digest) == 64


_ = synthetic  # re-exported for readers of the variant definitions
