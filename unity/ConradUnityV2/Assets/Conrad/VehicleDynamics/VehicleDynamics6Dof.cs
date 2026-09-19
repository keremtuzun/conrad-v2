// Six-degree-of-freedom vehicle dynamics (ch20/ch21):  M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau + tau_env
// computed by THIS component in double precision (Conrad body frame), then applied to the Rigidbody as
// accelerations (ForceMode.Acceleration). PhysX only integrates and resolves contacts: default Rigidbody drag,
// angular drag and gravity are disabled so none of Unity's built-in hydrodynamics-like terms leak in.
// Assumptions (validity L1): dynamics about the body origin, diagonal inertia and added mass, CoG offset enters
// only through the restoring moments (same as conrad.sim.kernel).
using System;
using Conrad.UnityV2.Core;
using Conrad.UnityV2.EnvironmentInteraction;
using Conrad.UnityV2.Hydrodynamics;
using Conrad.UnityV2.Propulsion;
using UnityEngine;

namespace Conrad.UnityV2.VehicleDynamics
{
    /// <summary>Simulator TRUE state. Served only on the truth endpoint and to sensor models.</summary>
    public struct TruthSnapshot
    {
        public long TimeNs;
        public Vec3d PositionWorld;       // m, Conrad WORLD
        public Quatd WorldFromBody;       // Conrad convention
        public Vec3d LinearVelocityBody;  // m/s
        public Vec3d AngularVelocityBody; // rad/s
        public Vec3d SpecificForceBody;   // m/s^2  (a - g) expressed in body, what an ideal accelerometer measures
        public Vec3d WaterCurrentWorld;   // m/s
    }

    [RequireComponent(typeof(Rigidbody))]
    public sealed class VehicleDynamics6Dof : MonoBehaviour
    {
        private Rigidbody _rb;
        private RobotParameters _p;
        private HydrodynamicsModel _hydro;
        private EnvironmentModel _env;
        private ThrusterBank _thrusters;
        private Vec3d _lastLinearAccelWorld = Vec3d.Zero;
        private Vec3d _externalForceBody = Vec3d.Zero;   // tau_env hook (e.g. tether), body frame
        public Wrench LastHydroWrench { get; private set; }
        public Wrench LastThrustWrench { get; private set; }
        public bool Configured => _p != null;

        public void Configure(RobotParameters parameters, EnvironmentModel environment, ThrusterBank thrusters)
        {
            _p = parameters ?? throw new ArgumentNullException(nameof(parameters));
            _env = environment ?? throw new ArgumentNullException(nameof(environment));
            _thrusters = thrusters ?? throw new ArgumentNullException(nameof(thrusters));
            _hydro = new HydrodynamicsModel(parameters, environment.WaterDensityKgM3);
            _rb = GetComponent<Rigidbody>();
            _rb.useGravity = false;          // gravity is part of g(eta) below
            _rb.linearDamping = 0f;          // hydrodynamic damping is part of D(nu) below
            _rb.angularDamping = 0f;
            _rb.maxAngularVelocity = 100f;
            _rb.sleepThreshold = 0f;         // never freeze a slowly drifting vehicle (neutral buoyancy)
            _rb.isKinematic = false;
            _rb.interpolation = RigidbodyInterpolation.None;
            _rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            _rb.mass = (float)parameters.MassKg;  // used by PhysX for contact impulses only
            _rb.centerOfMass = Vector3.zero;       // dynamics are written about the body origin
            Vec3d iu = ConradFrames.PointToUnity(parameters.InertiaDiag);
            _rb.inertiaTensor = new Vector3(Mathf.Abs((float)iu.X), Mathf.Abs((float)iu.Y), Mathf.Abs((float)iu.Z));
            _rb.inertiaTensorRotation = Quaternion.identity;
            // Hull collider: an axis-aligned box of the RobotConfig dimensions (contacts only; dynamics above).
            var hull = GetComponent<BoxCollider>();
            if (hull != null)
            {
                Vec3d du = ConradFrames.PointToUnity(parameters.Dimensions);
                hull.size = new Vector3(Mathf.Abs((float)du.X), Mathf.Abs((float)du.Y), Mathf.Abs((float)du.Z));
                hull.center = Vector3.zero;
            }
        }

        public void SetExternalForceBody(Vec3d forceBody) { _externalForceBody = forceBody; }

        public void TeleportConrad(Vec3d positionWorld, Quatd worldFromBody)
        {
            _rb.position = ConradFrames.ToEngine(ConradFrames.PointToUnity(positionWorld));
            _rb.rotation = ConradFrames.ToEngine(ConradFrames.QuatToUnity(worldFromBody.Normalized));
            _rb.linearVelocity = Vector3.zero;
            _rb.angularVelocity = Vector3.zero;
            transform.SetPositionAndRotation(_rb.position, _rb.rotation);
            _lastLinearAccelWorld = Vec3d.Zero;
        }

        private void ReadState(out Vec3d pos, out Quatd q, out Vec3d vBody, out Vec3d wBody)
        {
            pos = ConradFrames.PointToConrad(ConradFrames.FromEngine(_rb.position));
            q = ConradFrames.QuatToConrad(ConradFrames.FromEngine(_rb.rotation)).Normalized;
            Vec3d vWorld = ConradFrames.PointToConrad(ConradFrames.FromEngine(_rb.linearVelocity));
            Vec3d wWorld = ConradFrames.AxialToConrad(ConradFrames.FromEngine(_rb.angularVelocity));
            vBody = q.InverseRotate(vWorld);
            wBody = q.InverseRotate(wWorld);
        }

        /// <summary>Compute nu_dot from the model and hand it to PhysX. Call once per physics step, before simulation.</summary>
        public void ApplyDynamics(double simTimeS)
        {
            if (!Configured) throw new InvalidOperationException("VehicleDynamics6Dof.Configure was not called");
            ReadState(out Vec3d pos, out Quatd q, out Vec3d v, out Vec3d w);
            Vec3d currentWorld = _env.Current.VelocityAt(pos, simTimeS);
            Vec3d vRel = v - q.InverseRotate(currentWorld);

            Wrench thrust = _thrusters.TotalWrench();
            Wrench hydro = _hydro.Damping(vRel, w) + _hydro.AddedMassCoriolis(vRel, w)
                         + _hydro.Restoring(q, _hydro.SubmergedFraction(_env.DepthBelowSurface(pos)));
            LastHydroWrench = hydro;
            LastThrustWrench = thrust;

            Vec3d inertia = _p.InertiaDiag;
            // Rigid-body Coriolis/centripetal about the origin: [m w x v ; w x (I w)] moved to the right-hand side.
            Vec3d force = thrust.Force + hydro.Force + _externalForceBody - Vec3d.Cross(w, v) * _p.MassKg;
            Vec3d torque = thrust.Torque + hydro.Torque - Vec3d.Cross(w, Vec3d.Mul(inertia, w));

            Vec3d mLin = new Vec3d(_p.MassKg, _p.MassKg, _p.MassKg) + _hydro.AddedMassLinear;
            Vec3d mAng = inertia + _hydro.AddedMassAngular;
            Vec3d vDot = new Vec3d(force.X / mLin.X, force.Y / mLin.Y, force.Z / mLin.Z);
            Vec3d wDot = new Vec3d(torque.X / mAng.X, torque.Y / mAng.Y, torque.Z / mAng.Z);
            if (!vDot.IsFinite || !wDot.IsFinite) throw new InvalidOperationException("non-finite vehicle acceleration");

            // Inertial acceleration of the origin, expressed in WORLD: R (v_dot + w x v); angular: R w_dot.
            Vec3d aWorld = q.Rotate(vDot + Vec3d.Cross(w, v));
            Vec3d alphaWorld = q.Rotate(wDot);
            _lastLinearAccelWorld = aWorld;
            _rb.AddForce(ConradFrames.ToEngine(ConradFrames.PointToUnity(aWorld)), ForceMode.Acceleration);
            _rb.AddTorque(ConradFrames.ToEngine(ConradFrames.AxialToUnity(alphaWorld)), ForceMode.Acceleration);
        }

        public TruthSnapshot Snapshot(long timeNs, double simTimeS)
        {
            ReadState(out Vec3d pos, out Quatd q, out Vec3d v, out Vec3d w);
            var gravityWorld = new Vec3d(0, 0, -HydrodynamicsModel.StandardGravity);
            return new TruthSnapshot
            {
                TimeNs = timeNs,
                PositionWorld = pos,
                WorldFromBody = q,
                LinearVelocityBody = v,
                AngularVelocityBody = w,
                SpecificForceBody = q.InverseRotate(_lastLinearAccelWorld - gravityWorld),
                WaterCurrentWorld = _env.Current.VelocityAt(pos, simTimeS),
            };
        }
    }
}
