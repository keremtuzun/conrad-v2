// Thruster / actuation model (ch21 Thruster model, Thruster dynamics, Thruster geometry, Thruster failures).
//   T_c(u) = 0 for |u| < deadzone; k u^2 forward; -k_rev u^2 reverse (k_rev = k * maxRev / maxFwd); saturated
//   tau dT/dt + T = T_c(u(t - latency))       exact first-order discretisation per physics step
//   + multiplicative noise, faults: failure, degradation, stuck command, extra latency, intermittent dropout.
// Same functional form as conrad.sim.kernel and conrad.robotics.hardware.identification.models.
using System;
using System.Collections.Generic;
using Conrad.UnityV2.Core;
using Conrad.UnityV2.Hydrodynamics;

namespace Conrad.UnityV2.Propulsion
{
    public sealed class ThrusterFaultState
    {
        public double Effectiveness = 1.0;     // 0 = failed, (0,1) = degraded
        public double? StuckCommand;           // null = not stuck
        public double ExtraLatencyS;
        public double IntermittentDropProbability; // per physics step
        public bool Failed => Effectiveness <= 0.0;
    }

    public sealed class Thruster
    {
        private readonly ThrusterParameters _p;
        private readonly SeededRandom _rng;
        private readonly LinkedList<(long timeNs, double command)> _pending = new LinkedList<(long, double)>();
        private double _appliedCommand;
        public double NoiseFraction;
        public ThrusterFaultState Fault { get; } = new ThrusterFaultState();
        public double ThrustN { get; private set; }
        public double LastCommand { get; private set; }
        public ThrusterParameters Parameters => _p;

        public Thruster(ThrusterParameters p, SeededRandom rng, double noiseFraction)
        {
            _p = p;
            _rng = rng;
            NoiseFraction = noiseFraction;
            double n = p.DirectionBody.Norm;
            if (Math.Abs(n - 1.0) > 1e-3) throw new ArgumentException("thruster " + p.Id + " direction is not a unit vector");
        }

        public void Command(long nowNs, double u)
        {
            if (double.IsNaN(u) || u < -1 || u > 1) throw new ArgumentOutOfRangeException(nameof(u));
            LastCommand = u;
            _pending.AddLast((nowNs, u));
        }

        public double CommandedThrust(double u)
        {
            if (Math.Abs(u) < _p.Deadzone) return 0.0;
            if (u >= 0) return Math.Min(_p.K * u * u, _p.MaxForwardN);
            double kRev = _p.MaxForwardN > 0 ? _p.K * _p.MaxReverseN / _p.MaxForwardN : _p.K;
            return -Math.Min(kRev * u * u, _p.MaxReverseN);
        }

        public void Step(long nowNs, double dtS)
        {
            long delayNs = (long)Math.Round((_p.LatencyS + Fault.ExtraLatencyS) * SimClock.NsPerSecond);
            while (_pending.First != null && _pending.First.Value.timeNs + delayNs <= nowNs)
            {
                _appliedCommand = _pending.First.Value.command;
                _pending.RemoveFirst();
            }
            double u = Fault.StuckCommand ?? _appliedCommand;
            double target = CommandedThrust(u) * Math.Max(0.0, Fault.Effectiveness);
            if (Fault.IntermittentDropProbability > 0 && _rng.NextDouble() < Fault.IntermittentDropProbability) target = 0.0;
            double alpha = _p.TimeConstantS > 0 ? 1.0 - Math.Exp(-dtS / _p.TimeConstantS) : 1.0;
            ThrustN += alpha * (target - ThrustN);
            if (NoiseFraction > 0 && ThrustN != 0) ThrustN += _rng.Gaussian(NoiseFraction * Math.Abs(ThrustN));
        }

        public Wrench BodyWrench()
        {
            Vec3d f = _p.DirectionBody * ThrustN;
            return new Wrench(f, Vec3d.Cross(_p.PositionBody, f));
        }

        public void Reset()
        {
            _pending.Clear();
            _appliedCommand = 0;
            ThrustN = 0;
            LastCommand = 0;
            Fault.Effectiveness = 1.0;
            Fault.StuckCommand = null;
            Fault.ExtraLatencyS = 0;
            Fault.IntermittentDropProbability = 0;
        }
    }

    public sealed class ThrusterBank
    {
        private readonly Dictionary<string, Thruster> _byId = new Dictionary<string, Thruster>(StringComparer.Ordinal);
        public IReadOnlyList<Thruster> Thrusters => _list;
        private readonly List<Thruster> _list = new List<Thruster>();

        public ThrusterBank(RobotParameters p, ulong seed, double noiseFraction)
        {
            foreach (var t in p.Thrusters)
            {
                var th = new Thruster(t, new SeededRandom(seed, "thruster:" + t.Id), noiseFraction);
                _byId.Add(t.Id, th);
                _list.Add(th);
            }
        }

        public bool TryGet(string id, out Thruster t) => _byId.TryGetValue(id, out t);

        public IEnumerable<string> Ids => _byId.Keys;

        public void Step(long nowNs, double dtS)
        {
            foreach (var t in _list) t.Step(nowNs, dtS);
        }

        public Wrench TotalWrench()
        {
            Wrench w = Wrench.Zero;
            foreach (var t in _list) w += t.BodyWrench();
            return w;
        }

        /// <summary>Electrical power proxy P = c |T|^1.5 (SYNTHETIC_ONLY coefficient until bench power curves exist).</summary>
        public double ElectricalPowerW(double coefficient)
        {
            double p = 0;
            foreach (var t in _list) p += coefficient * Math.Pow(Math.Abs(t.ThrustN), 1.5);
            return p;
        }

        public void Reset()
        {
            foreach (var t in _list) t.Reset();
        }
    }
}
