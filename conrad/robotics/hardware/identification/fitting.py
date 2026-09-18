"""Least-squares core with parameter uncertainty from the Jacobian (ch20/ch21 Parameter uncertainty).

``cov = s^2 (J^T J)^-1`` with ``s^2 = SSR / (n - p)``. When ``J^T J`` is rank-deficient the parameter
is reported unidentifiable (sigma = None) instead of producing a meaningless number.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from pydantic import Field
from scipy.optimize import least_squares

from conrad.schemas.base import ConradModel

Residual = Callable[[np.ndarray], np.ndarray]


class FittedParameter(ConradModel):
    name: str
    value: float
    sigma: float | None = Field(description="1-sigma from the Jacobian; None = not identifiable")
    units: str
    ci95: tuple[float, float] | None = None


class FitResult(ConradModel):
    parameters: tuple[FittedParameter, ...]
    residual_rms: float
    n_residuals: int
    dof: int
    converged: bool
    message: str
    condition_number: float | None
    correlation: tuple[tuple[float, ...], ...] | None = None

    def value(self, name: str) -> float:
        return next(p.value for p in self.parameters if p.name == name)

    def get(self, name: str) -> FittedParameter:
        return next(p for p in self.parameters if p.name == name)


class IdentificationError(RuntimeError):
    pass


def fit(
    residual: Residual,
    x0: Sequence[float],
    names: Sequence[str],
    units: Sequence[str],
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
    x_scale: Sequence[float] | None = None,
) -> FitResult:
    x0a = np.asarray(x0, dtype=np.float64)
    p = x0a.size
    if not (len(names) == len(units) == p):
        raise ValueError("names/units/x0 lengths differ")
    lo = np.full(p, -np.inf) if lower is None else np.asarray(lower, dtype=np.float64)
    hi = np.full(p, np.inf) if upper is None else np.asarray(upper, dtype=np.float64)
    res = least_squares(
        residual,
        x0a,
        bounds=(lo, hi),
        x_scale="jac" if x_scale is None else np.asarray(x_scale),
        method="trf",
    )
    n = int(res.fun.size)
    dof = n - p
    if dof <= 0:
        raise IdentificationError(f"{n} residuals cannot identify {p} parameters")
    ssr = float(res.fun @ res.fun)
    s2 = ssr / dof
    jtj = res.jac.T @ res.jac
    sigmas: list[float | None] = [None] * p
    cond: float | None = None
    corr = None
    try:
        cond = float(np.linalg.cond(jtj))
        if np.isfinite(cond) and cond < 1e12:
            cov = s2 * np.linalg.inv(jtj)
            diag = np.clip(np.diag(cov), 0.0, None)
            sigmas = [float(np.sqrt(d)) for d in diag]
            denom = np.sqrt(np.outer(diag, diag))
            with np.errstate(divide="ignore", invalid="ignore"):
                c = np.where(denom > 0, cov / denom, 0.0)
            corr = tuple(tuple(float(v) for v in row) for row in c)
    except np.linalg.LinAlgError:
        cond = None
    params = tuple(
        FittedParameter(
            name=nm,
            value=float(v),
            sigma=sg,
            units=u,
            ci95=None if sg is None else (float(v) - 1.96 * sg, float(v) + 1.96 * sg),
        )
        for nm, v, sg, u in zip(names, res.x, sigmas, units, strict=True)
    )
    return FitResult(
        parameters=params,
        residual_rms=float(np.sqrt(ssr / n)),
        n_residuals=n,
        dof=dof,
        converged=bool(res.success),
        message=str(res.message),
        condition_number=cond,
        correlation=corr,
    )
