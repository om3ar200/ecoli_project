"""Independent checks: stationarity, closed form, Routh-Hurwitz, ODE convergence."""
import numpy as np
from models import *
rng = np.random.default_rng(1); bad = 0; mx = 0
for _ in range(5000):
    p = {k: 10 ** rng.uniform(-1.5, 1.2) for k in DEFAULTS}
    b, u, v = equilibrium("m2b", p)
    res = np.abs(rhs("m2b", 0, [b, u, v], p)).max(); mx = max(mx, res)
    s = b / (p["h"] + b); d, e, g = p["delta"], p["eta"], p["gamma"]; r = p["rho"]
    A, C = r * b, d + e * s; E = u * p["h"] / (p["h"] + b) ** 2
    a1, a2, a3 = A + C + 1, A * C + A + C + g * b * E, A * C + g * b * d * E
    ev = jacobian_eigs("m2b", p).real.max()
    if not (a1 > 0 and a3 > 0 and a1 * a2 - a3 > 0 and ev < 0 and 0 < b < 1): bad += 1
print("random parameter sets: 5000, violations of RH/stability:", bad, " max |rhs(eq)|:", f"{mx:.1e}")
# ODE converges to equilibrium from many initial conditions
worst = 0
for _ in range(50):
    y0 = (rng.uniform(.01, 1), rng.uniform(0, 2), rng.uniform(0, 1.5))
    _, y = simulate("m2b", DEFAULTS, y0, T=200); worst = max(worst, np.abs(y[:, -1] - equilibrium("m2b")).max())
print("50 random initial conditions, worst |y(T)-eq|:", f"{worst:.1e}")
