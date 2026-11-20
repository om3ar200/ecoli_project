"""Dimensionless E. coli - complement models (Model 1, 2A, 2B).
State y = (b, u, v). Parameters: rho, gamma, delta, eta, h.
All parameter values used here are ILLUSTRATIVE, not experimentally calibrated."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

DEFAULTS = dict(rho=1.0, gamma=4.0, delta=0.5, eta=1.0, h=0.1)
Y0 = (0.05, 1.0, 0.0)
NAMES = {"m1": "Model 1 (original)", "m2a": "Model 2A (mass balance)",
         "m2b": "Model 2B (infection-dependent)"}

def rhs(kind, t, y, p):
    b, u, v = y
    g = p["rho"] * b * (1 - b) - p["gamma"] * b * v
    if kind == "m1":
        return [g, p["delta"] * (1 - u), u - v]
    if kind == "m2a":
        return [g, p["delta"] - (p["delta"] + p["eta"]) * u, u - v]
    s = b / (p["h"] + b)
    # CORRECTED: with v scaled by alpha*Lambda/(mu1*mu2), v' = u*s - v (no eta factor)
    return [g, p["delta"] * (1 - u) - p["eta"] * u * s, u * s - v]

def simulate(kind, p=DEFAULTS, y0=Y0, T=40, n=801, rtol=1e-10, atol=1e-12):
    t = np.linspace(0, T, n)
    s = solve_ivp(lambda t, y: rhs(kind, t, y, p), (0, T), y0, t_eval=t,
                  method="LSODA", rtol=rtol, atol=atol)
    return s.t, s.y

def equilibrium(kind, p=DEFAULTS):
    """Returns (b*, u*, v*) of the biologically relevant branch."""
    r, g, d, e, h = (p[k] for k in ("rho", "gamma", "delta", "eta", "h"))
    if kind == "m1":
        return max(0, 1 - g / r), 1.0, 1.0
    if kind == "m2a":
        u = d / (d + e)
        return max(0, 1 - g * u / r), u, u
    # Closed form: with Gamma=gamma/rho, kappa=1+eta/delta, b* solves
    #   kappa b^2 + (h + Gamma - kappa) b - h = 0   (positive root)
    G, k = g / r, 1 + e / d
    B = k - h - G
    b = (B + np.sqrt(B * B + 4 * k * h)) / (2 * k)
    s = b / (h + b); u = d / (d + e * s)
    return b, u, u * s

def threshold(kind, p=DEFAULTS):
    """Invasion number R0 for m1/m2a (DFE stable iff R0<1). 2B: DFE always unstable."""
    r, g, d, e = p["rho"], p["gamma"], p["delta"], p["eta"]
    if kind == "m1": return r / g
    if kind == "m2a": return r * (d + e) / (g * d)
    return float("inf")

def jacobian_eigs(kind, p=DEFAULTS, eq=None):
    b, u, v = eq or equilibrium(kind, p)
    r, g, d, e, h = (p[k] for k in ("rho", "gamma", "delta", "eta", "h"))
    if kind == "m1":
        J = [[r*(1-2*b)-g*v, 0, -g*b], [0, -d, 0], [0, 1, -1]]
    elif kind == "m2a":
        J = [[r*(1-2*b)-g*v, 0, -g*b], [0, -(d+e), 0], [0, 1, -1]]
    else:
        s = b/(h+b); ds = h/(h+b)**2
        J = [[r*(1-2*b)-g*v, 0, -g*b],
             [-e*u*ds, -d-e*s, 0], [u*ds, s, -1]]
    return np.linalg.eigvals(np.array(J, float))

# ---- Dimensional parameters -> dimensionless groups -------------------------
def to_dimensionless(r, K, beta, Lam, mu1, mu2, alpha, H):
    """rho=r/mu2, gamma=beta*alpha*Lam/(mu1*mu2^2), delta=mu1/mu2, eta=alpha/mu2, h=H/K."""
    return dict(rho=r/mu2, gamma=beta*alpha*Lam/(mu1*mu2**2), delta=mu1/mu2,
                eta=alpha/mu2, h=H/K)

DIM = dict(r=1.0, K=1.0, beta=1.0, Lam=1.0, mu1=1.0, mu2=1.0, alpha=1.0, H=1.0)
def dim_default():
    """A dimensional parameter set that reproduces DEFAULTS (illustrative only)."""
    mu2 = 1.0; d = DEFAULTS
    mu1 = d["delta"]*mu2; alpha = d["eta"]*mu2; r = d["rho"]*mu2; K = 1.0; H = d["h"]*K
    Lam = 1.0; beta = d["gamma"]*mu1*mu2**2/(alpha*Lam)
    return dict(r=r, K=K, beta=beta, Lam=Lam, mu1=mu1, mu2=mu2, alpha=alpha, H=H)

def dim_elasticities(kind="m2b", dp=None, eps=1e-5):
    dp = dp or dim_default(); out = {}
    base = equilibrium(kind, to_dimensionless(**dp))[0]
    for k in dp:
        up, dn = dict(dp), dict(dp); up[k] *= 1+eps; dn[k] *= 1-eps
        bp = equilibrium(kind, to_dimensionless(**up))[0]; bm = equilibrium(kind, to_dimensionless(**dn))[0]
        # b = B/K, so the elasticity of B* itself adds +1 for K
        e_b = (bp-bm)/(2*eps)/base
        out[k] = e_b + (1.0 if k == "K" else 0.0)
    return out
