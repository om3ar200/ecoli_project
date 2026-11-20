import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import *
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150})
C = {"m1": "#2563eb", "m2a": "#0f9d8a", "m2b": "#d97706"}
OUT = "figures/"; import os; os.makedirs(OUT, exist_ok=True)

# 1. Numerical validation: tight reference vs RK45 at several tolerances + equilibrium convergence
print("== Validation ==")
for k in NAMES:
    _, ref = simulate(k, rtol=1e-12, atol=1e-14)
    for tol in (1e-4, 1e-6, 1e-8):
        _, y = simulate(k, rtol=tol, atol=tol*1e-2)
        print(f"{k} rtol={tol:g}  max err={np.abs(y-ref).max():.2e}")
    eq = np.array(equilibrium(k)); print(f"{k} |y(T)-eq*|={np.abs(ref[:,-1]-eq).max():.2e}")

# 2. Time series comparison
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for k in NAMES:
    t, y = simulate(k)
    for i, lab in enumerate(["b (bacteria)", "u (inactive)", "v (active)"]):
        ax[i].plot(t, y[i], color=C[k], lw=2, label=NAMES[k]); ax[i].set_title(lab); ax[i].set_xlabel(r"$\tau$")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT+"fig1_timeseries.png"); plt.close()

# 3. Phase plane b-v
fig, ax = plt.subplots(figsize=(6, 5))
for k in NAMES:
    t, y = simulate(k); ax.plot(y[0], y[2], color=C[k], lw=2, label=NAMES[k]); ax.plot(y[0, 0], y[2, 0], "ko")
    eq = equilibrium(k); ax.plot(eq[0], eq[2], "*", color=C[k], ms=14)
ax.set_xlabel("b"); ax.set_ylabel("v"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT+"fig2_phase.png"); plt.close()

# 4. Bifurcation in gamma (killing strength): equilibrium b*
gs = np.linspace(0.05, 8, 400)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for k in NAMES:
    bs = [equilibrium(k, {**DEFAULTS, "gamma": g})[0] for g in gs]
    ax[0].plot(gs, bs, color=C[k], lw=2, label=NAMES[k])
ax[0].set_xlabel(r"$\gamma$ (killing strength)"); ax[0].set_ylabel("equilibrium $b^*$"); ax[0].legend(fontsize=8)
ax[0].set_title("Transcritical in M1/M2A; no extinction in M2B")
# threshold map in (gamma, rho)
G, R = np.meshgrid(np.linspace(.1, 8, 200), np.linspace(.1, 4, 200))
for k, ls in (("m1", "-"), ("m2a", "--")):
    Z = np.array([[threshold(k, {**DEFAULTS, "gamma": g, "rho": r}) for g in G[0]] for r in R[:, 0]])
    ax[1].contour(G, R, Z, levels=[1], colors=C[k], linestyles=ls, linewidths=2)
ax[1].plot([], [], color=C["m1"], label="M1: R0=1"); ax[1].plot([], [], color=C["m2a"], ls="--", label="M2A: R0=1")
ax[1].plot(DEFAULTS["gamma"], DEFAULTS["rho"], "k*", ms=12, label="default")
ax[1].set_xlabel(r"$\gamma$"); ax[1].set_ylabel(r"$\rho$"); ax[1].legend(); ax[1].set_title("Control region: above curve = infection persists")
plt.tight_layout(); plt.savefig(OUT+"fig3_bifurcation.png"); plt.close()

# 5. Normalised sensitivity of b* (central differences)
print("== Sensitivity of b* (Model 2B, and 2A) ==")
for k in ("m2a", "m2b"):
    base = equilibrium(k)[0]
    for par in DEFAULTS:
        pp, pm = dict(DEFAULTS), dict(DEFAULTS); e = 1e-5
        pp[par] *= 1+e; pm[par] *= 1-e
        s = (equilibrium(k, pp)[0]-equilibrium(k, pm)[0])/(2*e*DEFAULTS[par])*DEFAULTS[par]/max(base, 1e-12)
        print(k, par, round(s, 4) if base > 0 else "n/a (b*=0)")

# 6. Scenarios (illustrative)
print("== Scenarios, Model 2B final b ==")
for name, upd in {"baseline": {}, "weak complement (gamma=1)": {"gamma": 1},
                  "fast bacteria (rho=2)": {"rho": 2}, "eta=3 at fixed gamma (NOT a pure alpha change)": {"eta": 3}, "alpha x3 (eta=3, gamma=12)": {"eta": 3, "gamma": 12},
                  "late activation (h=0.5)": {"h": 0.5}}.items():
    p = {**DEFAULTS, **upd}; print(f"{name:28s} b*={equilibrium('m2b', p)[0]:.4f}  eigs={np.round(jacobian_eigs('m2b', p), 3)}")

# 7. Tornado: elasticity of B* to DIMENSIONAL parameters (Model 2B)
el = dim_elasticities(); lab = {"r":"r (bacterial growth)","K":"K (carrying capacity)","beta":"β (killing coeff.)","Lam":"Λ (P1 production)",
 "mu1":"μ1 (P1 removal)","mu2":"μ2 (P2 removal)","alpha":"α (activation rate)","H":"H (half-saturation)"}
ks = sorted(el, key=lambda k: abs(el[k]))
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh([lab[k] for k in ks], [el[k] for k in ks], color=["#d97706" if el[k] > 0 else "#2563eb" for k in ks])
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("elasticity  (∂B*/B*)/(∂p/p)"); ax.set_title("Model 2B: what controls the endemic bacterial load?")
plt.tight_layout(); plt.savefig(OUT+"fig4_tornado.png"); plt.close()

# 8. Closed-form check: b* = [(k-h-G)+sqrt((k-h-G)^2+4kh)]/(2k), G=gamma/rho, k=1+eta/delta
G, k, h = 4/1, 1+1/0.5, 0.1
print("closed form b* (defaults):", ((k-h-G)+np.sqrt((k-h-G)**2+4*k*h))/(2*k))
