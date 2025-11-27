import numpy as np
import scipy.io as sio

# ----------------------------
# Parameters
# ----------------------------
g = 9.81
L = 1.0
T = 0.5
Nx = 256
Nt = 2000

dx = L / Nx
dt = T / Nt

x = np.linspace(0.0, L, Nx)
t = np.linspace(0.0, T, Nt)

# ----------------------------
# Fluxes
# ----------------------------
def flux(h, u):
    """
    Physical flux:
    F1 = hu
    F2 = hu^2 + 1/2 g h^2
    """
    q = h * u
    F1 = q
    F2 = q * u + 0.5 * g * h**2
    return F1, F2


def hll_flux(hL, uL, hR, uR):
    """
    HLL numerical flux for the shallow-water system.
    """
    qL = hL * uL
    qR = hR * uR

    cL = np.sqrt(g * hL)
    cR = np.sqrt(g * hR)

    sL = np.minimum(uL - cL, uR - cR)
    sR = np.maximum(uL + cL, uR + cR)

    FL1, FL2 = flux(hL, uL)
    FR1, FR2 = flux(hR, uR)

    F1 = np.where(
        sL >= 0, FL1,
        np.where(
            sR <= 0, FR1,
            (sR * FL1 - sL * FR1 + sL * sR * (hR - hL)) / (sR - sL)
        )
    )

    F2 = np.where(
        sL >= 0, FL2,
        np.where(
            sR <= 0, FR2,
            (sR * FL2 - sL * FR2 + sL * sR * (qR - qL)) / (sR - sL)
        )
    )

    return F1, F2


# ----------------------------
# Time integration (SSP-RK2)
# ----------------------------
def simulate_sv(h0, u0):
    """
    Evolve Saint-Venant equations from initial data (h0,u0)
    using HLL + SSP-RK2, fixed dt, dx, Nt, Nx.
    Returns h(t,x), u(t,x) of shape (Nt, Nx).
    """
    h = np.zeros((Nt, Nx))
    u = np.zeros((Nt, Nx))

    h[0, :] = h0
    u[0, :] = u0

    for n in range(Nt - 1):
        h_n = h[n, :]
        u_n = u[n, :]
        q_n = h_n * u_n

        # First stage
        hL = h_n[:-1]
        hR = h_n[1:]
        uL = u_n[:-1]
        uR = u_n[1:]

        F1, F2 = hll_flux(hL, uL, hR, uR)

        dF1 = np.zeros_like(h_n)
        dF2 = np.zeros_like(q_n)
        dF1[1:-1] = (F1[:-1] - F1[1:]) / dx
        dF2[1:-1] = (F2[:-1] - F2[1:]) / dx

        h1 = h_n + dt * dF1
        q1 = q_n + dt * dF2
        u1 = q1 / (h1 + 1e-12)

        # Second stage
        hL = h1[:-1]
        hR = h1[1:]
        uL = u1[:-1]
        uR = u1[1:]

        F1, F2 = hll_flux(hL, uL, hR, uR)
        dF1[1:-1] = (F1[:-1] - F1[1:]) / dx
        dF2[1:-1] = (F2[:-1] - F2[1:]) / dx

        h2 = h1 + dt * dF1
        q2 = q1 + dt * dF2
        u2 = q2 / (h2 + 1e-12)

        # Update
        h[n + 1, :] = 0.5 * (h_n + h2)
        u[n + 1, :] = 0.5 * (u_n + u2)

    return h, u


# ----------------------------
# Initial conditions
# ----------------------------
def ic_dam_break():
    """
    Classic dam-break: left depth 2, right depth 1, zero velocity.
    """
    h0_left = 2.0
    h0_right = 1.0
    x0 = 0.5 * L
    h0 = np.where(x < x0, h0_left, h0_right)
    u0 = np.zeros_like(x)
    return h0, u0


def ic_smooth_sine():
    """
    Smooth periodic perturbation on a background flow.
    """
    h_bg = 1.5
    amp_h = 0.3
    amp_u = 0.5
    k = 2.0 * np.pi / L

    h0 = h_bg + amp_h * np.sin(k * x)
    u0 = amp_u * np.sin(k * x)
    return h0, u0


def ic_gaussian_bump():
    """
    Localized Gaussian bump in depth on rest flow.
    """
    h_bg = 1.0
    amp = 0.8
    center = 0.5 * L
    width = 0.08

    h0 = h_bg + amp * np.exp(-((x - center) ** 2) / (2.0 * width ** 2))
    u0 = np.zeros_like(x)
    return h0, u0


# ----------------------------
# Generate and save datasets
# ----------------------------
if __name__ == "__main__":
    # Dam break
    h0, u0 = ic_dam_break()
    h_db, u_db = simulate_sv(h0, u0)
    sio.savemat("SV_dambreak.mat", {"x": x, "t": t, "h": h_db, "u": u_db})

    # Smooth sine wave
    h0, u0 = ic_smooth_sine()
    h_sine, u_sine = simulate_sv(h0, u0)
    sio.savemat("SV_sine.mat", {"x": x, "t": t, "h": h_sine, "u": u_sine})

    # Gaussian bump
    h0, u0 = ic_gaussian_bump()
    h_gauss, u_gauss = simulate_sv(h0, u0)
    sio.savemat("SV_gaussian.mat", {"x": x, "t": t, "h": h_gauss, "u": u_gauss})

    print("Saved SV_dambreak.mat, SV_sine.mat, SV_gaussian.mat")
