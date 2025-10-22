import numpy as np
import matplotlib.pyplot as plt
from Particles import kB, HarmonicParticleList, HardParticleList


def gaussian(x, mu, sigma2):
    return 1.0 / np.sqrt(2 * np.pi * sigma2) * np.exp(-((x - mu) ** 2) / (2 * sigma2))


def mb_pdf_speed(v, m, T):
    beta = m / (2 * kB * T)
    return 2 * beta * v * np.exp(-beta * v**2)


def harmonic_equilibrium_checks(
    N=400,
    m=1.0,
    kx=1.0,
    ky=1.0,
    Lx=3.0,
    Ly=3.0,
    T_bath=1.0,
    p_specular=0.4,
    seed=1,
    dt=0.01,
    steps=40000,
    sample_every=40,
    burn_in=8000,
):
    rng = np.random.default_rng(seed)

    x0, y0 = 0.5 * Lx, 0.5 * Ly

    parts = []
    x = x0 * np.ones(N) + rng.normal(0, 1, size=N)
    y = y0 * np.ones(N) + rng.normal(0, 1, size=N)
    r = np.column_stack((x, y))

    v = rng.normal(0, np.sqrt(kB * T_bath / m), size=(N, 2))
    parts = HarmonicParticleList(
        r,
        v,
        kx=kx,
        ky=ky,
        x0=x0,
        y0=y0,
        m=m,
        Lx=Lx,
        Ly=Ly,
        T=T_bath,
        p_specular=p_specular,
        seed=int(seed + 10),
    )

    for _ in range(burn_in):
        parts.step(dt)

    xs, vxs, Ks, Vs = np.array([]), np.array([]), np.array([]), np.array([])
    for s in range(steps):
        parts.step(dt)
        if s % sample_every == 0:
            velocities = parts.get_velocity_array()
            positions = parts.get_position_array()

            xs = np.append(xs, positions[:, 0] - x0)
            vxs = np.append(vxs, velocities[:, 0])
            Ks = np.append(
                Ks, np.mean(0.5 * m * (velocities[:, 0] ** 2 + velocities[:, 1] ** 2))
            )
            Vs = np.append(
                Vs,
                np.mean(
                    0.5 * kx * (positions[:, 0] - x0) ** 2
                    + 0.5 * ky * (positions[:, 1] - y0) ** 2
                ),
            )

    Kavg, Vavg = float(np.mean(Ks)), float(np.mean(Vs))

    # --- plots (unchanged) ---
    plt.figure()
    n, b, _ = plt.hist(xs, bins=40, density=True, label="sim p(x-x0)")
    centers = 0.5 * (b[:-1] + b[1:])
    plt.plot(
        centers, gaussian(centers, 0.0, kB * T_bath / kx), label="theory N(0, kT/kx)"
    )
    plt.xlabel("x")
    plt.ylabel("pdf")
    plt.title("Harmonic trap: x-marginal")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    n, b, _ = plt.hist(vxs, bins=40, density=True, label="sim p(vx)")
    centers = 0.5 * (b[:-1] + b[1:])
    plt.plot(
        centers, gaussian(centers, 0.0, kB * T_bath / m), label="theory N(0, kT/m)"
    )
    plt.xlabel("v_x")
    plt.ylabel("pdf")
    plt.title("Velocities at equilibrium")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(Ks, label="<K>/particle")
    plt.plot(Vs, label="<V>/particle")
    plt.axhline(
        kB * T_bath, linestyle="--", color="tab:gray", label="kT (expected <K> and <V>)"
    )
    plt.xlabel("sample index")
    plt.ylabel("energy")
    plt.title(f"Equipartition: <K>≈<V> (avg K={Kavg:.3f}, V={Vavg:.3f})")
    plt.legend()
    plt.tight_layout()


def mb_speed_histogram(
    N=200,
    radius=0.02,
    box=1.0,
    T_bath=1.0,
    p_specular=0.4,
    seed=3,
    dt=0.01,
    steps=40000,
    sample_every=40,
):
    rng = np.random.default_rng(seed)
    
    r = rng.random(size = (N,2))*box
    v = rng.normal(0, 0.5, size = (N,2))
    
    parts = HardParticleList(r, v,radius=radius,
                m=1.0,
                Lx=box,
                Ly=box,
                T=T_bath,
                p_specular=p_specular,
                seed=int(seed + 200))
        
    speeds = []
    for s in range(steps):
        parts.step(dt)
        if s % sample_every == 0:
            velocities = parts.get_velocity_array()
            speed_sample = (velocities[:,0]**2+velocities[:,1]**2)**0.5
            speeds = np.append(speeds, speed_sample)
    plt.figure()
    n, b, _ = plt.hist(speeds, bins=50, density=True, label="sim p(v)")
    vgrid = np.linspace(0, speeds.max(), 200)
    plt.plot(vgrid, mb_pdf_speed(vgrid, m=1.0, T=T_bath), label="MB theory (2D)")
    plt.xlabel("speed")
    plt.ylabel("pdf")
    plt.title("Thermal walls → Maxwell-Boltzmann")
    plt.legend()
    plt.tight_layout()


def pressure_temperature(
    N=200,
    box=1.0,
    T_list=(0.5, 1.0, 1.5, 2.0),
    p_specular=0.4,
    dt=0.01,
    steps=40000,
    burn_in=5000,
    seed=7,
):
    rng = np.random.default_rng(seed)
    perim = 2 * box
    Ps, P_theory = [], []

    for T in T_list:
        rng = np.random.default_rng(seed)
    
        r = rng.random(size = (N,2))*box
        v = rng.normal(0, 0.5, size = (N,2))
    
        parts = HardParticleList(r, v,
                    m=1.0,
                    Lx=box,
                    Ly=box,
                    T=T,
                    p_specular=p_specular,
                    seed=int(seed + 200))

        for _ in range(burn_in):
            parts.step(dt)

        dp_accum = 0.0
        for _ in range(steps):
            dp, temp = parts.step(dt)
            dp_accum += dp
        time_elapsed = steps * dt
        P_meas = dp_accum / (perim * time_elapsed)
        Ps.append(P_meas)

        # 2D ideal gas: P*A = N kT  =>  P = N kT / A,  A=box^2
        P_theory.append(N * T / (box * box))

    T_arr = np.array(T_list)
    plt.figure()
    plt.plot(T_arr, Ps, "o-", label="measured P")
    plt.plot(T_arr, P_theory, "--", label="theory N kT / A")
    plt.xlabel("T")
    plt.ylabel("Pressure")
    plt.legend()
    plt.tight_layout()
    plt.title("Ideal-gas pressure from wall momentum flux")
