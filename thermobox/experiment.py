import numpy as np
import matplotlib.pyplot as plt
from particle import kB, HarmonicParticle, HardParticle

def gaussian(x, mu, sigma2):
    return 1.0/np.sqrt(2*np.pi*sigma2) * np.exp(-(x-mu)**2/(2*sigma2))

def mb_pdf_speed(v, m, T):
    beta = m/(2*kB*T)
    return 2*beta*v*np.exp(-beta*v**2)

def do_pair_collisions(particles):
    N = len(particles)
    for i in range(N):
        for j in range(i+1, N):
            particles[i].particle_interact(particles[j])


def harmonic_equilibrium_checks(
    N=400, m=1.0, kx=1.0, ky=1.0, Lx=3.0, Ly=3.0,
    T_bath=1.0, p_specular=0.4, seed=1, dt=0.01,
    steps=40000, sample_every=40, burn_in=8000              
):
    rng = np.random.default_rng(seed)

    x0, y0 = 0.5*Lx, 0.5*Ly

    parts = []
    for i in range(N):
        # positions start near trap center
        r = np.array([x0, y0]) + rng.normal(0, 1, 2) 
        v = rng.normal(0, np.sqrt(kB*T_bath/m), 2)
        parts.append(
            HarmonicParticle(
                r=r, v=v, kx=kx, ky=ky, x0=x0, y0=y0,
                m=m, Lx=Lx, Ly=Ly, T=T_bath, p_specular=p_specular,
                seed=int(seed+10+i)
            )
        )

    for _ in range(burn_in):
        for p in parts:
            p.step(dt)

    xs, vxs, Ks, Vs = [], [], [], []
    for s in range(steps):
        for p in parts:
            p.step(dt)
        if s % sample_every == 0:
            xs.extend([p.r[0] - x0 for p in parts])  # centered
            vxs.extend([p.v[0] for p in parts])
            Ks.append(np.mean([0.5*m*(p.v[0]**2+p.v[1]**2) for p in parts]))
            Vs.append(np.mean([0.5*kx*(p.r[0]-x0)**2 + 0.5*ky*(p.r[1]-y0)**2 for p in parts]))

    xs = np.asarray(xs); vxs = np.asarray(vxs)
    Kavg, Vavg = float(np.mean(Ks)), float(np.mean(Vs))

    # --- plots (unchanged) ---
    plt.figure()
    n,b,_ = plt.hist(xs, bins=40, density=True, label='sim p(x-x0)')
    centers = 0.5*(b[:-1]+b[1:])
    plt.plot(centers, gaussian(centers, 0.0, kB*T_bath/kx), label='theory N(0, kT/kx)')
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.title('Position (x) at equilibrium')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    n,b,_ = plt.hist(vxs, bins=40, density=True, label='sim p(vx)')
    centers = 0.5*(b[:-1]+b[1:])
    plt.plot(centers, gaussian(centers, 0.0, kB*T_bath/m), label='theory N(0, kT/m)')
    plt.xlabel(r'$v_x$')
    plt.ylabel('pdf')
    plt.title('Velocities at equilibrium')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(Ks, label='<K>/particle'); plt.plot(Vs, label='<V>/particle')
    plt.axhline(kB*T_bath, linestyle='--', color='tab:gray', label='kT (expected <K> and <V>)')
    plt.xlabel('sample index'); plt.ylabel('energy')
    plt.title(f'Equipartition: <K>≈<V> (avg K={Kavg:.3f}, V={Vavg:.3f})')
    plt.legend()
    plt.tight_layout()



def mb_speed_histogram(N=200, radius=0.005, box=1.0, T_bath=1.0, p_specular=0.4, seed=3, dt=0.01, steps=40000, sample_every=40):
    rng = np.random.default_rng(seed)
    parts=[]
    for i in range(N):
        r = rng.random(2)*box
        v = rng.normal(0, 0.5, 2)
        parts.append(HardParticle(r, v, radius=radius, m=1.0, Lx=box, Ly=box, T=T_bath, p_specular=p_specular, seed=int(seed+200+i)))
    speeds=[]
    for s in range(steps):
        for p in parts: p.step(dt)
        do_pair_collisions(parts)
        if s % sample_every == 0:
            speeds.extend([np.hypot(p.v[0], p.v[1]) for p in parts])
    speeds = np.asarray(speeds)
    plt.figure()
    n,b,_=plt.hist(speeds, bins=50, density=True, label='sim p(v)')
    vgrid = np.linspace(0, speeds.max(), 200)
    plt.plot(vgrid, mb_pdf_speed(vgrid, m=1.0, T=T_bath), label='MB theory (2D)')
    plt.xlabel('speed')
    plt.ylabel('pdf')
    plt.title('Thermal walls → Maxwell-Boltzmann')
    plt.legend()
    plt.tight_layout()


def pressure_temperature(
    N=200, box=1.0, T_list=(0.5, 1.0, 1.5, 2.0), p_specular=0.4,
    dt=0.01, steps=40000, burn_in=5000, seed=7
):
    rng = np.random.default_rng(seed)
    perim = 2*box 
    Ps, P_theory = [], []

    for T in T_list:
        parts=[]
        for i in range(N):
            r = rng.random(2)*box
            v = rng.normal(0, 0.5, 2)
            parts.append(HardParticle(r, v, m=1.0, Lx=box, Ly=box, T=T, p_specular=p_specular, seed=int(seed+200+i)))

        for _ in range(burn_in):
            for p in parts: p.step(dt)

        dp_accum = 0.0
        for _ in range(steps):
            for p in parts:
                dp, _ = p.step(dt)
                dp_accum += dp
        time_elapsed = steps*dt
        P_meas = dp_accum / (perim * time_elapsed)
        Ps.append(P_meas)

        # 2D ideal gas: P*A = N kT  =>  P = N kT / A,  A=box^2
        P_theory.append(N*T/(box*box))

    T_arr = np.array(T_list)
    plt.figure()
    plt.plot(T_arr, Ps, 'o-', label='measured P')
    plt.plot(T_arr, P_theory, '--', label='theory N kT / A')
    plt.xlabel('T')
    plt.ylabel('Pressure')
    plt.legend()
    plt.tight_layout()
    plt.title('Ideal-gas pressure from wall momentum flux')
