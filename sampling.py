import numpy as np

kB = 1.0 # Boltzmann constant in reduced units

def inverse_cdf_speed(u, m=1.0, T=1.0):
    """
    In 2D, the CDF of MB velocity distribution is F(v) = 1 - exp(-v^2/(2*sigma^2)). 
    Inverse: v = sigma*sqrt(-2 ln(1-u)).
    """
    sigma = np.sqrt(kB*T/m)
    return sigma * np.sqrt(-2.0*np.log(1.0 - u))

def sample_velocity(T, m, rng, N_particles):
    """
    Sample a velocity from the Maxwell-Boltzmann distribution at temperature T for a particle of given mass.
    """
    
    sigma = np.sqrt(kB * T / m)
    uniform_random_speed = rng.uniform(size = N_particles)
    uniform_random_angle = rng.uniform(size = N_particles)*np.pi
    speed = inverse_cdf_speed(uniform_random_speed, m = m, T = T)
    vx = speed*np.cos(uniform_random_angle)
    vy = speed*np.sin(uniform_random_angle)
    return vx, vy

def rejection_sample_gauss(rng, n_samples):
    """
    Sample theta in [0, pi/2] with density cos(theta).
    """
    out = []
    while len(out) < n_samples:
        th = rng.uniform(0.0, 0.5*np.pi)
        y = rng.uniform(0.0, 1.0)
        if y <= np.cos(th):
            out.append(th)
    return np.array(out)
