import numpy as np
from particle import kB

def inverse_cdf_speed(u, m=1.0, T=1.0):
    """
    In 2D, the CDF of MB velocity distribution is F(v) = 1 - exp(-v^2/(2*sigma^2)). 
    Inverse: v = sigma*sqrt(-2 ln(1-u)).
    """
    sigma = np.sqrt(kB*T/m)
    return sigma * np.sqrt(-2.0*np.log(1.0 - u))


def rejection_sample_cos_theta(n_samples, rng):
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
