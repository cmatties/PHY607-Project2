import numpy as np

kB = 1.0  # Boltzmann constant in reduced units


def inverse_cdf_speed(u, m=1.0, T=1.0):
    """
    Sample speeds from the Maxwell-Boltzmann distribution, with inverrse CDF.

    In 2D, the CDF of MB velocity distribution is F(v) = 1 - exp(-v^2/(2*sigma^2)).
    This function has inverse v = sigma*sqrt(-2 ln(1-u)).

    Parameters
    ----------
    u : ndarray
        Random number (or numbers) picked from the uniform distribution on [0,1]
    m : float, optional
        Mass of each particle
    T : float, optional
        Overall temperature of the distribution

    Returns
    -------
    ndarray
        Random numbers chosen from the Maxwell-Boltzmann distribution.
        Has the same shape as u.
    """
    sigma = np.sqrt(kB * T / m)
    return sigma * np.sqrt(-2.0 * np.log(1.0 - u))


def sample_velocity(T, m, rng, N_particles):
    """
    Sample velocities from the Maxwell-Boltzmann distribution at temperature T for a particle of given mass.

    Selects angles uniformly between 0 and pi. Used for computing diffuse reflections.

    Parameters
    ----------
    T : float
        Maxwell-Boltzmann distribution temperature
    m : float
        particle mass
    rng : numpy.random.Generator
        Random number generator
    N_particles : int
        Number of particles to generate velocities for

    Returns
    -------
    vx : numpy.ndarray
        X velocities. Shape (N,)
    vy : numpy.ndarray
        Y velocities. Shape (N,)
    """

    sigma = np.sqrt(kB * T / m)
    uniform_random_speed = rng.uniform(size=N_particles)
    uniform_random_angle = rng.uniform(size=N_particles) * np.pi
    speed = inverse_cdf_speed(uniform_random_speed, m=m, T=T)
    vx = speed * np.cos(uniform_random_angle)
    vy = speed * np.sin(uniform_random_angle)
    return vx, vy


def gaussian_rejection_sampling(mean, sigma, rng, size):
    """
    Sample from a Gaussian distribution with rejection sampling.

    Parameters
    ----------
    mean : float
        Mean of the distribution
    sigma : float
        Standard deviation of the distribution
    rng : numpy.random.Generator
        Random number generator
    size : tuple
        Shape of the returned array of random numbers

    Returns
    -------
    numpy.ndarray
        Array of random numbers sampled from a Gaussian distribution. Shape given by size.
    """
    gauss = (
        lambda x: 1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * (x - mean) ** 2 / sigma**2)
    )

    out = []
    N = np.prod(size)
    while len(out) < N:
        x = rng.uniform(mean - 5 * sigma, mean + 5 * sigma)
        y = rng.uniform(0.0, gauss(mean))
        if y <= gauss(x):
            out.append(x)
    out_array = np.array(out)
    return out_array.reshape(size)
