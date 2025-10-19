import numpy as np


kB = 1.0 # Boltzmann constant in reduced units

def sample_velocity(T, m, rng):
    """
    Sample a velocity from the Maxwell-Boltzmann distribution at temperature T for a particle of given mass.
    """
    sigma = np.sqrt(kB * T / m)
    velocity = np.random.normal(0, sigma, size=2) # 2D velocity
    return velocity


class Particle:
    def __init__(self, r, v, m=1.0, Lx=1.0, Ly=1.0, T=1.0, p_specular=0.6, seed=None):
        self.r = np.array(r, dtype=np.float64)  # position
        self.v = np.array(v, dtype=np.float64)  # velocity
        self.m = m  # mass
        self.Lx = Lx  # box length in x
        self.Ly = Ly  # box length in y
        self.T = T  # temperature
        self.p_specular = p_specular  # probability of specular reflection
        self.rng = np.random.default_rng(seed)  # random number generator

    
    # -------- Utility functions --------
    def get_position(self):
        return self.r.copy()


    def set_position(self, r_new):
        self.r[:] = r_new


    def get_velocity(self):
        return self.v.copy()
    

    def set_velocity(self, v_new):
        self.v[:] = v_new


    def speed(self):
        return np.sqrt(self.v[0] ** 2 + self.v[1] ** 2)


    # -------- Motion --------
    def acceleration(self, r=None):
        """Hook for subclasses to implement specific forces. Default is no force."""
        return np.array([0.0, 0.0], dtype=np.float64)
    

    def move(self, dt):
        """Evolve particle position and velocity by time step dt using Symplectic Euler method."""
        a = self.acceleration(self.r)
        self.v += dt * a
        self.r += dt * self.v


    # -------- Wall interaction --------
    def wall_interact(self):
        """
        Handle wall collisions:
        - specular reflection with probability p_specular
        - diffuse thermal reflection (Maxwell-Boltzmann) otherwise

        Returns:
        dp_sum: total momentum transfer magnitude (for pressure)
        n_collisions: number of wall hits this step
        """
        dp_sum = 0.0
        n_collisions = 0
        x, y = self.r
        vx, vy = self.v
        m = self.m
        p = self.p_specular
        T = self.T

        # Left wall
        if x < 0.0:
            n_collisions += 1
            if self.rng.random() < p:
                vx = -vx
            else:
                vx, vy = sample_velocity(T, m, self.rng)
                vx = abs(vx)
            dp_sum += 2 * m * abs(vx)
            x = 0.0

        # Right wall
        if x > self.Lx:
            n_collisions += 1
            if self.rng.random() < p:
                vx = -vx
            else:
                vx, vy = sample_velocity(T, m, self.rng)
                vx = -abs(vx)
            dp_sum += 2 * m * abs(vx)
            x = self.Lx

        # Bottom wall
        if y < 0.0:
            n_collisions += 1
            if self.rng.random() < p:
                vy = -vy
            else:
                vx, vy = sample_velocity(T, m, self.rng)
                vy = abs(vy)
            dp_sum += 2 * m * abs(vy)
            y = 0.0

        # Top wall
        if y > self.Ly:
            n_collisions += 1
            if self.rng.random() < p:
                vy = -vy
            else:
                vx, vy = sample_velocity(T, m, self.rng)
                vy = -abs(vy)
            dp_sum += 2 * m * abs(vy)
            y = self.Ly

        self.r[:] = [x, y]
        self.v[:] = [vx, vy]
        return dp_sum, n_collisions
