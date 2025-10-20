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


class HarmonicParticle(Particle):
    """
    Particle with central-force in a 2D harmonic trap:
        V(x,y) = 1/2 kx x^2 + 1/2 ky y^2
    """
    def __init__(self, r, v, kx=1.0, ky=1.0, **kwargs):
        super().__init__(r, v, **kwargs)
        self.kx = float(kx)
        self.ky = float(ky)
        self.omega_x = np.sqrt(self.kx / self.m)
        self.omega_y = np.sqrt(self.ky / self.m)


    def acceleration(self, r=None):
        if r is None:
            r = self.r
        ax = -(self.kx / self.m) * r[0]
        ay = -(self.ky / self.m) * r[1]
        return np.array([ax, ay])


    # ---------- exact solution for error analysis ----------
    def analytic_state(self, t, r0=None, v0=None):
        if r0 is None: r0 = self.r.copy()
        if v0 is None: v0 = self.v.copy()

        ox, oy = self.omega_x, self.omega_y
        # x-channel
        cx, sx = np.cos(ox*t), np.sin(ox*t)
        x = r0[0]*cx + (v0[0]/ox)*sx
        vx = -r0[0]*ox*sx + v0[0]*cx
        # y-channel
        cy, sy = np.cos(oy*t), np.sin(oy*t)
        y = r0[1]*cy + (v0[1]/oy)*sy
        vy = -r0[1]*oy*sy + v0[1]*cy

        return np.array([x, y]), np.array([vx, vy])


    def kinetic_energy(self):
        return 0.5 * self.m * (self.v[0]**2 + self.v[1]**2)
    

    def potential_energy(self):
        return 0.5 * (self.kx * self.r[0]**2 + self.ky * self.r[1]**2)
    

class HardParticle(Particle):
    """
    No central force. Particles collide elastically if closer than radius.
    """
    def __init__(self, r, v, radius=0.02, **kwargs):
        super().__init__(r, v, **kwargs)
        self.radius = float(radius)


    def particle_interact(self, other):
        """
        Elastic 2-body collision
        """
        dr = self.r - other.r
        dist = np.linalg.norm(dr)
        min_dist = self.radius + getattr(other, "radius", self.radius)

        if dist == 0:
            dr = np.array([1e-12, 0])
            dist = 1e-12

        if dist >= min_dist:
            return

        # Check approaching (relative velocity toward each other)
        n = dr / dist
        dv = self.v - other.v
        rel_speed_n = np.dot(dv, n)
        if rel_speed_n >= 0:
            return  # particles are separating

        m1, m2 = self.m, other.m

        # 2D elastic collision along normal direction
        J = (2.0 * rel_speed_n) / (1/m1 + 1/m2)
        self.v -= (J / m1) * n
        other.v += (J / m2) * n

        # set dist to min_dist after collision to avoid overlap
        overlap = min_dist - dist
        if overlap > 0:
            # split by weights according to mass
            w1 = m2 / (m1 + m2)
            w2 = m1 / (m1 + m2)
            self.r += w1 * overlap * n
            other.r -= w2 * overlap * n

