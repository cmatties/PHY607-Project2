import numpy as np
from sampling import sample_velocity

kB = 1.0  # Boltzmann constant in reduced units
epsilon = 1e-12 # Definition of "very close" for the purposes of collisions

class ParticleList:
    def __init__(
        self,
        r_array,
        v_array,
        m=1.0,
        Lx=3.0,
        Ly=3.0,
        T=1.0,
        p_specular=0.6,
        seed=None,
    ):
        self.r_array = r_array  # position
        self.v_array = v_array # velocity
        self.m = m  # mass
        self.Lx = Lx  # box length in x
        self.Ly = Ly  # box length in y
        self.T = T  # temperature
        self.p_specular = p_specular  # probability of specular reflection
        self.rng = np.random.default_rng(seed)  # random number generator
        self.N = r_array.shape[0]

    # -------- Utility functions --------
    # View and modify the whole arrays of positions and velocities
    def get_position_array(self):
        return self.r_array.copy()

    def set_position_array(self, r_new):
        self.r_array = r_new

    def get_velocity_array(self):
        return self.v_array.copy()

    def set_velocity_array(self, v_new):
        self.v_array = v_new

    def get_speed_array(self):
        return np.sqrt(self.v[:, 0] ** 2 + self.v[:, 1] ** 2)

    # View and modify the position and velocity of a single particle
    def get_position_single(self, index):
        return self.r_array[index, :]

    def set_position_single(self, index, r_new):
        self.r_array[index, :] = r_new

    def get_velocity_single(self, index):
        return self.v_array[index, :]

    def set_velocity_single(self, index, v_new):
        self.v_array[index, :] = v_new

    def get_speed_single(self, index):
        return np.sqrt(self.v[index, 0] ** 2 + self.v[index, 1] ** 2)

    # -------- Motion --------
    def acceleration(self, r=None):
        """Hook for subclasses to implement specific forces. Default is no force."""
        return np.array([0.0, 0.0], dtype=np.float64)

    def move(self, dt):
        """Evolve particle position and velocity by time step dt using Symplectic Euler method."""
        a_array = self.acceleration(self.r_array)
        self.v_array += dt * a_array
        self.r_array += dt * self.v_array

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
        x, y = self.r_array.T
        vx, vy = self.v_array.T
        m = self.m
        p = self.p_specular
        T = self.T
        N_particles = self.N
        
        # Left wall
        diffused_vx, diffused_vy = sample_velocity(T, m, self.rng, N_particles)
        random = self.rng.uniform(size = N_particles)
        
        left_collisions = x < epsilon
        n_collisions += np.sum(left_collisions)
        
        vx_new = vx*(1-left_collisions) - vx*left_collisions*(random<p) + np.abs(diffused_vx*left_collisions*(random>=p))
        vy_new = vy*(1-left_collisions) + diffused_vy*left_collisions*(random>=p)
        
        dp_sum += np.sum(m*(vx_new-vx))
        
        vx = vx_new.copy()
        vy = vy_new.copy()
        
        x = x*(1-left_collisions)+epsilon*left_collisions

        # Right wall
        diffused_vx, diffused_vy = sample_velocity(T, m, self.rng, N_particles)
        diffused_vx *= -1
        random = self.rng.uniform(size = N_particles)
        
        right_collisions = x > self.Lx-epsilon
        n_collisions += np.sum(right_collisions)
        
        # Set new velocities based on whether a collisions has occurred, and whether that collision is specular or diffuse
        vx_new = vx*(1-right_collisions) - vx*right_collisions*(random<p) - np.abs(diffused_vx*right_collisions*(random>=p))
        vy_new = vy*(1-right_collisions) + diffused_vy*right_collisions*(random>=p)
        
        dp_sum += np.sum(np.abs(m*(vx_new-vx)))
        
        vx = vx_new.copy()
        vy = vy_new.copy()
        
        x = x*(1-right_collisions)+(self.Lx-epsilon)*right_collisions

        # Bottom wall
        diffused_vy, diffused_vx = sample_velocity(T, m, self.rng, N_particles)
        random = self.rng.uniform(size = N_particles)
        
        bot_collisions = x < epsilon
        n_collisions += np.sum(bot_collisions)
        
        # Set new velocities based on whether a collisions has occurred, and whether that collision is specular or diffuse
        vy_new = vy*(1-bot_collisions) - vy*bot_collisions*(random<p) + np.abs(diffused_vy*bot_collisions*(random>=p))
        vx_new = vx*(1-bot_collisions) + diffused_vx*bot_collisions*(random>=p)
        
        # Add to total momentum transferred to the wall
        dp_sum += np.sum(np.abs(m*(vy_new-vy)))
        
        vx = vx_new.copy()
        vy = vy_new.copy()
        
        y = y*(1-bot_collisions)+(epsilon)*bot_collisions

        # Top wall
        diffused_vy, diffused_vx = sample_velocity(T, m, self.rng, N_particles)
        diffused_vy *= -1
        random = self.rng.uniform(size = N_particles)
        
        top_collisions = x > (self.Ly - epsilon)
        n_collisions += np.sum(top_collisions)
        
        # Set new velocities based on whether a collisions has occurred, and whether that collision is specular or diffuse
        vy_new = vy*(1-top_collisions) - vy*top_collisions*(random<p) - np.abs(diffused_vy*top_collisions*(random>=p))
        vx_new = vx*(1-bot_collisions) + diffused_vx*bot_collisions*(random>=p)
        
        # Add to total momentum transferred to the wall
        dp_sum += np.sum(np.abs(m*(vy_new-vy)))
        
        vx = vx_new.copy()
        vy = vy_new.copy()
        
        y = y*(1-top_collisions)+(self.Lx - epsilon)*bot_collisions

        self.r_array = np.column_stack((x, y))
        self.v_array = np.column_stack((vx, vy))
        return dp_sum, n_collisions

    def particle_interact(self):
        return

    def step(self, dt):
        """Perform a full time step: move + wall interaction."""
        self.move(dt)
        self.particle_interact()
        return self.wall_interact()


class HarmonicParticleList(ParticleList):
    """
    Particle with central-force in a 2D harmonic trap:
        V(x,y) = 1/2 kx x^2 + 1/2 ky y^2
    """

    def __init__(self, r_array, v_array, kx=1.0, ky=1.0, x0=1.5, y0=1.5, **kwargs):
        super().__init__(r_array, v_array, **kwargs)
        self.kx = float(kx)
        self.ky = float(ky)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.omega_x = np.sqrt(self.kx / self.m)
        self.omega_y = np.sqrt(self.ky / self.m)

    def acceleration(self, r=None):
        if r is None:
            r = self.r_array
        dx = r[:,0] - self.x0
        dy = r[:,1] - self.y0
        ax = -(self.kx / self.m) * dx
        ay = -(self.ky / self.m) * dy
        return np.column_stack((ax, ay))

    # ---------- exact solution for error analysis ----------
    def analytic_state(self, t, r0=None, v0=None):
        if r0 is None:
            r0 = self.r_array.copy()
        if v0 is None:
            v0 = self.v_array.copy()

        ox, oy = self.omega_x, self.omega_y
        # x-channel
        cx, sx = np.cos(ox * t), np.sin(ox * t)
        x_array = r0[:, 0] * cx + (v0[:, 0] / ox) * sx
        vx_array = -r0[:, 0] * ox * sx + v0[:, 0] * cx
        # y-channel
        cy, sy = np.cos(oy * t), np.sin(oy * t)
        y_array = r0[:, 1] * cy + (v0[:, 1] / oy) * sy
        vy = -r0[:, 1] * oy * sy + v0[:, 1] * cy
        return np.column_stack((x_array, y_array)), np.column_stack(
            [vx_array, vy_array]
        )

    def kinetic_energy_array(self):
        return 0.5 * self.m * (self.v_array[:, 0] ** 2 + self.v_array[:, 1] ** 2)

    def potential_energy_array(self):
        return 0.5 * (
            self.kx * self.r_array[:, 0] ** 2 + self.ky * self.r_array[:, 1] ** 2
        )

    def kinetic_energy_single(self, index):
        return (
            0.5 * self.m * (self.v_array[index, 0] ** 2 + self.v_array[index, 1] ** 2)
        )

    def potential_energy_single(self, index):
        return 0.5 * (
            self.kx * self.r_array[index, 0] ** 2
            + self.ky * self.r_array[index, 1] ** 2
        )


class HardParticleList(ParticleList):
    """
    No central force. Particles collide elastically if closer than radius.
    """

    def __init__(self, r_array, v_array, radius=0.02, **kwargs):
        super().__init__(r_array, v_array, **kwargs)
        self.radius = float(radius)

    def particle_interact(self):
        """
        Elastic 2-body collision
        """
        dx = np.subtract.outer(self.r_array[:, 0], self.r_array[:, 0])
        dy = np.subtract.outer(self.r_array[:, 1], self.r_array[:, 1])
        dist = (dx**2 + dy**2) ** 0.5
        min_dist = (
            2 * self.radius
        )  # This only works if all of our particles are identical

        # If the distance between two particles is very small, set their relative displacement to [epsilon, 0]
        dx = dx * (dist > epsilon) + (dist <= epsilon) * epsilon
        dy = dy * (dist > epsilon)
        dist = dist * (dist > epsilon) + (dist <= epsilon) * epsilon

        # Get all the particle pairs which are overlapping
        close_particles = (dist <= min_dist) * (1 - np.diag(np.ones(len(self.v_array))))  

        # Check approaching (relative velocity toward each other)
        nx = dx / dist
        ny = dy / dist
        dvx = np.subtract.outer(self.v_array[:, 0], self.v_array[:, 0])
        dvy = np.subtract.outer(self.v_array[:, 1], self.v_array[:, 1])
        rel_speed_n = nx * dvx + ny * dvy

        # Get all the particle pairs which are getting closer to one another
        colliding_particles = rel_speed_n < 0

        m = self.m

        collisions = close_particles * colliding_particles

        dvx_final = np.sum(rel_speed_n * nx * collisions, axis=1)
        dvy_final = np.sum(rel_speed_n * ny * collisions, axis=1)

        # 2D elastic collision along normal direction
        self.v_array[:, 0] += dvx_final
        self.v_array[:, 1] += dvy_final

        # set dist to min_dist after collision to avoid overlap
        overlap = min_dist - dist
        dx_final = np.sum(0.5 * overlap * nx * collisions, axis=1)
        dy_final = np.sum(0.5 * overlap * ny * collisions, axis=1)
        self.r_array[:, 0] += dx_final
        self.r_array[:, 1] += dy_final
