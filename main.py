import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


def initialize_simulation(
    particle, N_particles, t_max, speed_distribution, dt=0.01, box_size=10.0
):
    positions_x = np.random.uniform(low=0, high=box_size, size=N_particles)
    positions_y = np.random.uniform(low=0, high=box_size, size=N_particles)

    velocity_angles = np.random.uniform(low=0, high=2 * np.pi, size=N_particles)
    speeds = speed_distribution(N_particles)
    velocities_x = np.cos(velocity_angles) * speeds
    velocities_y = np.sin(velocity_angles) * speeds

    particle_list = np.array(
        [
            particle_class(
                positions_x[i], positions_y[i], velocities_x[i], velocities_y[i]
            )
            for i in range(N_particles)
        ]
    )
    return particle_list
