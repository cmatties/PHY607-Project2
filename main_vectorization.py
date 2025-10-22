import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
#from experiment import harmonic_equilibrium_checks, mb_speed_histogram, pressure_temperature
from experiment_vectorized import harmonic_equilibrium_checks, mb_speed_histogram, pressure_temperature


# def initialize_simulation(
#     particle, N_particles, t_max, speed_distribution, dt=0.01, box_size=10.0
# ):
#     positions_x = np.random.uniform(low=0, high=box_size, size=N_particles)
#     positions_y = np.random.uniform(low=0, high=box_size, size=N_particles)

#     velocity_angles = np.random.uniform(low=0, high=2 * np.pi, size=N_particles)
#     speeds = speed_distribution(N_particles)
#     velocities_x = np.cos(velocity_angles) * speeds
#     velocities_y = np.sin(velocity_angles) * speeds

#     particle_list = np.array(
#         [
#             particle_class(
#                 positions_x[i], positions_y[i], velocities_x[i], velocities_y[i]
#             )
#             for i in range(N_particles)
#         ]
#     )
#     return particle_list


# def evolve_timestep(particle_list, t, dt):
#     new_velocities = []
#     new_positions = []
#     for particle in particle_list:
#         v_new = particle.get_v_update(particle_list, t, dt)
#         r_new = v_new * dt

#         new_velocities.append(v_new)
#         new_positions.append(r_new)

#     wall_interactions = 0

#     for i in range(len(particle_list)):
#         particle_list[i].update_velocity(new_velocities[i])
#         particle_list[i].update_position(new_positions[i])
#         wall_interactions += particle_list[i].get_wall_interaction()
#     return particle_list


if __name__ == "__main__":
    harmonic_equilibrium_checks()
    # mb_speed_histogram()
    # pressure_temperature()
    plt.show()
