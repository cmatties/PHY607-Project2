import numpy as np


class Particles:
    def __init__(self, x_positions, y_positions, x_velocities, y_velocities):
        # Create two lists of shape (N,2): one list of positions and one list of velocities.
        # We will manipulate these lists to evolve the state of the simulation
        self.positions = np.column_stack((x_positions, y_positions))
        self.velocities = np.column_stack((x_velocities, y_velocities))

    def get_positions(self):
        # return position list
        return self.positions

    def get_velocities(self):
        # return velocity list
        return self.velocities

    def evolve_particles(self):
        # Evolve particles according to rules that will be implemented in child classes
        return


class Charges(Particles):
    """Implement the list of particles with a 1/r potential between particles"""

    def __init__(self, x_positions, y_positions, x_velocities, y_velocities):
        super().__init__(x_positions, y_positions, x_velocities, y_velocities)

    def evolve_particles(self):
        # Iterate through list, getting the force on each particle from a 1/r potential with each other particle.
        # Then do Euler's method to evolve the state by 1 time step.
        return


class Balls(Particles):
    """Implement the list of particles which only interact through collisions"""

    def __init__(self, x_positions, y_positions, x_velocities, y_velocities):
        super().__init__(x_positions, y_positions, x_velocities, y_velocities)
