import matplotlib.pyplot as plt
from experiment import harmonic_equilibrium_checks, mb_speed_histogram, pressure_temperature

if __name__ == "__main__":
    harmonic_equilibrium_checks(seed = 1)
    mb_speed_histogram(seed = 1)
    pressure_temperature(seed = 1)
    plt.show()
