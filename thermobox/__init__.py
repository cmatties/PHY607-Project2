def harmonic_potential_plots():
    """
    Utility function creating a script for harmonic potential plots
    """
    import argparse
    import matplotlib.pyplot as plt
    from .experiment import harmonic_equilibrium_checks

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, help="Specify a seed for random number generation"
    )

    args = parser.parse_args()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = 1
    harmonic_equilibrium_checks(seed=seed)
    plt.show()


def maxwell_boltzmann_distribution():
    """
    Utility function creating a script for the measurement of the Maxwell-Boltzmann distribution
    """
    import argparse
    import matplotlib.pyplot as plt
    from .experiment import mb_speed_histogram

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, help="Specify a seed for random number generation"
    )

    args = parser.parse_args()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = 1
    mb_speed_histogram(seed=seed)
    plt.show()


def pressure_temp():
    """
    Utility function creating a script for plots of pressure as a function of temperature
    """
    import argparse
    import matplotlib.pyplot as plt
    from .experiment import pressure_temperature

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, help="Specify a seed for random number generation"
    )

    args = parser.parse_args()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = 1
    pressure_temperature(seed=seed)
    plt.show()


def benchmarking():
    """
    Utility function creating a script for time-complexity benchmarking.
    """
    import argparse
    import matplotlib.pyplot as plt
    from .benchmarking import benchmark

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, help="Specify a seed for random number generation"
    )

    args = parser.parse_args()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = 1
    benchmark(seed=seed)
    plt.show()
