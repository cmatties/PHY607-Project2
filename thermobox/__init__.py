def harmonic_potential_plots():
    import argparse
    import matplotlib.pyplot as plt
    from .experiment import harmonic_equilibrium_checks
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, help = "Specify a seed for random number generation")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        seed = args.seed
    else:
        seed = 1
    harmonic_equilibrium_checks(seed = seed)
    plt.show()

def maxwell_boltzmann_distribution():
    import argparse
    import matplotlib.pyplot as plt
    from .experiment import mb_speed_histogram
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, help = "Specify a seed for random number generation")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        seed = args.seed
    else:
        seed = 1
    mb_speed_histogram(seed = seed)
    plt.show()
    
def pressure_temp():
    import argparse
    import matplotlib.pyplot as plt
    from .experiment import pressure_temperature
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, help = "Specify a seed for random number generation")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        seed = args.seed
    else:
        seed = 1
    pressure_temperature(seed = seed)
    plt.show()
   
def benchmarking():
    import argparse
    import matplotlib.pyplot as plt
    from .benchmarking import benchmark
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, help = "Specify a seed for random number generation")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        seed = args.seed
    else:
        seed = 1
    benchmark(seed = seed)
    plt.show()
