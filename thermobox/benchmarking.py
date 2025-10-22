import numpy as np
import matplotlib.pyplot as plt
from Particles import kB, HarmonicParticleList, HardParticleList
from sampling import gaussian_rejection_sampling
import timeit

N_list = np.array([50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
box = 1
dt = 0.001

hard_particle_times = []
harmonic_particle_times = []

for N in N_list:
    r0 = np.random.uniform(size = (N, 2))
    v0 = np.random.uniform(size = (N, 2))
    hard_particles = HardParticleList(r0, v0, Lx = box, Ly = box, seed = 1)
    harmonic_particles = HarmonicParticleList(r0, v0, Lx = box, Ly = box, seed = 1)
    
    harmonic_time = timeit.timeit(lambda : harmonic_particles.step(dt), number = 10)
    hard_time = timeit.timeit(lambda : hard_particles.step(dt), number = 10)
    
    hard_particle_times.append(hard_time)
    harmonic_particle_times.append(harmonic_time)

plt.figure()
plt.semilogy(N_list, np.array(harmonic_particle_times)/N_list, label = "Harmonic Particles/N")
plt.semilogy(N_list, np.array(hard_particle_times)/N_list**2, label = "Hard Particles/N^2")
plt.title("Time Complexity of a Single Time Step")
plt.xlabel("Number of particles")
plt.ylabel("Time")
plt.legend()
plt.show()
