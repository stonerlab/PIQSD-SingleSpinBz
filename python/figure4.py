import os
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import time

# local imports
import analytic

def main():
    plt.style.use('resources/aps-paper.mplstyle')
    data_path = 'figures/figure4_data'
    os.makedirs(data_path, exist_ok=True)
    temperatures = np.linspace(0.07, 5, 50)

    quantum_spin = 2
    not_normalised = analytic.high_temperature_normalisation_sz(quantum_spin, temperatures)
    quantum_state = analytic.quantum_state_sz(quantum_spin, temperatures)
    normalised = (quantum_spin+1) / quantum_spin**2 * not_normalised

    np.savetxt(f"{data_path}/analytic_not_normalised_solution_s{quantum_spin}.tsv",
               np.column_stack((temperatures, not_normalised)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')
    np.savetxt(f"{data_path}/analytic_quantum_solution_s{quantum_spin}.tsv",
               np.column_stack((temperatures, quantum_state)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')
    np.savetxt(f"{data_path}/analytic_normalised_solution_s{quantum_spin}.tsv",
               np.column_stack((temperatures, normalised)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    # --- plotting ---
    plt.plot(temperatures, quantum_state, label='quantum solution', color="#FF3D00")
    plt.plot(temperatures, not_normalised, linestyle=(0, (4, 6)), label='not normalised', color="#FFD700")
    plt.plot(temperatures, normalised, linestyle=(0, (4, 6)), label='normalised', color="#00C2FF")

    plt.xlabel(r"$T$ (K)")
    plt.ylabel(r"$\langle\hat{S}_z\rangle/s$ ($\hbar$)")
    plt.legend(title=rf'$s={str(Fraction(quantum_spin))}$')
    plt.savefig('figures/figure4.pdf', transparent=True)


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(f'runtime: {end - start:.3f} (s)')