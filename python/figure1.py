import os
import numpy as np
import matplotlib.pyplot as plt
import time

# local imports
import analytic


def main():
    plt.style.use('resources/aps-paper.mplstyle')
    data_path = 'figures/figure1_data'
    os.makedirs(data_path, exist_ok=True)

    quantum_spin = 0.5
    temperature = np.linspace(0.02, 5, 500)
    num_correction_terms = 3

    # --- calculate solutions and save data ---
    quantum_solution = analytic.quantum_state_sz(quantum_spin, temperature)
    np.savetxt(f"{data_path}/analytic_quantum_solution_s{quantum_spin}.tsv",
               np.column_stack((temperature, quantum_solution)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    classical_limit = analytic.classical_limit_sz(quantum_spin, temperature)
    np.savetxt(f"{data_path}/analytic_classical_limit_s{quantum_spin}.tsv",
               np.column_stack((temperature, classical_limit)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    correction_terms = []
    for i in range(num_correction_terms):
        correction_terms.append(analytic.coherent_state_sz(quantum_spin, i+1, temperature))
        np.savetxt(f"{data_path}/analytic_coherent_state_terms{i+1}_s{quantum_spin}.tsv",
                   np.column_stack((temperature, correction_terms[i])), fmt='%.8e',
                   header='temperature_kelvin sz-expectation_hbar')

    # --- plotting ---
    plt.plot(temperature, quantum_solution, label='quantum solution', color="red")
    plt.plot(temperature, classical_limit, label='classical limit', color="blue")

    colors = plt.cm.cool(np.linspace(0, 1, num_correction_terms))
    for i in range(num_correction_terms):
        plt.plot(temperature, analytic.coherent_state_sz(quantum_spin, i+1, temperature), '--',
                 label=f'{i+1} correction terms', color=colors[i])

    plt.xlabel(r"$T$ (K)")
    plt.ylabel(r"$\langle\hat{S}_z\rangle/s$ ($\hbar$)")
    plt.legend(title=r'$s=1/2$')

    plt.savefig('figures/figure1.pdf', transparent=True)


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(f'runtime: {end - start:.3f} (s)')
