import os
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import time

# local imports
import analytic
import asd


def main():
    plt.style.use('resources/aps-paper.mplstyle')
    data_path = 'figures/figure5_data'
    os.makedirs(data_path, exist_ok=True)

    quantum_spin = 2
    temperature = np.linspace(0.02, 5, 500)
    correction_order = 13

    alpha = 0.5  # Gilbert Damping parameter.
    applied_field = np.array((0, 0, 1))  # Tesla, restricted to z-axis for this model

    # Temperature parameters
    temperatures_asd = np.linspace(0.3, 5, 100)
    num_realisation = 20

    # Initial conditions
    s0 = np.array([1 / np.sqrt(3), 1.0 / np.sqrt(3), -1.0 / np.sqrt(3)])  # Initial spin

    # Equilibration time, final time and time step
    equilibration_time = 5  # Equilibration time ns
    production_time = 15  # Final time ns
    time_step = 0.00005  # Time step ns, "linspace" so needs to turn num into int

    # --- calculate solutions and save data ---
    quantum_solution = analytic.quantum_state_sz(quantum_spin, temperature)
    np.savetxt(f"{data_path}/analytic_quantum_solution_s{quantum_spin}.tsv",
               np.column_stack((temperature, quantum_solution)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    classical_limit = analytic.classical_limit_sz(quantum_spin, temperature)
    np.savetxt(f"{data_path}/analytic_classical_limit_s{quantum_spin}.tsv",
               np.column_stack((temperature, classical_limit)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    hamiltonian = analytic.generate_hamiltonian_function(quantum_spin, correction_order)

    correction = analytic.high_temperature_exponential_approximation_correction_sz(quantum_spin,
                                                                                   temperature, hamiltonian)
    np.savetxt(f"{data_path}/analytic_coherent_state_terms{correction_order}_s{quantum_spin}.tsv",
               np.column_stack((temperature, correction)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    asd_data_file = f"{data_path}/qsd_high_temperature_limit_{correction_order-2}_order_solution_s{quantum_spin}.tsv"

    if os.path.exists(asd_data_file):
        temperatures_asd, sz_asd = np.loadtxt(asd_data_file, unpack=True)
    else:
        solver = asd.solver_factory('symplectic', 'high-temperature-9th-order', quantum_spin,
                                    applied_field, alpha, time_step)
        sz_asd = asd.compute_temperature_dependence(solver, temperatures_asd, 'high-temperature-9th-order',
                                                quantum_spin, time_step,
                                                equilibration_time, production_time, num_realisation, s0)

    np.savetxt(f"{data_path}/qsd_high_temperature_limit_{correction_order-2}_order_solution_s{quantum_spin}.tsv",
               np.column_stack((temperatures_asd, sz_asd)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    # --- plotting ---
    plt.plot(temperatures_asd, sz_asd, label='qsd high temperature approximation')
    plt.plot(temperature, quantum_solution, label='quantum solution', color="red")
    plt.plot(temperature, classical_limit, label='classical limit', color="blue")

    plt.plot(temperature, correction, '--', label=f'{correction_order-2}-th correction terms', color="purple")

    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)

    plt.xlabel(r"$T$ (K)")
    plt.ylabel(r"$\langle\hat{S}_z\rangle/s$ ($\hbar$)")
    plt.legend(title=rf'$s={str(Fraction(quantum_spin))}$')

    plt.savefig('figures/figure5.pdf', transparent=True)


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(f'runtime: {end - start:.3f} (s)')
