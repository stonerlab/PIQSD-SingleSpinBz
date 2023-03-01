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
    data_path = 'figures/figure2_data'
    os.makedirs(data_path, exist_ok=True)

    alpha = 0.5  # Gilbert Damping parameter.
    applied_field = np.array((0, 0, 1))  # Tesla, restricted to z-axis for this model

    # Temperature parameters
    temperatures = np.linspace(0.07, 5, 50)
    num_realisation = 20

    # Initial conditions
    s0 = np.array([1 / np.sqrt(3), 1.0 / np.sqrt(3), -1.0 / np.sqrt(3)])  # Initial spin

    # Equilibration time, final time and time step
    equilibration_time = 5  # Equilibration time ns
    production_time = 15  # Final time ns
    time_step = 0.00005  # Time step ns, "linspace" so needs to turn num into int

    fig, axs = plt.subplots(3, figsize=(3.5, 7))

    temperatures_analytical = np.linspace(0.02, 5, 100)

    i = 0
    for spin in np.array((0.5, 2, 5)):
        quantum_state = analytic.quantum_state_sz(spin, temperatures_analytical)
        classical_limit = analytic.classical_limit_sz(spin, temperatures_analytical)
        low_temperature_exponential = analytic.low_temperature_exponential_approximation_sz(spin
                                                                                            , temperatures_analytical)
        np.savetxt(f"{data_path}/analytic_quantum_solution_s{spin}.tsv",
                   np.column_stack((temperatures_analytical, quantum_state)), fmt='%.8e',
                   header='temperature_kelvin sz-expectation_hbar')
        np.savetxt(f"{data_path}/analytic_classical_limit_solution_s{spin}.tsv",
                   np.column_stack((temperatures_analytical, classical_limit)), fmt='%.8e',
                   header='temperature_kelvin sz-expectation_hbar')
        np.savetxt(f"{data_path}/analytic_low_temperature_exponential_solution_s{spin}.tsv",
                   np.column_stack((temperatures_analytical, low_temperature_exponential)), fmt='%.8e',
                   header='temperature_kelvin sz-expectation_hbar')
        axs[i].plot(temperatures_analytical, quantum_state, label='Quantum solution', color='#ff0000')
        axs[i].plot(temperatures_analytical, classical_limit, label='Classical limit', color='#28D75E')
        axs[i].plot(temperatures_analytical, low_temperature_exponential, label='From partition function', color='#0066ff')
        i += 1

    i = 0
    for spin in np.array((0.5, 2, 5)):

        classical_limit_solution_data_file = f'{data_path}/qsd_classical_limit_solution_s{spin}.tsv'

        if os.path.exists(classical_limit_solution_data_file):
            temperatures, sz_classical = np.loadtxt(classical_limit_solution_data_file, unpack=True)
        else:
            solver = asd.solver_factory('symplectic', 'classical-limit', spin, applied_field, alpha, time_step)

            sz_classical = asd.compute_temperature_dependence(solver, temperatures, 'classical-limit', spin, time_step,
                                                              equilibration_time, production_time, num_realisation, s0)

            np.savetxt(classical_limit_solution_data_file,
                   np.column_stack((temperatures, sz_classical)), fmt='%.8e',
                   header='temperature_kelvin sz-expectation_hbar')

        axs[i].plot(temperatures, sz_classical, label='Classical ASD simulation', linestyle=(0, (4, 6)), color='#D728A1')


        qsd_low_temperature_solution_data_file = f'{data_path}/qsd_low_temperature_solution_s{spin}.tsv'

        if os.path.exists(qsd_low_temperature_solution_data_file):
            temperatures, sz_low_temperature = np.loadtxt(qsd_low_temperature_solution_data_file, unpack=True)
        else:
            solver = asd.solver_factory('symplectic', 'low-temperature', spin, applied_field, alpha, time_step)

            sz_low_temperature = asd.compute_temperature_dependence(solver, temperatures, 'low-temperature', spin, time_step,
                                                                    equilibration_time, production_time, num_realisation, s0)

            np.savetxt(qsd_low_temperature_solution_data_file,
                       np.column_stack((temperatures, sz_low_temperature)), fmt='%.8e',
                       header='temperature_kelvin sz-expectation_hbar')

        axs[i].plot(temperatures, sz_low_temperature, label='Quantum ASD simulation', linestyle=(0, (4, 6)), color='#FF9900')
        axs[i].legend(title=rf'$s={str(Fraction(spin))}$')
        axs[i].set_ylabel(r"$\langle\hat{S}_z\rangle/s$ ($\hbar$)")

        i += 1

    plt.xlabel(r"$T$ (K)")
    axs[0].text(-1, 1, '(a)')
    axs[1].text(-1, 1, '(b)')
    axs[2].text(-1, 1, '(c)')
    plt.savefig('figures/figure2.pdf', transparent=True)


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(f'runtime: {end - start:.3f} (s)')
