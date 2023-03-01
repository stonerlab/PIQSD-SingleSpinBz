import os
import numpy as np
import matplotlib.pyplot as plt
import time

# local imports
import analytic
import asd

def main():
    plt.style.use('resources/aps-paper.mplstyle')
    data_path = 'figures/figure3_data'
    os.makedirs(data_path, exist_ok=True)

    fig, axs = plt.subplots(1)
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
    labels = np.array(
        ('Classical ASD simulation', 'QASD $1^{st}$ correction simulation', 'QASD $2^{nd}$ correction simulation'))
    colors = np.array(('#D728A1', '#FF9900', '#DFD620'))

    axins = axs.inset_axes([0.3, 0.05, 0.53, 0.53])

    temperature_analytical = np.linspace(0.07, 5, 100)

    quantum_state = analytic.quantum_state_sz(2, temperature_analytical)
    classical_limit = analytic.classical_limit_sz(2, temperature_analytical)
    high_temperature_first_correction = \
        analytic.high_temperature_exponential_approximation_first_correction_sz(2, temperature_analytical)
    high_temperature_second_correction = \
        analytic.high_temperature_exponential_approximation_second_correction_sz(2, temperature_analytical)

    np.savetxt(f"{data_path}/analytical_quantum_state_solution_s2.0.tsv",
               np.column_stack((temperature_analytical, quantum_state)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')
    np.savetxt(f"{data_path}/analytical_classical_limit_solution_s2.0.tsv",
               np.column_stack((temperature_analytical, classical_limit)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')
    np.savetxt(f"{data_path}/analytical_high_temperature_first_correction_solution_s2.0.tsv",
               np.column_stack((temperature_analytical, high_temperature_first_correction)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')
    np.savetxt(f"{data_path}/analytical_high_temperature_second_correction_solution_s2.0.tsv",
               np.column_stack((temperature_analytical, high_temperature_second_correction)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    axs.plot(temperature_analytical, quantum_state, label='Quantum solution', color='#ff0000')
    axs.plot(temperature_analytical, classical_limit, label='Classical limit', color='#28D75E')
    axs.plot(temperature_analytical, high_temperature_first_correction, label='First correction', color='#0066ff')
    axs.plot(temperature_analytical, high_temperature_second_correction, label='Second correction', color='#2029DF')

    axins.plot(temperature_analytical, quantum_state, color='#ff0000')
    axins.plot(temperature_analytical, classical_limit, color='#28D75E')
    axins.plot(temperature_analytical, high_temperature_first_correction, color='#0066ff')
    axins.plot(temperature_analytical, high_temperature_second_correction, color='#2029DF')

    i = 0
    for approximation in ('classical-limit', 'high-temperature-first-order', 'high-temperature-second-order'):

        asd_data_file = f'{data_path}/qsd_{approximation}_solution_s2.0.tsv'

        if os.path.exists(asd_data_file):
            temperatures, sz = np.loadtxt(asd_data_file, unpack=True)
        else:
            solver = asd.solver_factory('symplectic', approximation, 2, applied_field, alpha, time_step)
            sz = asd.compute_temperature_dependence(solver, temperatures, approximation, 2, time_step
                                                , equilibration_time, production_time, num_realisation, s0)

            np.savetxt(f"{data_path}/qsd_{approximation}_solution_s2.0.tsv",
                       np.column_stack((temperatures, sz)), fmt='%.8e',
                       header='temperature_kelvin sz-expectation_hbar')
        axs.plot(temperatures, sz, label=labels[i], linestyle=(0, (4, 6)), color=colors[i])
        axins.plot(temperatures, sz, linestyle=(0, (4, 6)), color=colors[i])
        i += 1

    x1, x2, y1, y2 = 0.5, 1.7, 0.65, 0.91
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.set_xticks([])
    axins.set_yticks([])
    axs.axhline(0, color='grey', linestyle='--', linewidth=0.5)

    axs.indicate_inset_zoom(axins, edgecolor="black")
    plt.xlabel(r"$T$ (K)")
    plt.ylabel(r"$\langle\hat{S}_z\rangle/s$ ($\hbar$)")
    plt.legend(bbox_to_anchor=(0.2, -0.5), loc='center', fancybox=True)
    plt.text(3.5, 0.75, '$s=2$')
    plt.savefig('figures/figure3.pdf', transparent=True)


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(f'runtime: {end - start:.3f} (s)')
