import time
import argparse
import numpy as np
import asd

# Parsing parameters from command line
parser = argparse.ArgumentParser(description='Simulation parameters from command line.')

parser.add_argument('--integrator',
                    choices=['runge-kutta-4', 'symplectic'],
                    default='symplectic',
                    help='Numerical integration method for solving the spin dynamics')

parser.add_argument('--approximation',
                    choices=['low-temperature', 'classical-limit', 'high-temperature-first-order',
                             'high-temperature-second-order'],
                    required=True,
                    help='Approximation scheme to use')

parser.add_argument('--spin',
                    type=float,
                    required=True,
                    help='Quantum spin value (should normally be an integer multiple of 1/2)')

args = parser.parse_args()

integrator = args.integrator
qs = args.spin
approximation = args.approximation


def main():
    alpha = 0.5  # Gilbert Damping parameter.

    applied_field = np.array((0, 0, 1))  # Tesla, restricted to z-axis for this model

    # Temperature parameters
    temperatures = np.linspace(0.07, 5, 50)
    num_realisation = 4

    # Initial conditions
    s0 = np.array([1 / np.sqrt(3), 1.0 / np.sqrt(3), -1.0 / np.sqrt(3)])  # Initial spin

    # Equilibration time, final time and time step
    equilibration_time = 1  # Equilibration time ns
    production_time = 10  # Final time ns
    time_step = 0.00001  # Time step ns, "linspace" so needs to turn num into int


    solver = asd.solver_factory(integrator, approximation, qs, applied_field, alpha, time_step)
    sz = asd.compute_temperature_dependence(solver, temperatures, approximation, qs, time_step,
                                        equilibration_time, production_time, num_realisation, s0)

    file_name = f'qsd_{integrator}_{approximation}_{qs:.1f}.txt'

    header = f'spin: {qs}\n' \
             f'alpha: {alpha}\n' \
             f's0: {s0}\n' \
             f'integrator: {integrator}\n' \
             f'approximation: {approximation}\n' \
             f'time_step: {time_step}\n' \
             f'equilibration_time: {equilibration_time}\n' \
             f'production_time: {production_time}\n' \
             f'num_realisation: {num_realisation}\n' \
             f'\n' \
             'temperature_kelvin sz'

    np.savetxt(file_name, np.column_stack((temperatures, sz)), fmt='%.8e', header=header)


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(f'runtime: {end-start:.3f} (s)')
