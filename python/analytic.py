import numpy as np
from scipy.special import factorial
import scipy.integrate as integrate
from sympy import ln, exp, series, symbols, lambdify, simplify, diff
from sympy.abc import x, a, b
from numba import njit
import asd
vars = symbols('a x b')


def quantum_state_sz(quantum_spin, temperature):
    """Returns the expectation value of z component of spin computed by

    <S_z>=(Sum_[m=-s..s] <s,m|S_z*exp(g * mu_b * S_z /(hbar * k_B * temperature) )|s,m>)/(Sum_[m=-s..s] <s,m|exp(beta * g * mu_b * S_z / hbar )|s,m>)

    """
    z = np.zeros(np.array(temperature).shape)
    for m in np.arange(-quantum_spin, quantum_spin + 1):
        z = z + np.exp(asd.g_muB_by_kB * m / temperature)

    p = np.zeros(np.array(temperature).shape)
    for m in np.arange(-quantum_spin, quantum_spin + 1):
        p = p + m * np.exp(asd.g_muB_by_kB * m / temperature)

    return (1.0/quantum_spin) * p / z


def classical_limit_sz_numerator(quantum_spin, temperature):
    """Returns the numerator of the expectation value for the z-component of the spin computed as

    (2s+1)/s * (k_B / (g * mu_B))^2 * temperature (g * mu_B / k_B * s * cosh(g * mu_B * s / ( k_B * temperature)) - temperature * sinh(g * mu_B * s / (k_B * temperature)))

    """
    return (2 * quantum_spin + 1) * temperature * (asd.g_muB_by_kB * quantum_spin * np.cosh(asd.g_muB_by_kB * quantum_spin / temperature) - temperature * np.sinh(asd.g_muB_by_kB * quantum_spin / temperature)) / (asd.g_muB_by_kB ** 2 * quantum_spin)


def classical_limit_sz_denominator(quantum_spin, temperature):
    """Returns the denominator of the expectation value for the z-component of the spin computed as

    (2s+1)/s * (k_B / (g * mu_B)) * temperature * sinh(g * mu_B * s / (k_B * temperature))

    """
    return (2 * quantum_spin + 1) * temperature * np.sinh(asd.g_muB_by_kB * quantum_spin / temperature) / (asd.g_muB_by_kB * quantum_spin)


def classical_limit_sz(quantum_spin, temperature):
    """Returns the normalised expectation value of s_z in the classical limit"""
    return (1.0/quantum_spin) * classical_limit_sz_numerator(quantum_spin, temperature) / classical_limit_sz_denominator(quantum_spin, temperature)


def quantum_correction(quantum_spin, correction_order, x):
    """Returns the n-th non-commuting terms of the matrix elements of S_z function multiplied by the integration measure computed as

    (<z|S_z^n|z>-<z|S_z|z>^n)* |z| / (1 + |z|^2)^2

    """
    total = np.zeros(np.array(x).shape)

    for p in range(0, int(np.round(2 * quantum_spin)) + 1):
        total = total + factorial(2 * quantum_spin) / (factorial(p) * factorial(2 * quantum_spin - p)) * (x ** 2) ** p * (quantum_spin - p) ** correction_order / (1 + x ** 2) ** (2 * quantum_spin)

    return (total - (quantum_spin * (1 - x ** 2) / (1 + x ** 2)) ** correction_order) * x / (1 + x ** 2)**2


def quantum_correction_sz_numerator(quantum_spin, correction_order, temperature):
    """Returns the n-th correction to the numerator for the expectation value of s_z computed as

    (2s+1) * 2 * sum_[k=2..n+3] {((g * mu_B)/(k_B * temperature))^(k-1) * 1 / (k-1)!) integral_[x=0..infinity] quantum_correction(k)}

    """
    total = np.zeros(np.array(temperature.shape))
    for k in range(2, correction_order + 3):
        integral, _ = integrate.quad(lambda x: quantum_correction(quantum_spin, k, x), 0, np.inf)
        total = total + 1 / (factorial(k - 1) * (temperature / asd.g_muB_by_kB) ** (k - 1)) * (2 * quantum_spin + 1) * 2 * integral

    return total


def quantum_correction_sz_denominator(quantum_spin, correction_order, temperature):
    """Returns the n-th correction to the denominator for the expectation value of s_z computed as

    (2s+1) * 2 * sum_[k=2..n+3] { 1 / k! * ((g * mu_B)/(k_B * temperature))^(k) * integral_[x=0..infinity] quantum_correction(k)}

    """
    total = np.zeros(np.array(temperature.shape))
    for k in range(2, correction_order + 2):
        integral, _ = integrate.quad(lambda x: quantum_correction(quantum_spin, k, x), 0, np.inf)
        total = total + 1 / (factorial(k) * (temperature / asd.g_muB_by_kB)**k) * (2 * quantum_spin + 1) * 2 * integral

    return total


def coherent_state_sz(quantum_spin, correction_order, temperature):
    """Returns the normalised expectation value for the z-component of spin for corrected to the n-th order in the spin
    coherent state basis
    """
    numerator = classical_limit_sz_numerator(quantum_spin, temperature) + quantum_correction_sz_numerator(quantum_spin, correction_order, temperature)
    denominator = classical_limit_sz_denominator(quantum_spin, temperature) + quantum_correction_sz_denominator(quantum_spin, correction_order, temperature)
    return (1.0/quantum_spin) * numerator / denominator


def low_temperature_numerator_integrand(quantum_spin, temperature, x):
    """Returns the integrand of the numerator of the expectation value for the low temperature approximation"""
    return x / (1 + x**2)**2 * quantum_spin * (1 - x ** 2) / (1 + x ** 2) * np.exp((asd.g_muB_by_kB / temperature)
                                                                                   * (quantum_spin * (1 - x ** 2) / (1 + x ** 2) - np.sqrt(2 * quantum_spin) * x / (1 + x ** 2)))


def low_temperature_denominator_integrand(quantum_spin, temperature, x):
    """Returns the integrand of the denominator of the expectation value for the low temperature approximation"""
    return x / (1 + x**2)**2 * np.exp((asd.g_muB_by_kB / temperature) * (quantum_spin * (1 - x ** 2) / (1 + x ** 2) - np.sqrt(2 * quantum_spin) * x / (1 + x ** 2)))


def low_temperature_exponential_approximation_numerator(quantum_spin, temperatures):
    """Returns the integral of the function low_temperature_numerator integral for set of temperatures"""
    total = np.zeros(np.array(temperatures.shape))
    i = 0
    for temperature in temperatures:
        total[i], _ = integrate.quad(lambda x: low_temperature_numerator_integrand(quantum_spin, temperature, x), 0, np.inf)
        i += 1
    return total


def low_temperature_exponential_approximation_denominator(quantum_spin, temperatures):
    """Returns the integral of the function low_temperature_denominator integral for set of temperatures"""
    total = np.zeros(np.array(temperatures.shape))
    i = 0
    for temperature in temperatures:
        total[i], _ = integrate.quad(lambda x: low_temperature_denominator_integrand(quantum_spin, temperature, x), 0, np.inf)
        i += 1
    return total


def low_temperature_exponential_approximation_sz(quantum_spin, temperature):
    """Returns the expectation value of the z-component of the spin for the exponential approximation
    of the low temperature approximation
    """
    numerator = low_temperature_exponential_approximation_numerator(quantum_spin, temperature)
    denominator = low_temperature_exponential_approximation_denominator(quantum_spin, temperature)
    return (1.0/quantum_spin) * numerator/denominator


def high_temperature_numerator_first_correction_integrand(quantum_spin, temperature, x):
    """Returns the integrand of the numerator of the expectation value for the  high approximation
     with the first correction"""
    return x / (1 + x**2)**2 * quantum_spin * (1 - x**2) / (1 + x**2) * np.exp(-((quantum_spin * (-1 + x**2) * asd.g_muB_by_kB/temperature)/(1 + x**2)) + (quantum_spin * x**2 * (asd.g_muB_by_kB/temperature)**2)/(1 + x**2)**2)


def high_temperature_denominator_first_correction_integrand(quantum_spin, temperature, x):
    """Returns the integrand of the denominator of the expectation value for the  high approximation
     with the first correction"""
    return x / (1 + x**2)**2 * np.exp((-(quantum_spin * (-1 + x**2) * asd.g_muB_by_kB / temperature)/(1 + x**2)) + (quantum_spin * x**2 * (asd.g_muB_by_kB/temperature)**2)/(1 + x**2)**2)


def high_temperature_exponential_approximation_first_correction_numerator(quantum_spin, temperatures):
    """Returns the integral of the function high_temperature_numerator_first_correction_integrand integral for set of temperatures"""
    total = np.zeros(np.array(temperatures.shape))
    i = 0
    for temperature in temperatures:
        total[i], _ = integrate.quad(lambda x: high_temperature_numerator_first_correction_integrand(quantum_spin, temperature, x), 0, np.inf)
        i += 1
    return total


def high_temperature_exponential_approximation_first_correction_denominator(quantum_spin, temperatures):
    """Returns the integral of the function high_temperature_denominator_first_correction_integrand integral for set of temperatures"""
    total = np.zeros(np.array(temperatures.shape))
    i = 0
    for temperature in temperatures:
        total[i], _ = integrate.quad(lambda x: high_temperature_denominator_first_correction_integrand(quantum_spin, temperature, x), 0, np.inf)
        i += 1
    return total


def high_temperature_exponential_approximation_first_correction_sz(quantum_spin, temperature):
    """Returns the expectation value of the z-component of the spin for the exponential approximation
    of the high temperature approximation with first correction
    """
    numerator = high_temperature_exponential_approximation_first_correction_numerator(quantum_spin, temperature)
    denominator = high_temperature_exponential_approximation_first_correction_denominator(quantum_spin, temperature)
    return (quantum_spin + 1)/quantum_spin**2 * numerator/denominator


def high_temperature_numerator_second_correction_integrand(quantum_spin, temperature, x):
    """Returns the integrand of the numerator of the expectation value for the  high approximation
     with the second correction"""
    return x / (1.0 + x**2)**2 * quantum_spin * (1.0 - x**2) / (1.0 + x**2) * np.exp(((quantum_spin * (1.0 - x**2) * asd.g_muB_by_kB/temperature)/(1.0 + x**2)) + (quantum_spin * x**2 * (asd.g_muB_by_kB/temperature)**2)/((1 + x**2)**2) - 1.0/3.0 * quantum_spin * (asd.g_muB_by_kB/temperature)**3 * x**2 * (1.0 - x**2)/((1.0 + x**2)**3))


def high_temperature_denominator_second_correction_integrand(quantum_spin, temperature, x):
    """Returns the integrand of the denominator of the expectation value for the  high approximation
     with the second correction"""
    return x / (1.0 + x**2)**2 * np.exp(((quantum_spin * (1.0 - x**2) * asd.g_muB_by_kB / temperature)/(1.0 + x**2)) + (quantum_spin * x**2 * (asd.g_muB_by_kB/temperature)**2)/(1.0 + x**2)**2 - 1.0/3.0 * quantum_spin * (asd.g_muB_by_kB/temperature)**3 * x**2 * (1.0 - x**2)/(1.0 + x**2)**3)


def high_temperature_exponential_approximation_second_correction_numerator(quantum_spin, temperatures):
    """Returns the integral of the function high_temperature_numerator_second_correction_integrand integral
    for set of temperatures
    """
    total = np.zeros(np.array(temperatures.shape))
    i = 0
    for temperature in temperatures:
        total[i], _ = integrate.quad(lambda x: high_temperature_numerator_second_correction_integrand(quantum_spin, temperature, x), 0, np.inf)
        i += 1
    return total


def high_temperature_exponential_approximation_second_correction_denominator(quantum_spin, temperatures):
    """Returns the integral of the function high_temperature_denominator_second_correction_integrand integral
    for set of temperatures
    """
    total = np.zeros(np.array(temperatures.shape))
    i = 0
    for temperature in temperatures:
        total[i], _ = integrate.quad(lambda x: high_temperature_denominator_second_correction_integrand(quantum_spin, temperature, x), 0, np.inf)
        i += 1
    return total


def high_temperature_exponential_approximation_second_correction_sz(quantum_spin, temperature):
    """Returns the expectation value of the z-component of the spin for the exponential approximation
    of the high temperature approximation with second correction
    """
    numerator = high_temperature_exponential_approximation_second_correction_numerator(quantum_spin, temperature)
    denominator = high_temperature_exponential_approximation_second_correction_denominator(quantum_spin, temperature)
    return (quantum_spin + 1)/quantum_spin**2 * numerator/denominator


def high_temperature_numerator_normalisation_integrand(quantum_spin, temperature, x):
    """Returns the integrand of the numerator of the expectation value for the non-normalised expectation
    value due to approximation

    <z|S_z*exp(S_z)|z>~<z|S_z|z> * <z|exp(S_z)|z>

    """
    return x / (1.0 + x**2)**2 * quantum_spin * (1.0 - x**2) / (1.0 + x**2) * ((np.exp(asd.g_muB_by_kB/temperature) + x**2)**2 / (1 + x**2)**2)**quantum_spin


def high_temperature_denominator_normalisation_integrand(quantum_spin, temperature, x):
    """Returns the integrand of the denominator of the expectation value for the non-normalised expectation
    value due to approximation

    <z|S_z*exp(S_z)|z>~<z|S_z|z> * <z|exp(S_z)|z>

    """
    return x / (1.0 + x**2)**2 * ((np.exp(asd.g_muB_by_kB/temperature) + x**2)**2 / (1 + x**2)**2)**quantum_spin


def high_temperature_normalisation_numerator(quantum_spin, temperatures):
    """Returns the integral of the function high_temperature_numerator_normalisation_integrand integral
    for set of temperatures
    """
    total = np.zeros(np.array(temperatures.shape))
    i = 0
    for temperature in temperatures:
        total[i], _ = integrate.quad(lambda x: high_temperature_numerator_normalisation_integrand(quantum_spin, temperature, x), 0, np.inf)
        i += 1
    return total


def high_temperature_normalisation_denominator(quantum_spin, temperatures):
    """Returns the integral of the function high_temperature_denominator_normalisation_integrand integral
    for set of temperatures
    """
    total = np.zeros(np.array(temperatures.shape))
    i = 0
    for temperature in temperatures:
        total[i], _ = integrate.quad(lambda x: high_temperature_denominator_normalisation_integrand(quantum_spin, temperature, x), 0, np.inf)
        i += 1
    return total


def high_temperature_normalisation_sz(quantum_spin, temperature):
    """Returns the expectation value of the z-component of the spin for the non-normalised value due to approximation

    <z|S_z*exp(S_z)|z>~<z|S_z|z> * <z|exp(S_z)|z>

    """
    numerator = high_temperature_normalisation_numerator(quantum_spin, temperature)
    denominator = high_temperature_normalisation_denominator(quantum_spin, temperature)
    return numerator/denominator


def high_temperature_hamiltonian_series(quantum_spin, order):
    """Returns a sympy polynomial approximation in powers of beta of the exponential
    approximation of the integrand of the partition function
    """
    f = 2 * quantum_spin * (series(ln(1 + x**2*exp(-asd.g_muB_by_kB*b*a)), a, 0, order, dir='+').removeO() - ln(1 + x**2))
    return f


def generate_hamiltonian_function(quantum_spin, order):
    """Casts the sympy expression from the method high_temperature_hamiltonian_series into a njit python function
    with variables a = 1/temperature, x = spin and b = applied_field.

    Usable as:

    hamiltonian = generate_hamiltonian_function(quantum_spin, order)
    hamiltonian(1/temperature, spin, applied_field)

    """
    g = lambdify(vars, high_temperature_hamiltonian_series(quantum_spin, order), 'numpy')
    return njit(g)


def high_temperature_effective_field(quantum_spin, order):
    """Returns a sympy expression for the effective field corresponding to
    the approximation in high_temperature_hamiltonian_series
    """
    f = simplify(high_temperature_hamiltonian_series(quantum_spin, order).collect(a).subs(x**2, (1-x)/(1+x)))
    g = diff(f, x)/(a*asd.g_muB_by_kB*quantum_spin)
    return g


def generate_field_function(quantum_spin, order):
    """Casts the sympy expression from the method high_temperature_effective_field into a njit python function
    with variables a = 1/temperature, x = spin and b = applied_field.

    Usable as:

    field = generate_field_function(quantum_spin, order)
    field(1/temperature, spin, applied_field)

    """
    g = lambdify(vars, high_temperature_effective_field(quantum_spin, order), 'numpy')
    return njit(g)


def high_temperature_numerator_correction_integrand(quantum_spin, temperature, hamiltonian, x):
    """Returns the integrand of the numerator of the expectation value for the high temperature approximation
     with the specified correction in the hamiltonian generation
     """
    return x / (1.0 + x**2)**2 * quantum_spin * (1.0 - x**2) / (1.0 + x**2) * np.exp(hamiltonian(1.0/temperature, x, 1))


def high_temperature_denominator_correction_integrand(temperature, hamiltonian, x):
    """Returns the integrand of the denominator of the expectation value for the high temperature approximation
    with the specified correction in the hamiltonian generation
    """
    return x / (1.0 + x**2)**2 * np.exp(hamiltonian(1.0/temperature, x, 1))


def high_temperature_exponential_approximation_numerator(quantum_spin, temperatures, hamiltonian):
    """Returns the integral of the function high_temperature_numerator_correction_integrand integral
    for set of temperatures
    """
    total = np.zeros(np.array(temperatures.shape))
    i = 0
    for temperature in temperatures:
        total[i], _ = integrate.quad(lambda x: high_temperature_numerator_correction_integrand(quantum_spin, temperature, hamiltonian, x), 0, np.inf)
        i += 1
    return total


def high_temperature_exponential_approximation_denominator(temperatures, hamiltonian):
    """Returns the integral of the function high_temperature_denominator_correction_integrand integral
    for set of temperatures
    """
    total = np.zeros(np.array(temperatures.shape))
    i = 0
    for temperature in temperatures:
        total[i], _ = integrate.quad(lambda x: high_temperature_denominator_correction_integrand(temperature, hamiltonian,  x), 0, np.inf)
        i += 1
    return total


def high_temperature_exponential_approximation_correction_sz(quantum_spin, temperature, hamiltonian):
    """Returns the expectation value of the z-component of the spin for the exponential approximation
    of the high temperature approximation the specified correction in the hamiltonian generation
    """
    numerator = high_temperature_exponential_approximation_numerator(quantum_spin, temperature, hamiltonian)
    denominator = high_temperature_exponential_approximation_denominator(temperature, hamiltonian)
    return (quantum_spin + 1)/quantum_spin**2 * numerator/denominator

