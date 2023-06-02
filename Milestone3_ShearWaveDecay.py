import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as opt
from scipy.signal import argrelextrema
from LBM import LBM

def calc_analytical_solution(a0 : float, viscosity : float, L : int, timestep : int, offset : float = 0.0):
    """
    Calculate the analytical solution for the shear wave decay.
    """
    return offset + a0 * np.exp(-viscosity * (2*np.pi/L)**2 * timestep) * np.sin(2 * np.pi * (np.arange(0, L, 1) / L))

def calc_kinematic_viscosity(indices : np.ndarray, maxima : np.ndarray, period):

    # fit exponential function to maxima
    def func(x, b):
        return a0 * np.exp(-b * period**2 * x)
    popt, pcov = opt.curve_fit(func, indices, maxima)
    viscosity = popt[0]
    return viscosity

def density_shear_wave_decay(grid_size_x : int, grid_size_y : int, omega : float, timesteps : int, rho0 : float, a0 : float, period_multiplier : int, plot=False):
    """
    Simulate a shear wave decay in a 2D fluid with a density perturbation.
    """

    # create shear wave with density field perturbation
    L = grid_size_x
    period = period_multiplier * 2 * np.pi / L
    initial_density_row = rho0 + a0*np.sin(period * np.arange(L))
    initial_density_yx = np.tile(initial_density_row, (grid_size_y,1))

    # initialize LBM
    lbm = LBM(grid_size_x, grid_size_y, omega, inital_density_field_yx=initial_density_yx)

    analytical_viscositity = lbm.viscosity
    simulated_density_field_tyx = []
    simulated_density_field_tyx.append(lbm.density_field_yx)
    analytical_density_field_tx = []
    analytical_density_field_tx.append(calc_analytical_solution(a0, analytical_viscositity, L, timestep=0, offset=rho0))

    timesteps_for_plotting = np.linspace(0, timesteps, 10, dtype=int)
    fig_density_over_time = None
    ax1 = None
    if plot:
        fig_density_over_time = plt.figure()
        ax1 = fig_density_over_time.add_subplot(111)
    for i in range(timesteps):
        lbm.step()
        lbm.update_density_field()
        simulated_density_field_tyx.append(lbm.density_field_yx)
        analytical_density_field_tx.append(calc_analytical_solution(a0, analytical_viscositity, L, timestep=i+1, offset=rho0))

        if plot and i in timesteps_for_plotting:
            ax1.plot(np.arange(L), lbm.density_field_yx[0, :], label="t={}".format(i))
    
    if plot:
        ax1.set_xlabel("x")
        ax1.set_ylabel("Density")
        # ax1.set_title("Density at x = {}".format(int(L/(4*period_multiplier))))
        ax1.legend()
        fig_density_over_time.tight_layout()
        fig_density_over_time.savefig("density_over_time_{}.png".format(timesteps))
        plt.show()

    # find maxima of simulated solution over time
    simulated_density_field_tyx = np.array(simulated_density_field_tyx)
    simulated_solution_t = np.amax(np.abs(simulated_density_field_tyx - rho0), axis=(1,2))
    # find indices of maxima
    maxima_indices = argrelextrema(simulated_solution_t, np.greater)[0]
    indices = [0]
    indices.extend(maxima_indices)
    indices = np.array(indices)
    new_maxima = simulated_solution_t[maxima_indices]
    maxima = [a0]
    maxima.extend(new_maxima)   

    simulated_viscosity = calc_kinematic_viscosity(indices, maxima, period)

    # print("Analytical viscosity: {}".format(analytical_viscositity))
    # print("Simulated viscosity: {}".format(simulated_viscosity))
    if plot:
        position = int(L/(4*period_multiplier))
        analytical_velocity_t = np.array(analytical_density_field_tx)[:, position] - rho0
        simulated_velocity_t = np.array(simulated_density_field_tyx)[:, 0, position] - rho0
        plot_decaying_wave(analytical_velocity_t, simulated_velocity_t, "Density", "Density at {} = {}".format("x", position))
    return analytical_viscositity, simulated_viscosity

def velocity_shear_wave_decay(grid_size_x : int, grid_size_y : int, omega : float, timesteps : int, rho0 : float, a0 : float, period_multiplier : int, plot=False):
    # shear wave with velocity field perturbation
    L = grid_size_y
    period = period_multiplier * 2 * np.pi / L

    initial_average_velocity_col = a0*np.sin(period * np.arange(L))
    initial_average_velocity_yx = np.tile(initial_average_velocity_col, (grid_size_x,1)).T

    initial_density_yx =  np.ones((grid_size_y, grid_size_x)) * rho0
    initial_average_velocity_Cyx = np.einsum("yx, C -> Cyx", initial_average_velocity_yx, np.array([1,0]))

    # initialize LBM
    lbm = LBM(grid_size_x, grid_size_y, omega, inital_density_field_yx=initial_density_yx, inital_velocity_field_Cyx=initial_average_velocity_Cyx)

    analytical_viscositity = lbm.viscosity
    simulated_velocity_field_tyx = []
    simulated_velocity_field_tyx.append(lbm.velocity_field_Cyx[0, :, :])
    analytical_velocity_field_ty = []
    analytical_velocity_field_ty.append(calc_analytical_solution(a0, analytical_viscositity, L, timestep=0))

    timesteps_for_plotting = np.linspace(0, timesteps, 10, dtype=int)
    fig_velocity_over_time = None
    ax1 = None
    if plot:
        fig_velocity_over_time = plt.figure()
        ax1 = fig_velocity_over_time.add_subplot(111)
    for i in range(timesteps):
        lbm.step()
        lbm.update_velocity_field()
        # only keep the x component of the velocity field
        simulated_velocity_field_tyx.append(lbm.velocity_field_Cyx[0, :, :])
        analytical_velocity_field_ty.append(calc_analytical_solution(a0, analytical_viscositity, L, timestep=i+1))

        if plot and i in timesteps_for_plotting:
            ax1.plot(np.arange(L), lbm.velocity_field_Cyx[0, :, 0], label="t={}".format(i))
    
    if plot:
        ax1.set_xlabel("y")
        ax1.set_ylabel("Velocity")
        # ax1.set_title("Velocity at y = {}".format(int(L/(4*period_multiplier))))
        ax1.legend()
        fig_velocity_over_time.tight_layout()
        fig_velocity_over_time.savefig("velocity_over_time_{}.png".format(timesteps))
        plt.show()

    # find maxima of simulated solution over time
    simulated_solution_t = np.amax(np.abs(simulated_velocity_field_tyx), axis=(1,2))
    # calculate viscosity from maxima
    simulated_viscosity = calc_kinematic_viscosity(np.arange(timesteps + 1), simulated_solution_t, period)

    # print("Analytical viscosity: {}".format(analytical_viscositity))
    # print("Simulated viscosity: {}".format(simulated_viscosity))
    if plot:
        position = int(L/(4*period_multiplier))
        analytical_velocity_t = np.array(analytical_velocity_field_ty)[:, position]
        simulated_velocity_t = np.array(simulated_velocity_field_tyx)[:, position, 0]
        plot_decaying_wave(analytical_velocity_t, simulated_velocity_t, "Velocity", "Velocity at {} = {}".format("y", position))
    return analytical_viscositity, simulated_viscosity

def viscosity_over_omega(grid_size_x, grid_size_y, timesteps, rho0, a0, k, use_density_perturbation=True):
    omega_values = np.arange(0.1, 2, 0.1)
    analytical_viscosity_values = []
    simulated_viscosity_values = []

    analytical_viscosity = 0
    simulated_viscosity = 0
    for omega in omega_values:
        if use_density_perturbation:
            analytical_viscosity, simulated_viscosity = density_shear_wave_decay(grid_size_x, grid_size_y, omega, timesteps, rho0, a0, k)
        else:
            analytical_viscosity, simulated_viscosity = velocity_shear_wave_decay(grid_size_x, grid_size_y, omega, timesteps, rho0, a0, k)
        analytical_viscosity_values.append(analytical_viscosity)
        simulated_viscosity_values.append(simulated_viscosity)

    # create a figure that displays the plot
    plt.figure()
    plt.plot(omega_values, analytical_viscosity_values, label="Analytical viscosity")
    plt.plot(omega_values, simulated_viscosity_values, label="Simulated viscosity")
    plt.xlabel("Omega")
    plt.ylabel("Viscosity")
    plt.legend()
    plt.savefig("viscosity_over_omega_{}.png".format("density" if use_density_perturbation else "velocity"))
    plt.show()

def plot_decaying_wave(analytical_data_values_t, simulated_data_values_t, value_type, title):
    # plot the data values over time
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(analytical_data_values_t, label="Analytical")
    ax1.plot(simulated_data_values_t, label="Simulated")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel(value_type)
    ax1.set_title(title)
    ax1.legend()
    fig.tight_layout()
    fig.savefig(title + ".png")
    plt.show()


if __name__ == "__main__":
    # Simulation parameters
    grid_size_x = 150
    grid_size_y = 100
    epsilon=0.01
    timesteps = 1000
    rho0 = 1.0
    omega = 0.3
    a0 = epsilon
    k = 1
    use_density_perturbation = False
    plot_wave = True
    density_shear_wave_decay(grid_size_x, grid_size_y, omega, timesteps, rho0, a0, k, plot=plot_wave)
    # velocity_shear_wave_decay(grid_size_x, grid_size_y, omega, timesteps, rho0, a0, k, plot=plot_wave)
    # viscosity_over_omega(grid_size_x, grid_size_y, timesteps, rho0, a0, k, use_density_perturbation=use_density_perturbation)
