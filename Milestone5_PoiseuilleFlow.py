import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from LBM import LBM

def poiseuille_flow(grid_size_x : int, grid_size_y : int, omega : float, timesteps : int):
    """
    Simulate Poiseuille flow.

    Parameters
    ----------
    grid_size_x : int
        The number of grid points in the x direction.
    grid_size_y : int
        The number of grid points in the y direction.
    omega : float
        The relaxation parameter.
    timesteps : int
        The number of timesteps to simulate.

    Returns
    -------
    None.
    """
    base_pressure = 1 / 3
    pressure_difference = 0.001
    inlet_pressure = base_pressure + pressure_difference
    outlet_pressure = base_pressure - pressure_difference
    
    boundary_conditions = {"bottom": "bounce_back", "top": "bounce_back", "left": "periodic", "right": "periodic"}
    boundary_pressure = {"bottom": None, "top": None, "left": inlet_pressure, "right": outlet_pressure, "output": "right", "input": "left"}

    # initialize LBM
    lbm = LBM(grid_size_x, grid_size_y, omega, boundary_conditions=boundary_conditions, boundary_pressure_info=boundary_pressure)

    # simulate Poiseuille flow
    print("Simulating Poiseuille flow...")
    simulated_velocity_field_tCyx = [lbm.get_velocity_field_Cyx(False)]
    indices = [0]
    safe_timesteps = np.linspace(1000, timesteps, 10, dtype=int)
    safe_timesteps = np.append(safe_timesteps, np.arange(100, 1000, 100, dtype=int))
    for i in range(1, timesteps):
        lbm.step()

        if i%100 == 0 or i == timesteps-1 or i in safe_timesteps:
            print("Timestep " + str(i) + " of " + str(timesteps))
            
            if i in safe_timesteps:
                lbm.update_velocity_field()
                simulated_velocity_field_tCyx.append(lbm.get_velocity_field_Cyx(False))
                indices.append(i)

    max_velocity = np.max(np.array(simulated_velocity_field_tCyx)[:, 0, :, :])

    analytical_viscosity = (1/3) * (1/omega - 0.5)
    analytical_solution_y = calc_poiseuille_flow_analytical_solution(lbm.height-2, lbm.width-2, inlet_pressure, outlet_pressure, analytical_viscosity, density=1)

    # plot the velocity profile at L_x / 2 over time
    lx2 = int(lbm.width/2)
    fig_velocity_profile_Lx2 = plt.figure()
    ax_vel_profile_Lx2 = fig_velocity_profile_Lx2.add_subplot(111)
    ax_vel_profile_Lx2.set_xlabel("Velocity")
    ax_vel_profile_Lx2.set_ylabel("y")
    # ax_vel_profile_Lx2.set_ylim(0, lbm.height)
    y = np.arange(lbm.height-1, -1, -1)
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(indices) + 1))
    for step, (i, simulated_velocity_field_Cyx) in enumerate(zip(indices, simulated_velocity_field_tCyx)):
        ax_vel_profile_Lx2.plot(np.array(simulated_velocity_field_Cyx)[0, :, lx2], y, label="t = " + str(i), color=colors[step])

    ax_vel_profile_Lx2.plot(analytical_solution_y, np.arange(1, lbm.height-1), label="Analytical solution", color="black", linestyle="--")
    ax_vel_profile_Lx2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3, labelspacing = 1)
    plt.tight_layout()
    plt.savefig("PoiseuilleFlowResults/PoiseuilleFlow_VelocityProfile_Lx2_over_y{}_x_{}_T{}_omega{}.png".format(grid_size_y, grid_size_x, timesteps, omega), bbox_inches='tight')
    plt.show()


def calc_poiseuille_flow_analytical_solution(pipe_height : int, pipe_length : int, inlet_pressure : float, outlet_pressure : float, viscosity : float, density : float):
    """
    Calculate the analytical solution for Poiseuille flow.

    Parameters
    ----------
    pipe_height : int
        The height of the pipe.
    inlet_pressure : float
        The pressure at the inlet.
    outlet_pressure : float
        The pressure at the outlet.
    viscosity : float
        The viscosity of the fluid.

    Returns
    -------
    analytical_solution : np.ndarray
        The analytical solution for the velocity profile.
    """
    dynamic_viscosity = viscosity * density
    pressure_difference = outlet_pressure - inlet_pressure
    derivative = pressure_difference / pipe_length

    y = np.arange(pipe_height)
    analytical_solution = 0.5 * (1/dynamic_viscosity) * derivative * y * (y - (pipe_height - 1))
    return analytical_solution



if __name__ == "__main__":

    grid_size_x = 100
    grid_size_y = 100
    omega = 0.3
    timesteps = 10000
    poiseuille_flow(grid_size_x, grid_size_y, omega, timesteps)