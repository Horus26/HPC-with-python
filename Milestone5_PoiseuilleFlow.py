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

    # define inlet and outlet pressure
    base_pressure = 1 / 3
    pressure_difference = 0.001
    inlet_pressure = base_pressure + pressure_difference
    outlet_pressure = base_pressure - pressure_difference
    
    # prepare lbm parameters
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
    last_density_field = None
    for i in range(1, timesteps):
        lbm.step()

        if i%100 == 0 or i == timesteps-1 or i in safe_timesteps:
            print("Timestep " + str(i) + " of " + str(timesteps))
            
            if i in safe_timesteps or i == timesteps-1:
                lbm.update_velocity_field()
                simulated_velocity_field_tCyx.append(lbm.get_velocity_field_Cyx(False))
                indices.append(i)
        # store last density field
        if i == timesteps-1:
            lbm.update_density_field()
            last_density_field = lbm.get_density_field_yx()

    max_velocity = np.max(np.array(simulated_velocity_field_tCyx)[:, 0, :, :])

    # calculate analytical solution
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
    # plot analytical solution
    ax_vel_profile_Lx2.plot(analytical_solution_y, np.arange(1, lbm.height-1), label="Analytical solution", color="black", linestyle="--")
    ax_vel_profile_Lx2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3, labelspacing = 1)
    plt.tight_layout()
    plt.savefig("PoiseuilleFlowResults/PoiseuilleFlow_VelocityProfile_Lx2_over_y{}_x_{}_T{}_omega{}.png".format(grid_size_y, grid_size_x, timesteps, omega), bbox_inches='tight')

    # plot velocity vectors for steady state velocity field
    steady_state_velocity_field_Cyx = simulated_velocity_field_tCyx[-1]
    fig_velocity_vectors = plt.figure()
    ax_velocity_vectors = fig_velocity_vectors.add_subplot(111)
    ax_velocity_vectors.set_xlabel("x")
    ax_velocity_vectors.set_ylabel("y")
    ax_velocity_vectors.set_aspect("equal")
    ax_velocity_vectors.set_xlim(-10, lbm.width+10)
    ax_velocity_vectors.set_ylim(-10, lbm.height+10)
    ax_velocity_vectors.set_title("Velocity vectors for steady state velocity field")
    steady_state_velocity_field_Cyx = np.flip(steady_state_velocity_field_Cyx, axis=1)
    u_x = steady_state_velocity_field_Cyx[0][1:-1, 1:-1]
    u_y = steady_state_velocity_field_Cyx[1][1:-1, 1:-1]
    ax_velocity_vectors.streamplot(np.arange(1, lbm.width-1), np.arange(1, lbm.height-1), u_x, u_y, color=np.sqrt(u_x**2 + u_y**2), density=1.2, norm=plt.Normalize(0, max_velocity))
    ax_velocity_vectors.plot([-0.5, lbm.width - 0.5], [lbm.height-1.5, lbm.height-1.5], color="black")
    ax_velocity_vectors.plot([-0.5, lbm.width - 0.5], [0.5, 0.5], color="black", label = "Fixed wall")
    # draw vertical boundaries
    ax_velocity_vectors.plot([-0.5, -0.5], [0.5, lbm.height- 1.5], color="black", label="Periodic pressure gradient", linestyle="dashed")
    ax_velocity_vectors.plot([lbm.width-0.5, lbm.width-0.5], [0.5, lbm.height - 1.5], color="black", linestyle="dashed")
    # cbar
    cbar = fig_velocity_vectors.colorbar(ax=ax_velocity_vectors, mappable=matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, max_velocity), cmap="viridis"))
    cbar.set_label("Velocity strength")
    ax_velocity_vectors.legend(bbox_to_anchor=(0.45, 0), loc="lower center",
                bbox_transform=fig_velocity_vectors.transFigure, ncol=3)
    plt.tight_layout()
    plt.savefig("PoiseuilleFlowResults/PoiseuilleFlow_VelocityVectors_{}_x_{}_T{}_omega{}.png".format(grid_size_y, grid_size_x, timesteps, omega), bbox_inches='tight')
    
    # plot the last density field at L_y / 2
    ly2 = int(lbm.height/2)
    fig_density_Ly2 = plt.figure()
    ax_density_Ly2 = fig_density_Ly2.add_subplot(111)
    ax_density_Ly2.set_xlabel("x")
    ax_density_Ly2.set_ylabel("Density")
    x = np.arange(0, lbm.width-2)
    ax_density_Ly2.plot(x, last_density_field[ly2, :])
    plt.tight_layout()
    plt.savefig("PoiseuilleFlowResults/PoiseuilleFlow_Density_Ly2_over_x{}_y_{}_T{}_omega{}.png".format(grid_size_x, grid_size_y, timesteps, omega), bbox_inches='tight') 
    
    plt.show()

    # calculate area of velocity profile at inlet
    area = 0
    for i in range(1, lbm.height-1):
        area += simulated_velocity_field_tCyx[-1][0][i][1]
    print("Area of velocity profile at inlet: " + str(area))

    # area at middle of pipe
    area = 0
    for i in range(1, lbm.height-1):
        area += simulated_velocity_field_tCyx[-1][0][i][int(lbm.width/2)]
    print("Area of velocity profile at middle of pipe: " + str(area))


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
    density : float
        The density of the fluid.

    Returns
    -------
    analytical_solution : np.ndarray
        The analytical solution for the velocity profile.
    """
    dynamic_viscosity = viscosity * density
    pressure_difference = inlet_pressure - outlet_pressure
    derivative = pressure_difference / pipe_length

    y = np.arange(pipe_height)
    analytical_solution = -0.5 * (1/dynamic_viscosity) * derivative * y * (y - (pipe_height - 1))
    return analytical_solution



if __name__ == "__main__":
    grid_size_x = 140
    grid_size_y = 60
    omega = 0.5
    timesteps = 20000
    poiseuille_flow(grid_size_x, grid_size_y, omega, timesteps)