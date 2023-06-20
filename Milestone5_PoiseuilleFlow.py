import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from LBM import LBM

def poiseuille_flow(grid_size_x : int, grid_size_y : int, omega : float, timesteps : int):
    """
    Simulate Poisuille flow.

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
    inlet_pressure = 0.1
    outlet_pressure = 0.05
    
    boundary_conditions = {"bottom": "bounce_back", "top": "bounce_back", "left": "periodic", "right": "periodic"}
    boundary_pressure = {"bottom": None, "top": None, "left": inlet_pressure, "right": outlet_pressure, "output": "right", "input": "left"}

    # initialize LBM
    lbm = LBM(grid_size_x, grid_size_y, omega, boundary_conditions=boundary_conditions, boundary_pressure=boundary_pressure)

    # simulate Poiseuille flow
    print("Simulating Poiseuille flow...")
    simulated_velocity_field_tCyx = [lbm.get_velocity_field_Cyx(False)]
    indices = [0]
    for i in range(1, timesteps):
        lbm.step()

        if i%100 == 0:# or i in safe_timesteps:
            print("Timestep " + str(i) + " of " + str(timesteps))
        
            lbm.update_velocity_field()
            simulated_velocity_field_tCyx.append(lbm.get_velocity_field_Cyx(False))
            indices.append(i)

    max_velocity = np.max(np.array(simulated_velocity_field_tCyx)[:, 0, :, :])


    # plot the velocity profile at L_x / 2 over time
    # expectation is a linear profile for t-->infinitiy
    lx2 = int(lbm.width/2)
    fig_velocity_profile_Lx2 = plt.figure()
    ax_vel_profile_Lx2 = fig_velocity_profile_Lx2.add_subplot(111)
    ax_vel_profile_Lx2.set_xlabel("Velocity")
    ax_vel_profile_Lx2.set_ylabel("y")
    ax_vel_profile_Lx2.set_ylim(0, lbm.height-1)
    # ax_vel_profile_Lx2.set_xlim(0, max_velocity)
    # ax_vel_profile_Lx2.set_aspect("equal")
    y = np.arange(lbm.height-2, 0, -1)
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, np.ceil(timesteps/1000).astype(int) + 1))
    for i, simulated_velocity_field_Cyx in zip(indices, simulated_velocity_field_tCyx):
        if i % 1000 == 0 or i == indices[-1]:
            ax_vel_profile_Lx2.plot(np.array(simulated_velocity_field_Cyx)[0, 1:-1, lx2], y, label="t = " + str(i), color=colors[int(i/1000)])

    ax_vel_profile_Lx2.legend()
    plt.savefig("PoiseuilleFlowResults/PoiseuilleFlow_VelocityProfile_Lx2_over_timesteps.png")
    # plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.show()


def calc_poiseuille_flow_analytical_solution():
    pass


if __name__ == "__main__":

    grid_size_x = 200
    grid_size_y = 100
    omega = 0.6
    timesteps = 10000
    poiseuille_flow(grid_size_x, grid_size_y, omega, timesteps)