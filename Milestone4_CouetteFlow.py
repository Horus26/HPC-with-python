import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from LBM import LBM

def couette_flow(grid_size_x : int, grid_size_y : int, omega : float, timesteps : int):
    """
    Simulate Couette flow.

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
    
    top_boundary_velocity = 0.1
    # boundary_conditions = {"bottom": "bounce_back", "top": "moving_wall", "left": "periodic", "right": "periodic"}
    boundary_conditions = {"bottom": "bounce_back", "top": "moving_wall", "left": "periodic", "right": "periodic"}
    boundary_velocities = {"bottom": 0.0, "top": top_boundary_velocity, "left": 0.0, "right": 0.0}
    # boundary_velocities = {"bottom": top_boundary_velocity, "top": top_boundary_velocity, "left": top_boundary_velocity, "right": top_boundary_velocity}
    # boundary_velocities = None
    initial_velocity_field_Cyx = None

    # initialize LBM
    lbm = LBM(grid_size_x, grid_size_y, omega, inital_velocity_field_Cyx=initial_velocity_field_Cyx, boundary_conditions=boundary_conditions, boundary_velocities=boundary_velocities)

    # simulate Couette flow
    simulated_velocity_field_tCyx = [lbm.get_velocity_field_Cyx()]
    indices = [0]
    print("Simulating Couette flow...")

    # safe at 10 linearly spaced timesteps
    safe_timesteps = np.linspace(1, timesteps-1, 10).astype(int)
    safe_timesteps = np.append(safe_timesteps, [10, 50, 100])
    for i in range(1, timesteps):
        lbm.step()

        if i%100 == 0 or i in safe_timesteps:
            print("Timestep " + str(i) + " of " + str(timesteps))
        
            lbm.update_velocity_field()
            simulated_velocity_field_tCyx.append(lbm.get_velocity_field_Cyx())
            indices.append(i)


    print("Plotting results...")
    fig_velocity_over_time = plt.figure()
    ax1 = fig_velocity_over_time.add_subplot(111)
    # ax1.set_title("Velocity field over time")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")
    ax1.set_xlim(-10, lbm.width+10)
    ax1.set_ylim(-10, lbm.height+10)
    x = np.arange(lbm.width)
    y = np.arange(lbm.height)

    # max_velocity = np.max(np.array(simulated_velocity_field_tCyx)[:, 0, :, :])

    # plot strength of velocity field
    for i, simulated_velocity_field_Cyx in zip(indices, simulated_velocity_field_tCyx):
        
        # continue   
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_aspect("equal")
        ax1.set_xlim(-10, lbm.width+10)
        ax1.set_ylim(-10, lbm.height+10)

        # draw the boundaries
        ax1.plot([-0.5, grid_size_x-0.5], [grid_size_y-0.5, grid_size_y-0.5], color="red", label="Moving", linewidth=1)
        ax1.plot([-0.5, grid_size_x-0.5], [-0.5, -0.5], color="black", label = "Fixed", linewidth=1)
        # draw vertical boundaries
        ax1.plot([-0.5, -0.5], [-0.5, grid_size_y-0.5], color="black", linestyle="dashed", label="Periodic", linewidth=1)
        ax1.plot([grid_size_x-0.5, grid_size_x-0.5], [-0.5, grid_size_y-0.5], color="black", linestyle="dashed", linewidth=1)
        # ax1.set_title("Velocity field over time, t = " + str(i))
        velocity_magnitude_field_yx = np.sqrt(simulated_velocity_field_Cyx[0]**2 + simulated_velocity_field_Cyx[1]**2)
        mappable = ax1.imshow(np.flip(velocity_magnitude_field_yx, axis=0), cmap="plasma", origin="lower")

        cbar = fig_velocity_over_time.colorbar(mappable, ax=ax1)
        cbar.set_ticks([0, top_boundary_velocity])
        cbar.solids.set_edgecolor("face")
        cbar.minorticks_on()
        cbar.set_label("Velocity strength")

        ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
            mode="expand", borderaxespad=0, ncol=3, labelspacing = 1)
        plt.tight_layout()
        if i in safe_timesteps:
            plt.savefig("CouetteFlowResults/CouetteFlow_VelocityField_t" + str(i) + ".png")
        plt.pause(0.01)
        cbar.remove()
        ax1.clear()
        
            
    # plot the velocity profile at L_x / 2 over time
    # expectation is a linear profile for t-->infinitiy
    lx2 = int(lbm.width/2)
    fig_velocity_profile_Lx2 = plt.figure()
    ax_vel_profile_Lx2 = fig_velocity_profile_Lx2.add_subplot(111)
    ax_vel_profile_Lx2.set_xlabel("Velocity")
    ax_vel_profile_Lx2.set_ylabel("y")
    ax_vel_profile_Lx2.set_ylim(0, lbm.height-1)
    ax_vel_profile_Lx2.set_xlim(0, top_boundary_velocity)
    # ax_vel_profile_Lx2.set_aspect("equal")
    y = np.arange(grid_size_y - 1, -1, -1) + 0.5
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, np.ceil(timesteps/1000).astype(int) + 1))
    for i, simulated_velocity_field_Cyx in zip(indices, simulated_velocity_field_tCyx):
        if (i % 500 == 0 and i > 0) or i == indices[-1] or i in [10, 50, 100]:
            ax_vel_profile_Lx2.plot(np.array(simulated_velocity_field_Cyx)[0, :, lx2], y, label="t = " + str(i), color=colors[int(i/1000)])

    ax_vel_profile_Lx2.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.savefig("CouetteFlowResults/CouetteFlow_VelocityProfile_Lx2_over_timesteps.png")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    grid_size_x = 100
    grid_size_y = 100
    omega = 0.6
    timesteps = 10000
    couette_flow(grid_size_x, grid_size_y, omega, timesteps)