import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from LBM import LBM

def couette_flow(grid_size_x : int, grid_size_y : int, omega : float, timesteps : int, show_animation : bool = False):
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
    show_animation : bool, optional
       Whether to show the animation of the simulated velocity field. The default is False.

    Returns
    -------
    None.
    """
    
    top_boundary_velocity = 0.1
    boundary_conditions = {"bottom": "bounce_back", "top": "moving_wall", "left": "periodic", "right": "periodic"}
    boundary_velocities = {"bottom": 0.0, "top": top_boundary_velocity, "left": 0.0, "right": 0.0}
    initial_velocity_field_Cyx = None

    # initialize LBM
    lbm = LBM(grid_size_x, grid_size_y, omega, inital_velocity_field_Cyx=initial_velocity_field_Cyx, boundary_conditions=boundary_conditions, boundary_velocities=boundary_velocities)

    # simulate Couette flow
    simulated_velocity_field_tCyx = [lbm.get_velocity_field_Cyx(False)]
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
            simulated_velocity_field_tCyx.append(lbm.get_velocity_field_Cyx(False))
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
        velocity_magnitude_field_yx = np.sqrt(simulated_velocity_field_Cyx[0]**2 + simulated_velocity_field_Cyx[1]**2)[1:-1, 1:-1]
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
            plt.savefig("CouetteFlowResults/CouetteFlow_VelocityField_t{}_y{}_x{}_omega{}.png".format(i, grid_size_y, grid_size_x, omega), bbox_inches='tight')
        
        if show_animation:
            plt.pause(0.001)
        cbar.remove()
        ax1.clear()
    
    if not show_animation:
        plt.close(fig_velocity_over_time)
        
            
    # plot the velocity profile at L_x / 2 over time
    # expectation is a linear profile for t-->infinitiy
    lx2 = int(lbm.width/2)
    fig_velocity_profile_Lx2 = plt.figure()
    ax_vel_profile_Lx2 = fig_velocity_profile_Lx2.add_subplot(111)
    ax_vel_profile_Lx2.set_xlabel("Velocity")
    ax_vel_profile_Lx2.set_ylabel("y")
    ax_vel_profile_Lx2.set_ylim(0, lbm.height-1)
    ax_vel_profile_Lx2.set_xlim(0, top_boundary_velocity)
    y = np.arange(grid_size_y - 1, -1, -1)
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(indices) + 1))
    for step, (i, simulated_velocity_field_Cyx) in enumerate(zip(indices, simulated_velocity_field_tCyx)):
        if (i % 1000 == 0 and i > 0) or i == indices[-1] or i in [10, 50, 100]:
            ax_vel_profile_Lx2.plot(np.array(simulated_velocity_field_Cyx)[0, 1:-1, lx2], y, label="t = " + str(i), color=colors[step])
    # plot analytical solution
    analytically_expected_velocity_profile = np.arange(grid_size_y - 1, -1, -1) * (top_boundary_velocity / (grid_size_y-1))
    ax_vel_profile_Lx2.plot(analytically_expected_velocity_profile, y, label="Analytical", color="black", linestyle="dashed")
    ax_vel_profile_Lx2.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.savefig("CouetteFlowResults/CouetteFlow_VelocityProfile_Lx2_over_timesteps_t{}_y{}_x{}_omega{}.png".format(timesteps, grid_size_y, grid_size_x, omega), bbox_inches='tight')
    plt.tight_layout()

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
    ax_velocity_vectors.streamplot(np.arange(1, lbm.width-1), np.arange(1, lbm.height-1), u_x, u_y, color=np.sqrt(u_x**2 + u_y**2), density=1.2, norm=plt.Normalize(0, top_boundary_velocity))
    ax_velocity_vectors.plot([-0.5, lbm.width - 0.5], [lbm.height-1.5, lbm.height-1.5], color="red", label="Moving wall")
    ax_velocity_vectors.plot([-0.5, lbm.width - 0.5], [0.5, 0.5], color="black", label = "Fixed wall")
    # draw vertical boundaries
    ax_velocity_vectors.plot([-0.5, -0.5], [0.5, lbm.height- 1.5], color="black", label="Periodic", linestyle="dashed")
    ax_velocity_vectors.plot([lbm.width-0.5, lbm.width-0.5], [0.5, lbm.height - 1.5], color="black", linestyle="dashed")
    # cbar
    cbar = fig_velocity_vectors.colorbar(ax=ax_velocity_vectors, mappable=matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, top_boundary_velocity), cmap="viridis"))
    cbar.set_label("Velocity strength")
    # ax_velocity_vectors.quiver(np.arange(lbm.width), np.arange(lbm.height), steady_state_velocity_field_Cyx[0], steady_state_velocity_field_Cyx[1])
    ax_velocity_vectors.legend(bbox_to_anchor=(0.45, 0), loc="lower center",
                bbox_transform=fig_velocity_vectors.transFigure, ncol=3)
    plt.savefig("CouetteFlowResults/CouetteFlow_VelocityVectors_t{}_y{}_x{}_omega{}.png".format(timesteps, grid_size_y, grid_size_x, omega), bbox_inches='tight')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    grid_size_x = 140
    grid_size_y = 100
    omega = 1.0
    timesteps = 10000

    # set to true to see an animation of the velocity field for various timesteps while they are being saved
    show_animation = False
    couette_flow(grid_size_x, grid_size_y, omega, timesteps, show_animation)