import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from LBM import LBM

def couette_flow():
    """
    Simulate Couette flow.
    """
    
    # initialize LBM
    grid_size_x = 200
    grid_size_y = 100
    omega = 0.6

    top_boundary_velocity = 0.1
    boundary_conditions = {"bottom": "bounce_back", "top": "moving_wall", "left": "periodic", "right": "periodic"}
    # boundary_conditions = {"bottom": "moving_wall", "top": "moving_wall", "left": "moving_wall", "right": "moving_wall"}
    boundary_velocities = {"bottom": 0.0, "top": top_boundary_velocity, "left": 0.0, "right": 0.0}
    # boundary_velocities = {"bottom": top_boundary_velocity, "top": top_boundary_velocity, "left": top_boundary_velocity, "right": top_boundary_velocity}
    
    initial_velocity_field_Cyx = None

    lbm = LBM(grid_size_x, grid_size_y, omega, inital_velocity_field_Cyx=initial_velocity_field_Cyx, boundary_conditions=boundary_conditions, boundary_velocities=boundary_velocities)

    # simulate Couette flow
    timesteps = 5000
    simulated_velocity_field_tCyx = []
    simulated_velocity_field_tCyx.append(lbm.velocity_field_Cyx)
    print("Simulating Couette flow...")
    for i in range(timesteps):
        lbm.step()
        lbm.update_velocity_field()
        simulated_velocity_field_tCyx.append(lbm.velocity_field_Cyx)

    # # plot velocity field streamplot
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
    X, Y = np.meshgrid(x,y)

    analytical_solution_velocity_x = top_boundary_velocity * np.arange(lbm.height-2, 0, -1) / (lbm.height-2)
    simulated_velocity_x = np.array(simulated_velocity_field_tCyx)[-1, 0, :, :]
    max_velocity = np.max(np.array(simulated_velocity_field_tCyx)[:, 0, :, :])
    safe_timesteps = [1, 10, 100, 1000, timesteps-1]

    # plot strength of velocity field
    for i in range(timesteps):        
        if i%50 == 0 or i in safe_timesteps:
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_aspect("equal")
            ax1.set_xlim(-10, lbm.width+10)
            ax1.set_ylim(-10, lbm.height+10)
            # ax1.set_title("Velocity field over time, t = " + str(i))
            velocity_magnitude_field_yx = np.sqrt(simulated_velocity_field_tCyx[i][0]**2 + simulated_velocity_field_tCyx[i][1]**2)
            mappable = ax1.imshow(np.flip(velocity_magnitude_field_yx, axis=0), cmap="hot", origin="lower")
            cbar = fig_velocity_over_time.colorbar(mappable, ax=ax1, extend='both')
            cbar.set_ticks([0, max_velocity])
            cbar.minorticks_on()
            cbar.set_label("Velocity strength")
            # draw the boundaries
            ax1.plot([-0.5, lbm.width - 0.5], [lbm.height-1.5, lbm.height-1.5], color="red", label="Moving wall")
            ax1.plot([-0.5, lbm.width - 0.5], [0.5, 0.5], color="black", label = "Fixed wall")
            # draw vertical boundaries
            ax1.plot([-0.5, -0.5], [0.5, lbm.height- 1.5], color="black", linestyle="dashed", label="Periodic boundaries")
            ax1.plot([lbm.width-0.5, lbm.width-0.5], [0.5, lbm.height - 1.5], color="black", linestyle="dashed")
            ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
            plt.tight_layout()
            if i in safe_timesteps:
                plt.savefig("CouetteFlowResults/CouetteFlow_VelocityField_t" + str(i) + ".png")
            plt.pause(0.01)
            cbar.remove()
            ax1.clear()
    

    def plot_velocity_vectors(timestep, lbm : LBM, ax_vectors, X, Y, simulated_velocity_field_tCyx, max_velocity):
        ax_vectors.set_xlabel("x")
        ax_vectors.set_ylabel("y")
        ax_vectors.set_aspect("equal")
        ax_vectors.set_title("Velocity field at t = " + str(timestep))
        ax_vectors.set_xlim(-0.5, lbm.width)
        ax_vectors.set_ylim(lbm.height, -0.5)
        data_timestep = np.array(simulated_velocity_field_tCyx)[timestep, :, :, :]
        ax_vectors.quiver(X, Y, data_timestep[0], data_timestep[1], scale=max_velocity*10)
        # draw the boundaries
        ax_vectors.plot([-0.5, lbm.width - 0.5], [0.5, 0.5], color="red", label="Moving wall")
        ax_vectors.plot([-0.5, lbm.width - 0.5], [lbm.height-1.5, lbm.height-1.5], color="black", label = "Fixed wall")
        # draw vertical boundaries
        ax_vectors.plot([-0.5, -0.5], [0.5, lbm.height- 1.5], color="black", linestyle="dashed", label="Periodic boundaries")
        ax_vectors.plot([lbm.width-0.5, lbm.width-0.5], [0.5, lbm.height - 1.5], color="black", linestyle="dashed")
        # draw all grid points
        ax_vectors.scatter(X, Y, color="green", s=4, label="Grid points outside boundaries")
        # make grid points outside the boundaries different color
        ax_vectors.scatter(X[1:-1, :], Y[1:-1, :], color="blue", s=4, label="Grid points")
        # do not show axes
        # ax_vectors.set_axis_off()
        # plt.tight_layout()
        # ax_vectors.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    # plt.tight_layout(rect=[0, 0, 0.75, 1])
    # plt.show()

    # plot velocity profile four times over all timesteps
    fig_velocity_profile = plt.figure()
    
    for i in range(0, timesteps, int(timesteps/4)):
        ax_vel_profile = fig_velocity_profile.add_subplot(2, 2, int(i/(timesteps/4))+1)
        plot_velocity_vectors(i, lbm, ax_vel_profile, X, Y, simulated_velocity_field_tCyx, max_velocity)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.show()

    # # plot velocity for all y values at x = 1 at 4 timesteps as one horizontal bar plot
    # fig_velocity_profile = plt.figure()
    # ax_vel_profile = fig_velocity_profile.add_subplot(111)
    # ax_vel_profile.set_xlabel("Velocity")
    # ax_vel_profile.set_ylabel("y")
    # ax_vel_profile.set_ylim(0, lbm.height-1)
    # # ax_vel_profile.set_xlim(0, max_velocity)
    # # ax_vel_profile.set_aspect("equal")
    # y = np.arange(lbm.height-1, -1, -1)
    # colors = matplotlib.cm.rainbow(np.linspace(0, 1, 4))
    # for i in range(timesteps-1, 0, -int(timesteps/4)):
    #     ax_vel_profile.barh(y, np.array(simulated_velocity_field_tCyx)[i, 0, :, 1], height=-1, align="edge", label="t = " + str(i), color=colors[int(i/(timesteps/4))])

    # ax_vel_profile.legend()
    # # plt.tight_layout(rect=[0, 0, 0.75, 1])
    # plt.show()

    plt.show()
if __name__ == "__main__":
    couette_flow()


# plot at L_x / 2 the u_x velocity over time --> linear profile