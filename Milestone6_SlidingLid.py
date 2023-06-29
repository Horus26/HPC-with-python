import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from LBM import LBM

def sliding_lid(grid_size_x : int, grid_size_y : int, omega : float, timesteps : int, lid_velocity : float):
    """
    Simulate Sliding Lid.

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
    top_boundary_velocity = lid_velocity

    boundary_conditions = {"bottom": "bounce_back", "top": "moving_wall", "left": "bounce_back", "right": "bounce_back"}
    boundary_velocities = {"bottom": 0.0, "top": top_boundary_velocity, "left": 0.0, "right": 0.0}

    # initialize LBM
    lbm = LBM(grid_size_x, grid_size_y, omega, boundary_conditions=boundary_conditions, boundary_velocities=boundary_velocities)

    # simulate Sliding Lid
    print("Simulating Sliding Lid...")
    simulated_velocity_field_tCyx = [lbm.get_velocity_field_Cyx(False)]
    indices = [0]

    # change matplotlib backend to show animation (not TkAgg)
    # matplotlib.use("Qt5Agg")


    fig_velocity_animation = plt.figure()
    ax = fig_velocity_animation.add_subplot(111)
    cbar = fig_velocity_animation.colorbar(ax=ax, mappable=matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, lid_velocity), cmap="viridis"))
    cbar.set_label("Velocity magnitude")
    cbar.set_ticks(np.linspace(0, lid_velocity, 5))
    for i in range(1, timesteps):
        lbm.step()

        if i%(int(timesteps/200)) == 0:
            print("Timestep " + str(i) + " of " + str(timesteps))
            ax.clear()
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            ax.set_xlim(-10, lbm.width+10)
            ax.set_ylim(-10, lbm.height+10)
            ax.set_title("Velocity streamplot at t = " + str(i))
            lbm.update_velocity_field()
            velocity_field = lbm.get_velocity_field_Cyx(False)
            velocity_field = np.flip(velocity_field, axis=1)
            u_x = velocity_field[0]
            u_y = velocity_field[1]
            
            ax.streamplot(np.arange(0, lbm.width), np.arange(0, lbm.height), u_x, u_y, color=np.sqrt(u_x**2 + u_y**2), density=1.5, norm=plt.Normalize(0, lid_velocity))
            ax.plot([-0.5, lbm.width - 0.5], [lbm.height-1.5, lbm.height-1.5], color="red", label="Moving wall")
            ax.plot([-0.5, lbm.width - 0.5], [0.5, 0.5], color="black", label = "Fixed wall")
            # draw vertical boundaries
            ax.plot([-0.5, -0.5], [0.5, lbm.height- 1.5], color="black")
            ax.plot([lbm.width-0.5, lbm.width-0.5], [0.5, lbm.height - 1.5], color="black")

            # plt.pause(0.1)
            plt.savefig("SlidingLidResults/Sliding_Lid_Velocity_Streamplot.png")
            plt.tight_layout()

        if i%10000 == 0 or i == timesteps-1:# or i in safe_timesteps:
            print("Timestep " + str(i) + " of " + str(timesteps))
        
            lbm.update_velocity_field()
            simulated_velocity_field_tCyx.append(lbm.get_velocity_field_Cyx(False))
            indices.append(i)

    # plot velocity field streamplot
    print("Plotting results...")
    
    fig_velocity_over_time = plt.figure()
    ax1 = fig_velocity_over_time.add_subplot(111)
    safe_timesteps = np.linspace(0, timesteps-1, 10, dtype=int)

    # for i, simulated_velocity_field_Cyx in zip(indices, simulated_velocity_field_tCyx):     
    #     ax1.set_xlabel("x")
    #     ax1.set_ylabel("y")
    #     ax1.set_aspect("equal")
    #     ax1.set_xlim(-10, lbm.width+10)
    #     ax1.set_ylim(-10, lbm.height+10)
    #     ax1.set_title("Velocity streamplot at t = " + str(i))
    #     ax1.streamplot(np.arange(0, lbm.width), np.arange(0, lbm.height), simulated_velocity_field_Cyx[0], simulated_velocity_field_Cyx[1], color="black", density=1, linewidth=0.5, arrowsize=0.5)
    #     # draw the boundaries
    #     ax1.plot([-0.5, lbm.width - 0.5], [lbm.height-1.5, lbm.height-1.5], color="red", label="Moving wall")
    #     ax1.plot([-0.5, lbm.width - 0.5], [0.5, 0.5], color="black", label = "Fixed wall")
    #     # draw vertical boundaries
    #     ax1.plot([-0.5, -0.5], [0.5, lbm.height- 1.5], color="black", linestyle="dashed", label="Periodic boundaries")
    #     ax1.plot([lbm.width-0.5, lbm.width-0.5], [0.5, lbm.height - 1.5], color="black", linestyle="dashed")
    #     ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    #         mode="expand", borderaxespad=0, ncol=3)
    #     plt.tight_layout()
    #     if i in safe_timesteps:
    #         plt.savefig("SlidingLidResults/Sliding_Lid_Velocity_Streamplot_" + str(i) + ".png")
    #     plt.pause(0.01)
    #     ax1.clear()
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")
    ax1.set_xlim(-10, lbm.width+10)
    ax1.set_ylim(-10, lbm.height+10)
    ax1.set_title("Velocity streamplot at t = " + str(i))

    u_x = simulated_velocity_field_tCyx[-1][0]
    u_y = simulated_velocity_field_tCyx[-1][1]
    ax1.streamplot(np.arange(0, lbm.width), np.arange(0, lbm.height), u_x, u_y, color=np.sqrt(u_x**2 + u_y**2), density=1.5, norm=plt.Normalize(0, lid_velocity))
    # add a colorbar
    cbar = fig_velocity_over_time.colorbar(ax=ax1, mappable=matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, lid_velocity), cmap="viridis"))
    cbar.set_label("Velocity magnitude")
    cbar.set_ticks(np.linspace(0, lid_velocity, 5))


    # draw the boundaries
    ax1.plot([-0.5, lbm.width - 0.5], [lbm.height-1.5, lbm.height-1.5], color="red", label="Moving wall")
    ax1.plot([-0.5, lbm.width - 0.5], [0.5, 0.5], color="black", label = "Fixed wall")
    # draw vertical boundaries
    ax1.plot([-0.5, -0.5], [0.5, lbm.height- 1.5], color="black", linestyle="dashed")
    ax1.plot([lbm.width-0.5, lbm.width-0.5], [0.5, lbm.height - 1.5], color="black", linestyle="dashed")
    ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        mode="expand", borderaxespad=0, ncol=3)
    plt.tight_layout()
    plt.savefig("SlidingLidResults/Sliding_Lid_Velocity_Streamplot_Single_" + str(timesteps) + ".png")
    plt.show()


if __name__ == "__main__":

    grid_size_x = 300
    grid_size_y = 300
    timesteps = 10000

    reynolds_number = 1000
    characteristic_length = grid_size_x
    characteristic_velocity = 0.3
    kinematic_viscosity = characteristic_length * characteristic_velocity / reynolds_number
    omega = 1.0 / (3 * kinematic_viscosity + 0.5)

    print("Kinematic viscosity = " + str(kinematic_viscosity))
    print("omega = " + str(omega))

    sliding_lid(grid_size_x, grid_size_y, omega, timesteps, characteristic_velocity)