import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from LBM import LBM

def couette_flow():
    """
    Simulate Couette flow.
    """
    
    # initialize LBM
    grid_size_x = 10
    grid_size_y = 10
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
    for i in range(timesteps):
        lbm.step()
        lbm.update_velocity_field()
        simulated_velocity_field_tCyx.append(lbm.velocity_field_Cyx)

    # plot velocity field streamplot
    fig_velocity_over_time = plt.figure()
    ax1 = fig_velocity_over_time.add_subplot(111)
    ax1.set_title("Velocity field over time")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")
    ax1.set_xlim(0, lbm.width)
    # ax1.set_ylim(lbm.height, 0)
    x = np.arange(lbm.width)
    y = np.arange(lbm.height)
    X, Y = np.meshgrid(x,y)

    analytical_solution_velocity_x = top_boundary_velocity * np.arange(lbm.height-2, 0, -1) / (lbm.height-2)
    print(analytical_solution_velocity_x)
    simulated_velocity_x = np.array(simulated_velocity_field_tCyx)[-1, 0, :, :]
    print(simulated_velocity_x[1:-1, 4])
    # plot velocity profile 
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(analytical_solution_velocity_x / top_boundary_velocity, np.arange(lbm.height-2, 0, -1) / (lbm.height -2), label="Analytical solution")
    # ax.plot(simulated_velocity_x[1:-1, 4] / top_boundary_velocity, np.arange(lbm.height-2, 0, -1) / (lbm.height -2), label="Simulated solution")
    # ax.set_xlabel("Velocity / U_x")
    # ax.set_ylabel("y/H")
    # ax.set_title("Velocity profile")
    # ax.legend()

    # for i in range(timesteps):        
    #     # plot strength of velocity field with imshow
    #     if i%99 == 0:
    #         ax1.set_title("Velocity field over time, t = " + str(i))
    #         velocity_magnitude_field_yx = np.sqrt(simulated_velocity_field_tCyx[i][0]**2 + simulated_velocity_field_tCyx[i][1]**2)
    #         mappable = ax1.imshow(velocity_magnitude_field_yx, cmap="hot", origin="upper")
    #         cbar = fig_velocity_over_time.colorbar(mappable, ax=ax1, extend='both')
    #         cbar.minorticks_on()
    #         cbar.set_label("Velocity strength")
    #         plt.tight_layout()
    #         plt.pause(0.01)
    #         cbar.remove()
    
    # plot the velocity vectors for last timestep
    fig_vectors = plt.figure()
    ax_vectors = fig_vectors.add_subplot(111)
    # ax_vectors.set_title("Velocity field at t = " + str(timesteps))
    ax_vectors.set_xlabel("x")
    ax_vectors.set_ylabel("y")
    ax_vectors.set_aspect("equal")
    ax_vectors.set_xlim(-0.5, lbm.width)
    ax_vectors.set_ylim(lbm.height, -0.5)
    ax_vectors.quiver(X, Y, simulated_velocity_field_tCyx[-1][0], simulated_velocity_field_tCyx[-1][1])
    # draw the boundaries
    ax_vectors.plot([0.5, lbm.width - 1.5], [0.5, 0.5], color="red", label="Moving wall")
    ax_vectors.plot([0.5, lbm.width - 1.5], [lbm.height-1.5, lbm.height-1.5], color="black", label = "Fixed wall")
    # draw vertical boundaries
    ax_vectors.plot([0.5, 0.5], [0.5, lbm.height- 1.5], color="black", linestyle="dashed", label="Periodic boundaries")
    ax_vectors.plot([lbm.width-1.5, lbm.width-1.5], [0.5, lbm.height - 1.5], color="black", linestyle="dashed")
    # draw all grid points
    ax_vectors.scatter(X, Y, color="green", s=4, label="Grid points outside boundaries")
    # make grid points outside the boundaries different color
    ax_vectors.scatter(X[1:-1, 1:-1], Y[1:-1, 1:-1], color="blue", s=4, label="Grid points")
    # do not show axes
    ax_vectors.set_axis_off()
    # plt.tight_layout()
    ax_vectors.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.show()

if __name__ == "__main__":
    couette_flow()
