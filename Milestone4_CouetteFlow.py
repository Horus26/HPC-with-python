import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from LBM import LBM

def couette_flow():
    """
    Simulate Couette flow.
    """
    
    # initialize LBM
    grid_size_x = 20
    grid_size_y = 10
    omega = 1.0

    top_boundary_velocity = 0.1
    boundary_conditions = {"bottom": "bounce_back", "top": "moving_wall", "left": "bounce_back", "right": "bounce_back"}
    boundary_velocities = {"bottom": 0.0, "top": top_boundary_velocity, "left": 0.0, "right": 0.0}
    
    # initial_velocity_field_Cyx = np.zeros((2, grid_size_y, grid_size_x))
    # initial_velocity_field_Cyx[0, :, int(grid_size_x * 0.25)] = 1.5
    # initial_velocity_field_Cyx[0, :, int(grid_size_x * 0.75)] = -1.5
    initial_velocity_field_Cyx = None

    lbm = LBM(grid_size_x, grid_size_y, omega, inital_velocity_field_Cyx=initial_velocity_field_Cyx, boundary_conditions=boundary_conditions, boundary_velocities=boundary_velocities)

    # simulate Couette flow
    timesteps = 1000
    simulated_velocity_field_tCyx = []
    simulated_velocity_field_tCyx.append(lbm.velocity_field_Cyx)
    for i in range(timesteps):
        lbm.step()
        lbm.update_velocity_field()
        # print(np.sqrt(lbm.velocity_field_Cyx[0]**2 + lbm.velocity_field_Cyx[1]**2)[1,:])
        # print(np.sqrt(lbm.velocity_field_Cyx[0]**2 + lbm.velocity_field_Cyx[1]**2)[-2,:])
        # exit(0)
        simulated_velocity_field_tCyx.append(lbm.velocity_field_Cyx)

    # plot velocity field streamplot
    fig_velocity_over_time_quiver = plt.figure()
    ax1 = fig_velocity_over_time_quiver.add_subplot(111)
    ax1.set_title("Velocity field over time")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")
    ax1.set_xlim(0, lbm.width)
    # ax1.set_ylim(lbm.height, 0)
    x = np.arange(lbm.width)
    y = np.arange(lbm.height)
    X, Y = np.meshgrid(x,y)
    for i in range(timesteps):
        # print(X.shape)
        # print(Y.shape)
        # print(simulated_velocity_field_tCyx[i][0].shape)
        # print(simulated_velocity_field_tCyx[i][1].shape)
        # ax1.streamplot(X, Y, simulated_velocity_field_tCyx[i][0], simulated_velocity_field_tCyx[i][1], color="black")
        
        # plot strength of velocity field with imshow
        if i%2 == 0:
            ax1.set_title("Velocity field over time, t = " + str(i))
            velocity_magnitude_field_yx = np.sqrt(simulated_velocity_field_tCyx[i][0]**2 + simulated_velocity_field_tCyx[i][1]**2)
            mappable = ax1.imshow(velocity_magnitude_field_yx, cmap="hot", origin="upper")
            cbar = fig_velocity_over_time_quiver.colorbar(mappable, ax=ax1, extend='both')
            cbar.minorticks_on()
            cbar.set_label("Velocity strength")
            plt.tight_layout()

            plt.pause(0.01)
            cbar.remove()



    plt.show()

if __name__ == "__main__":
    couette_flow()
