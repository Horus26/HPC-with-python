import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from LBM import LBM
from time import time

def sliding_lid(grid_size_x : int, grid_size_y : int, omega : float, timesteps : int, lid_velocity : float, save_figures : bool):
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
    lid_velocity : float
        The velocity of the lid.
    save_figures : bool
        Whether to save the figures or not.

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

    safe_timesteps = [1, 10, 50, 100, 500]
    safe_timesteps = np.append(safe_timesteps, np.arange(500, 10001, 500, dtype=int))
    fig_velocity_animation = plt.figure()
    ax = fig_velocity_animation.add_subplot(111)
    cbar = fig_velocity_animation.colorbar(ax=ax, mappable=matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, lid_velocity), cmap="viridis"))
    cbar.set_label("Velocity magnitude")
    cbar.set_ticks(np.linspace(0, lid_velocity, 5))
    
    # start time
    start_time = time()
    for i in range(1, timesteps):
        lbm.step()

        if i%10000 == 0:
            print("Timestep " + str(i) + " of " + str(timesteps))
            # ax.clear()
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_aspect("equal")
            # ax.set_xlim(-10, lbm.width+10)
            # ax.set_ylim(-10, lbm.height+10)
            # ax.set_title("Velocity streamplot at t = " + str(i))
            # lbm.update_velocity_field()
            # velocity_field = lbm.get_velocity_field_Cyx(False)
            # velocity_field = np.flip(velocity_field, axis=1)
            # u_x = velocity_field[0]
            # u_y = velocity_field[1]
            
            # ax.streamplot(np.arange(0, lbm.width), np.arange(0, lbm.height), u_x, u_y, color=np.sqrt(u_x**2 + u_y**2), density=1.5, norm=plt.Normalize(0, lid_velocity))
            # ax.plot([-0.5, lbm.width - 0.5], [lbm.height-1.5, lbm.height-1.5], color="red", label="Moving wall")
            # ax.plot([-0.5, lbm.width - 0.5], [0.5, 0.5], color="black", label = "Fixed wall")
            # # draw vertical boundaries
            # ax.plot([-0.5, -0.5], [0.5, lbm.height- 1.5], color="black")
            # ax.plot([lbm.width-0.5, lbm.width-0.5], [0.5, lbm.height - 1.5], color="black")

            # # plt.pause(0.1)
            # plt.savefig("SlidingLidResults/Sliding_Lid_Velocity_Streamplot_" + str(i) + ".png")
            # plt.tight_layout()

        # if i%10000 == 0 or i == timesteps-1 or i in safe_timesteps:
        #     print("Timestep " + str(i) + " of " + str(timesteps))
        
        #     lbm.update_velocity_field()
        #     simulated_velocity_field_tCyx.append(lbm.get_velocity_field_Cyx(False))
        #     indices.append(i)

    # end time
    end_time = time()
    print("Time elapsed: " + str(end_time - start_time))

    return


    # store last velocity field
    np.save("SlidingLidResults/Sliding_Lid_Velocity_Field_RE" + str(reynolds_number) + ".npy", simulated_velocity_field_tCyx[-1])

    # plot velocity field streamplot
    print("Plotting results...")

    # return

    for i, simulated_velocity_field_Cyx in zip(indices, simulated_velocity_field_tCyx):    
        ax.clear()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(-10, lbm.width+10)
        ax.set_ylim(-10, lbm.height+10)
        ax.set_title("Velocity streamplot at t = " + str(i))
        # lbm.update_velocity_field()
        velocity_field = simulated_velocity_field_Cyx
        velocity_field = np.flip(velocity_field, axis=1)
        u_x = velocity_field[0]
        u_y = velocity_field[1]
        
        ax.streamplot(np.arange(0, lbm.width), np.arange(0, lbm.height), u_x, u_y, color=np.sqrt(u_x**2 + u_y**2), density=1.5, norm=plt.Normalize(0, lid_velocity))
        ax.plot([-0.5, lbm.width - 0.5], [lbm.height-1.5, lbm.height-1.5], color="red", label="Moving wall")
        ax.plot([-0.5, lbm.width - 0.5], [0.5, 0.5], color="black", label = "Fixed wall")
        # draw vertical boundaries
        ax.plot([-0.5, -0.5], [0.5, lbm.height- 1.5], color="black")
        ax.plot([lbm.width-0.5, lbm.width-0.5], [0.5, lbm.height - 1.5], color="black")

        plt.tight_layout()
        if save_figures:
            plt.savefig("SlidingLidResults/Sliding_Lid_Velocity_Streamplot_T" + str(i) + "_RE" + str(reynolds_number) + ".png")
        plt.pause(0.1)

    plt.show()


if __name__ == "__main__":

    grid_size_x = 300
    grid_size_y = 300
    timesteps = 100000

    reynolds_number = 1000
    characteristic_length = grid_size_x
    characteristic_velocity = 0.3
    kinematic_viscosity = characteristic_length * characteristic_velocity / reynolds_number
    omega = 1.0 / (3 * kinematic_viscosity + 0.5)

    save_figures = True
    print("Kinematic viscosity = " + str(kinematic_viscosity))
    print("omega = " + str(omega))

    sliding_lid(grid_size_x, grid_size_y, omega, timesteps, characteristic_velocity, save_figures)

    # Parallel 156 CPUs for 300x300: Time elapsed: 69.806691849
    # Serial 300x300: Time elapsed: 2500.722847223282

    # # load numpy results
    # simulated_velocity_field_tCyx = np.load("SlidingLidResults/Sliding_Lid_Velocity_Field_RE1000.npy")
    # # remove boundary
    # simulated_velocity_field_tCyx = simulated_velocity_field_tCyx[:, 1:-1, 1:-1]
    # # load parallel result
    # simulated_velocity_field_tCyx_parallel = np.load("SlidingLidResults/PARALLEL_Sliding_Lid_Velocity_Field_T_100000_RE_1000_FULL.npy")

    # # calculate difference
    # difference = np.abs(simulated_velocity_field_tCyx - simulated_velocity_field_tCyx_parallel)
    # print("Maximum difference: " + str(np.max(difference)))

    # 300 x 300 --> linear up to 100 cpus, ca. then saturates
    # omega 1,7 , U_x=0.1 (when using smaller omega, make L larger)
    # --> kinematic viscosity = 1/34
    # --> Reynolds number ca. 1000

    # bwunicluster
    # 1. ask for processes and memory


    #sbatch --> ask for resources
    # each node has 40 processors
    # partition is job queue (multiple is standard, dev_multiple is developer queue for testing)

    # load modules example
    # module load compiler/intel/2021.2.0
    # module load mpi/impi/2021.2.0
    # module load numlib/mkl/2021.2.0
    # like this
    # mpirun -n 160 python3 SlidingLid.py

    # use python3
    # python3 -m pip pip install mpi4py --user
    # if something is missing install in local environment

    # ${SLURM_NTASKS} --> total number of processes (= nodes * ntasks-per-node)