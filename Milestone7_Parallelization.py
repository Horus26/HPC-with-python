import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib
from LBM import LBM

def communicate(lbm : LBM, cartcomm, sd):
    """
    Communicate populations between subgrids

    Parameters
    ----------
    lbm : LBM
        LBM object
    cartcomm : MPI.Cartcomm
        Cartesian communicator
    sd : np.array
        Array of send and receive directions

    Returns
    -------
    f_iyx : np.array
        Updated f_iyx
    """

    # decompose sd in to the send and receive directions
    sR,dR,sL,dL,sU,dU,sD,dD = sd
    # cache f
    f_iyx = lbm.f_iyx
    # cache subgrid size including boundary
    subgrid_size_y_with_b = lbm.height
    subgrid_size_x_with_b = lbm.width
    # cache boundary conditions
    boundary_conditions = lbm.boundary_conditions

    # send and receive right
    # get relevant channels of f_iyx and flatten them
    i1 = f_iyx[1, :, -2].copy()
    i5 = f_iyx[5, :, -2].copy()
    i8 = f_iyx[8, :, -2].copy()
    sendbuf = np.array([i1, i5, i8]).flatten()
    # prepare receive buffer
    recvbuf = np.zeros(subgrid_size_y_with_b*3)
    # send and receive
    cartcomm.Sendrecv(sendbuf, dest = dR, recvbuf = recvbuf, source = sR)
    # check if received input is expected/valid (only if not full domain boundary)
    if boundary_conditions["left"] == None:
        # decompose recvbuf into channels and put them into f_iyx
        f_iyx[1, :, 0] = recvbuf[:subgrid_size_y_with_b].copy()
        f_iyx[5, :, 0] = recvbuf[subgrid_size_y_with_b:2*subgrid_size_y_with_b].copy()
        f_iyx[8, :, 0] = recvbuf[2*subgrid_size_y_with_b:].copy()

    # send and receive left
    # get relevant channels of f_iyx and flatten them
    i3 = f_iyx[3, :, 1].copy()
    i6 = f_iyx[6, :, 1].copy()
    i7 = f_iyx[7, :, 1].copy()
    sendbuf = np.array([i3, i6, i7]).flatten()
    recvbuf = np.zeros(subgrid_size_y_with_b*3)
    # send and receive
    cartcomm.Sendrecv(sendbuf, dest = dL, recvbuf = recvbuf, source = sL)
    # check if received input is expected/valid (only if not full domain boundary)
    if boundary_conditions["right"] == None:
        # decompose recvbuf into channels and put them into f_iyx
        f_iyx[3, :, -1] = recvbuf[:subgrid_size_y_with_b].copy()
        f_iyx[6, :, -1] = recvbuf[subgrid_size_y_with_b:2*subgrid_size_y_with_b].copy()
        f_iyx[7, :, -1] = recvbuf[2*subgrid_size_y_with_b:].copy()

    # send and receive up
    # get relevant channels of f_iyx and flatten them
    i2 = f_iyx[2, 1, :].copy()
    i5 = f_iyx[5, 1, :].copy()
    i6 = f_iyx[6, 1, :].copy()
    sendbuf = np.array([i2, i5, i6]).flatten()
    recvbuf = np.zeros(subgrid_size_x_with_b*3)
    # send and receive
    cartcomm.Sendrecv(sendbuf, dest = dU, recvbuf = recvbuf, source = sU)
    # check if received input is expected/valid (only if not full domain boundary)
    if boundary_conditions["bottom"] == None:
        # decompose recvbuf into channels and put them into f_iyx
        f_iyx[2, -1, :] = recvbuf[:subgrid_size_x_with_b]
        f_iyx[5, -1, :] = recvbuf[subgrid_size_x_with_b:2*subgrid_size_x_with_b]
        f_iyx[6, -1, :] = recvbuf[2*subgrid_size_x_with_b:]

    # send and receive down
    # get relevant channels of f_iyx and flatten them
    i4 = f_iyx[4, -2, :].copy()
    i7 = f_iyx[7, -2, :].copy()
    i8 = f_iyx[8, -2, :].copy()
    sendbuf = np.array([i4, i7, i8]).flatten()
    recvbuf = np.zeros(subgrid_size_x_with_b*3)
    # send and receive
    cartcomm.Sendrecv(sendbuf, dest = dD, recvbuf = recvbuf, source = sD)
    # check if received input is expected/valid (only if not full domain boundary)
    if boundary_conditions["top"] == None:
        # decompose recvbuf into channels and put them into f_iyx
        f_iyx[4, 0, :] = recvbuf[:subgrid_size_x_with_b]
        f_iyx[7, 0, :] = recvbuf[subgrid_size_x_with_b:2*subgrid_size_x_with_b]
        f_iyx[8, 0, :] = recvbuf[2*subgrid_size_x_with_b:]
    
    return f_iyx

def domain_decomposition(NX, NY, num_processes):
    """
    Calculate the number of subgrids in x and y direction for a given number of processes

    Parameters
    ----------
    NX : int
        Width of the grid
    NY : int
        Height of the grid
    num_processes : int
        Number of processes

    Returns
    -------
    subgrids_number_X : int
        Number of subgrids in x direction
    subgrids_number_Y : int
        Number of subgrids in y direction
    """
    # Find all dividers of NX and NY
    dividers_x = [d for d in range(1, NX+1) if NX % d == 0]
    dividers_y = [d for d in range(1, NY+1) if NY % d == 0]

    valid_dividers = []

    # only keep dividers that are valid for the given number of processes
    for dx in dividers_x:
        for dy in dividers_y:
            if dx * dy <= num_processes:
                valid_dividers.append((dx, dy))

    # Sort valid dividers based on load balancing
    sorted_dividers = sorted(valid_dividers, key=lambda x: NX/ x[0] + NY / x[1])
    # Take the first divider, which is the one with the best load balancing
    subgrids_number_X, subgrids_number_Y = sorted_dividers[0]

    return subgrids_number_X, subgrids_number_Y

def run_lbm(cartcomm : MPI.Cartcomm, rank, size, NX, NY, subgrids_number_X, subgrids_number_Y, lbm_parameter, timesteps):
    """
    Run LBM for each process. This function initializes the LBM for each process and runs it for the given number of timesteps.
    The last velocity field is gathered and plotted on rank 0.

    Parameters
    ----------
    cartcomm : MPI.Cartcomm
        Cartesian communicator
    rank : int
        Rank of current process
    size : int
        Number of processes used for simulation
    NX : int
        Width of the grid
    NY : int
        Height of the grid
    subgrids_number_X : int
        Number of subgrids in x direction
    subgrids_number_Y : int
        Number of subgrids in y direction
    lbm_parameter : dict
        Dictionary of LBM parameters
    timesteps : int
        Number of timesteps

    Returns
    -------
    spent_time : float
        Time spent for simulation
    """

    rcoords = cartcomm.Get_coords(rank)
    # print("Rank {} has coordinates {}".format(rank, rcoords))

    # where to receive from and where send to 
    # syntax: Shift(direction,displacement) --> direction = 0 is y-direction, 
    #                                           direction = 1 is x-direction, 
    #                                           displacement = -1 is left/up, 
    #                                           displacement = 1 is right/down
    # cartesian origin is top left with x-direction to the right and y-direction down	
    # cartesian coordinates are given as (y,x)
    sR,dR = cartcomm.Shift(1,1)
    sL,dL = cartcomm.Shift(1,-1)
    sU,dU = cartcomm.Shift(0,-1)
    sD,dD = cartcomm.Shift(0,1)
    sd = np.array([sR,dR,sL,dL,sU,dU,sD,dD], dtype = int)

    # print("Rank {} has sd {}".format(rank, sd))

    if rank == 0:
        print("Sending and receiving directions")

    # gather all coordinates of processes
    allrcoords = cartcomm.gather(rcoords,root = 0)

    if rank == 0:
        print("Communication stats:")
        print_communication_stats(allrcoords, subgrids_number_X, subgrids_number_Y, size)

    # define actual grid size per process
    subgrid_size_x = NX//subgrids_number_X
    subgrid_size_y = NY//subgrids_number_Y

    # check if grid size is valid
    if subgrid_size_x == 0 or subgrid_size_y == 0:
        if rank == 0:
            print("ERROR: subgrid_size_x or subgrid_size_y is 0. subgrid_size_x: {}, subgrid_size_y: {}".format(subgrid_size_x, subgrid_size_y))
            print("Use larger grid size")
        return
    # print("Rank {} has subdomain of size x*y = {}x{}".format(rank, subgrid_size_x, subgrid_size_y))

    # get lbm parameters
    reynolds_number = lbm_parameter["reynolds_number"]
    characteristic_velocity = lbm_parameter["characteristic_velocity"]
    omega = lbm_parameter["omega"]
    boundary_conditions_full = lbm_parameter["boundary_conditions_full"]
    boundary_velocities_full = lbm_parameter["boundary_velocities_full"]


    # set correct boundary conditions
    boundary_conditions = {
        "bottom" : None,
        "top" : None,
        "left" : None,
        "right" : None
    }
    if rcoords[1] == 0: boundary_conditions["left"] = boundary_conditions_full["left"]
    if rcoords[1] == subgrids_number_X-1: boundary_conditions["right"] = boundary_conditions_full["right"]
    if rcoords[0] == 0: boundary_conditions["top"] = boundary_conditions_full["top"]
    if rcoords[0] == subgrids_number_Y-1: boundary_conditions["bottom"] = boundary_conditions_full["bottom"]
    # print('Rank {} has boundary conditions {}'.format(rank, boundary_conditions))

    # set correct boundary velocities
    boundary_velocities = {
        "bottom": 0.0, 
        "top": 0.0, 
        "left": 0.0, 
        "right": 0.0
    }
    if boundary_conditions["top"] == "moving_wall":
        boundary_velocities["top"] = boundary_velocities_full["top"]
    if boundary_conditions["bottom"] == "moving_wall":
        boundary_velocities["bottom"] = boundary_velocities_full["bottom"]
    if boundary_conditions["left"] == "moving_wall":
        boundary_velocities["left"] = boundary_velocities_full["left"]
    if boundary_conditions["right"] == "moving_wall":
        boundary_velocities["right"] = boundary_velocities_full["right"]

    # print('Rank {} has boundary velocities {}'.format(rank, boundary_velocities))
    # initialize lbm per process
    lbm = LBM(subgrid_size_x, subgrid_size_y, omega, boundary_conditions=boundary_conditions, boundary_velocities=boundary_velocities)
    
    spent_time = None
    if rank == 0:
        # start timer
        start_time = MPI.Wtime()
        print("STARTING LBM SIMULATION")
    for t in range(timesteps):
        # do communication
        lbm.f_iyx = communicate(lbm, cartcomm, sd)
        # update step
        lbm.step()

        if rank == 0 and t%1000 == 0:
            print('Rank {} timestep {}/{}'.format(rank, t, timesteps))

    if rank == 0:
        print("LBM SIMULATION FINISHED")
        end_time = MPI.Wtime()
        print("CPU{}, Y{}, X{} --> Time elapsed: {}".format(size, NY, NX, end_time - start_time))
        spent_time = end_time - start_time
    
    # gather data for plotting last velocity field
    lbm.update_velocity_field()
    velocity_field_size = lbm.velocity_field_Cyx.shape
    velocity_field_size_x = velocity_field_size[2] - 2
    velocity_field_size_y = velocity_field_size[1] - 2
    
    # gather velocity field sizes of each process
    local_velocity_field_sizes_x = np.array(cartcomm.gather(velocity_field_size_x, root=0))
    local_velocity_field_sizes_y = np.array(cartcomm.gather(velocity_field_size_y, root=0))
    local_velocity_field_size_full = None

    recvbuf = None
    if rank == 0:
        print("Velocity fields gathered")
        # calculate full size of velocity field
        local_velocity_field_size_full = 2 * local_velocity_field_sizes_x * local_velocity_field_sizes_y
        recvbuf = np.zeros(np.sum(local_velocity_field_size_full))

    # gather velocity field data of each process
    cartcomm.Gatherv(lbm.get_velocity_field_Cyx(True).flatten(), recvbuf=(recvbuf, local_velocity_field_size_full), root=0)
    velocityFieldsBuffer = np.array(recvbuf)

    if rank == 0:
        print("Building full velocity field...")
        # merge single velocity fields to one big velocity field

        # sum over the rows of x sizes to get the final x size
        final_velocity_field_size_x = np.sum(local_velocity_field_sizes_x) // subgrids_number_Y
        # sum over the columns of y sizes to get the final y size
        final_velocity_field_size_y = np.sum(local_velocity_field_sizes_y) // subgrids_number_X
        if subgrids_number_X == 1:
            final_velocity_field_size_x = velocity_field_size_x
        if subgrids_number_Y == 1:
            final_velocity_field_size_y = velocity_field_size_y

        # prepare the fully combined velocity field
        simulated_velocity_field_Cyx = np.zeros((2, final_velocity_field_size_y, final_velocity_field_size_x))
    
        # go row-wise through processes and put them into simulated_velocity_field_Cyx
        current_rank = 0
        previous_end_index = 0
        for j in np.arange(subgrids_number_Y):
            for i in np.arange(subgrids_number_X):
                # check if current rank aligns with current coordinates
                if allrcoords[current_rank][0] != j or allrcoords[current_rank][1] != i:
                    print("ERROR: Rank {} does not align with coordinates y,x=({}, {})".format(current_rank, j, i))
                    return

                grid_length = local_velocity_field_size_full[current_rank]
                rank_velocity_field_height = local_velocity_field_sizes_y[current_rank]
                rank_velocity_field_width = local_velocity_field_sizes_x[current_rank]
                start_index = previous_end_index
                end_index = previous_end_index + grid_length
                velocity_field_part_data = velocityFieldsBuffer[start_index:end_index]
                velocity_field_Cyx = velocity_field_part_data.reshape((2, rank_velocity_field_height, rank_velocity_field_width))

                # put velocity field part into simulated_velocity_field_Cyx
                simulated_velocity_field_Cyx[:, j*rank_velocity_field_height:(j+1)*rank_velocity_field_height, i*rank_velocity_field_width:(i+1)*rank_velocity_field_width] = velocity_field_Cyx
                previous_end_index = end_index
                current_rank += 1
        print("Plotting...")
        v_shape = simulated_velocity_field_Cyx.shape
        plot_name = "SlidingLidResults/PARALLEL_Sliding_Lid_Velocity_Streamplot_Y{}_X{}_T_{}_RE_{}_CPU_{}_FULL.png".format(v_shape[1], v_shape[2], timesteps, reynolds_number, size)
        plot_velocity_field(simulated_velocity_field_Cyx, plot_name, timesteps-1, characteristic_velocity)
        # store velocity field data
        np.save("SlidingLidResults/PARALLEL_Sliding_Lid_Velocity_Field_Y{}_X{}_T_{}_RE_{}_CPU_{}_FULL.npy".format(v_shape[1], v_shape[2], timesteps, reynolds_number, size), simulated_velocity_field_Cyx)
    
    if rank == 0:
        return spent_time
    else:
        return None 

def run_lbm_parallel(lbm_parameter, timesteps=100000, number_of_processes_list=None, more_grid_sizes=None):
    """
    Run LBM in parallel. This function runs the grid decomposition and creates the cartesian communicator.
    It then runs the lbm in parallel with the given number of processes and grid sizes.
    MLUPS and time spent are plotted on rank 0.

    Parameters
    ----------
    lbm_parameter : dict
        Dictionary of LBM parameters
    timesteps : int, optional
        Number of timesteps. The default is 100000.
    number_of_processes_list : list, optional
        List of number of processes. The default is None, then the number of processes is the number of total processes available.
    more_grid_sizes : list, optional
        List of grid sizes. The default is None, then only the first grid size, defined in lbm_parameter, is used.

    Returns
    -------
    None.
    """
    comm = MPI.COMM_WORLD
    # get the size (number of total processes)
    size = comm.Get_size()
    # get the rank (number of current process)
    rank = comm.Get_rank()

    first_NX = lbm_parameter["width"]
    first_NY = lbm_parameter["height"]

    # build list of grid sizes if given
    grid_sizes_list = [[first_NX, first_NY]]
    if more_grid_sizes is not None:
        grid_sizes_list += more_grid_sizes

    # prepare figures for plotting on rank 0
    mlups_fig = None
    mlups_ax = None
    time_spent_fig = None
    time_spent_ax = None
    if rank == 0:
        mlups_fig = plt.figure()
        mlups_ax = mlups_fig.gca()
        time_spent_fig = plt.figure()
        time_spent_ax = time_spent_fig.gca()
        
    # loop over grid sizes
    for NX, NY in grid_sizes_list:

        if rank == 0:
            print('NX = {} NY = {}'.format(NX,NY))
        time_spent_list = []

        if number_of_processes_list is None:
            number_of_processes_list = [size]
        
        # loop over number of processes and run lbm in parallel
        for i, number_of_processes in enumerate(number_of_processes_list):
            if number_of_processes > size:
                number_of_processes = size
                print("Number of processes is larger than the number of processes requested. Using {} processes instead.".format(size))

            # calculate number of subgrids in x and y direction with the domain decomposition
            subgrids_number_X, subgrids_number_Y = domain_decomposition(NX, NY, number_of_processes)

            if rank == 0:
                print('DECOMPOSITION: sectX = {} sectY = {}'.format(subgrids_number_X,subgrids_number_Y))       

            comm.barrier()
            # create cartesian communicator and get coordinates (rcoords) of rank in form (y,x)
            cartcomm=comm.Create_cart(dims=[subgrids_number_Y, subgrids_number_X],periods=[False,False],reorder=False)
            
            time_spent = None

            # only use processes that are needed by decomposition
            if rank >= subgrids_number_X*subgrids_number_Y:
                if rank == subgrids_number_X*subgrids_number_Y:
                    print('Rank {} and larger ranks are idle.'.format(rank))
            else:
                used_size = subgrids_number_X*subgrids_number_Y
                if rank == 0:
                    print("Actual number of processes used: {}/{}".format(used_size, number_of_processes))
                
                # run lbm in parallel with defined number of subgrids and processes
                time_spent = run_lbm(cartcomm, rank, used_size, NX, NY, subgrids_number_X, subgrids_number_Y, lbm_parameter, timesteps)
                if rank == 0:
                    time_spent_list.append(time_spent)

                cartcomm.barrier()
                cartcomm.Free()
        
            if rank == 0:
                # plot time spent over number of processes
                time_spent_array = np.array(time_spent_list)
                # set x tick to int
                time_spent_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                time_spent_ax.set_xlabel("Number of processes")
                time_spent_ax.set_ylabel("Time spent [s]")
                time_spent_ax.set_title("Time spent over number of processes for t={}".format(timesteps))
                label = None
                if i == len(number_of_processes_list) - 1:
                    label = "X={}, Y={}".format(NX, NY)
                time_spent_ax.plot(number_of_processes_list[:i+1], time_spent_array, marker="*", label=label)
                time_spent_ax.legend()
                time_spent_ax.grid(True)
                # plot marker
                # ax.plot(number_of_processes_list, time_spent_list, "o")
                time_spent_fig.savefig("SlidingLidResults/PARALLEL_Sliding_Lid_Time_Spent_T_{}_CPU{}to{}.png".format(timesteps, number_of_processes_list[0], number_of_processes_list[-1]))

                # plot mlups with log scale
                mlups_ax.set_xlabel("Number of processes")
                mlups_ax.set_ylabel("MLUPS")
                mlups_ax.set_title("MLUPS over number of processes for t={}".format(timesteps))
                mlups_ax.loglog(number_of_processes_list[:i+1], (NX*NY*timesteps)/(time_spent_array*1000000), marker="*", label=label)
                mlups_ax.set_yscale("log")
                mlups_ax.set_xscale("log")
                # set xticks to powers of 10
                mlups_ax.set_xticks([1, 10, 100, 1000])
                mlups_ax.set_xticklabels([1, 10, 100, 1000])
                # set yticks to powers of 10
                mlups_ax.set_yticks([1, 10, 100, 1000])
                mlups_ax.set_yticklabels([1, 10, 100, 1000])
                #set xlim slightly larger than last tick
                mlups_ax.set_xlim(0.5, 1100)
                # set legend as grid size
                mlups_ax.legend()
                # show grid
                mlups_ax.grid(True, which="both")
                mlups_fig.savefig("SlidingLidResults/PARALLEL_Sliding_Lid_MLUPS_T_{}_CPU{}to{}.png".format(timesteps, number_of_processes_list[0], number_of_processes_list[-1]))

def print_communication_stats(allrcoords, subgrids_number_X, subgrids_number_Y, size):
    """
    Print communication stats

    Parameters
    ----------
    allrcoords : np.array
        Array of all coordinates of processes
    subgrids_number_X : int
        Number of subgrids in x direction
    subgrids_number_Y : int
        Number of subgrids in y direction
    size : int
        Number of processes

    Returns
    -------
    None.
    """
    cartarray = np.ones((subgrids_number_Y,subgrids_number_X),dtype=int)
    for i in np.arange(size):
        cartarray[allrcoords[i][0],allrcoords[i][1]] = i
    print("Cartesian topology:")
    print(cartarray)

def plot_velocity_field(velocity_field_Cyx, plot_name, timesteps, characteristic_velocity):
    """
    Plot velocity field

    Parameters
    ----------
    velocity_field_Cyx : np.array
        Velocity field
    plot_name : str
        Name of the plot
    timesteps : int
        Timestep of the velocity field
    characteristic_velocity : float
        Characteristic velocity

    Returns
    -------
    None.
    """
    width = velocity_field_Cyx.shape[2]
    height = velocity_field_Cyx.shape[1]
    
    fig_velocity_field = plt.figure()
    ax = fig_velocity_field.add_subplot(111)
    cbar = fig_velocity_field.colorbar(ax=ax, mappable=matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, characteristic_velocity), cmap="viridis"))
    cbar.set_label("Velocity magnitude")
    cbar.set_ticks(np.linspace(0, characteristic_velocity, 5))
    ax.clear()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-10, width + 10)
    ax.set_ylim(-10, height + 10)
    ax.set_title("Velocity streamplot at t = " + str(timesteps))
    velocity_field = np.flip(velocity_field_Cyx, axis=1)
    u_x = velocity_field[0]
    u_y = velocity_field[1]

    ax.streamplot(np.arange(0, u_x.shape[1]), np.arange(0, u_y.shape[0]), u_x, u_y, color=np.sqrt(u_x**2 + u_y**2), density=1.5, norm=plt.Normalize(0, characteristic_velocity))
    ax.plot([-0.5, width - 0.5], [height-1.5, height-1.5], color="red", label="Moving wall")
    ax.plot([-0.5, width - 0.5], [0.5, 0.5], color="black", label = "Fixed wall")
    # draw vertical boundaries
    ax.plot([-0.5, -0.5], [0.5, height- 1.5], color="black")
    ax.plot([width-0.5, width-0.5], [0.5, height - 1.5], color="black")

    plt.tight_layout()
    fig_velocity_field.savefig(plot_name)
    plt.close(fig_velocity_field)

if __name__ == "__main__":
    # -------------------------- Milestone 7 --------------------------
    # Change here parameters for different simulations
    # Omega is calculated from the Reynolds number, but every parameter can bet set manually aswell in the lbm_parameter dictionary
    # Different lattice dimension can be set by changing NX and NY or by adding more grid sizes in more_grid_sizes (also format [NX, NY]])
    # Different number of processes can be set by changing number_of_processes_list, currently set to 2^x from 1 up to 1024
    #
    # Run this code locally in parallel with the following command (specify number of processes with -n):
    # mpiexec -n 4 python3 Milestone7_Parallelization.py
    # ------------------------------------------------------------------
    
    # define simulation parameters
    # number of timesteps
    nt = 1000
    # grid size in x and y direction
    NX = 300
    NY = 300

    # define decomposition parameters
    # define number of processes to use by 2^x up to 1024, those are plotted on the MLUPS plot
    # set to None to use maximum number of processes available
    number_of_processes_list = [2**i for i in range(0, 11)]
    # number_of_processes_list = None
    # define different grid sizes to use (after grid size defined by [NX, NY]), those are plotted on the time MLUPS plot
    # set to None if not needed
    more_grid_sizes = [[200, 200], [400, 400], [800, 800]]
    # more_grid_sizes = None

    # define lbm parameters
    reynolds_number = 1000
    characteristic_length = NX if NX > NY else NY
    characteristic_velocity = 0.3
    kinematic_viscosity = characteristic_length * characteristic_velocity / reynolds_number
    omega = 1.0 / (3 * kinematic_viscosity + 0.5)
    # define the boundary velocity for the full domain
    boundary_velocities_full = {"bottom": 0.0, "top": characteristic_velocity, "left": 0.0, "right": 0.0}
    # define the boundary conditions for the full domain
    boundary_conditions_full = {
        "bottom" : "bounce_back",
        "top" : "moving_wall",
        "left" : "bounce_back",
        "right" : "bounce_back"
    }

    # define lbm parameter dictionary
    lbm_parameter = {
        "reynolds_number": reynolds_number,
        "width": NX,
        "height": NY,
        "characteristic_length": characteristic_length,
        "characteristic_velocity": characteristic_velocity,
        "kinematic_viscosity": kinematic_viscosity,
        "omega": omega,
        "boundary_velocities_full": boundary_velocities_full,
        "boundary_conditions_full": boundary_conditions_full 
    }

    # run lbm in parallel
    run_lbm_parallel(lbm_parameter, timesteps=nt, number_of_processes_list=number_of_processes_list, more_grid_sizes=more_grid_sizes)