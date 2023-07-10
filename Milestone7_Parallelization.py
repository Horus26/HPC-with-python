import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib
from LBM import LBM


def convert_physical_to_lattice(lbm_parameter_physical):
    return {
        "reynolds_number": lbm_parameter_physical["reynolds_number"],
        "width": lbm_parameter_physical["width"],
        "height": lbm_parameter_physical["height"],
        "characteristic_length": lbm_parameter_physical["characteristic_length"],
        "characteristic_velocity": lbm_parameter_physical["characteristic_velocity"],
        "kinematic_viscosity": lbm_parameter_physical["kinematic_viscosity"],
        "omega": 1.0 / (3 * lbm_parameter_physical["kinematic_viscosity"] + 0.5),
        "boundary_velocities": lbm_parameter_physical["boundary_velocities"],
        "boundary_conditions": lbm_parameter_physical["boundary_conditions"]
    }

def communicate(lbm : LBM, cartcomm, sd):
    # decompose sd
    sR,dR,sL,dL,sU,dU,sD,dD = sd

    # cache f
    f_iyx = lbm.f_iyx
    nysub_with_b = lbm.height
    nxsub_with_b = lbm.width
    boundary_conditions = lbm.boundary_conditions

    # send and receive right
    # print("CASE RIGHT: RANK {} SENDS TO {} AND RECEIVES FROM {}".format(rank, dR, sR))
    # get relevant channels of f_iyx and flatten them
    i1 = f_iyx[1, :, -2].copy()
    i5 = f_iyx[5, :, -2].copy()
    i8 = f_iyx[8, :, -2].copy()
    sendbuf = np.array([i1, i5, i8]).flatten()
    recvbuf = np.zeros(nysub_with_b*3)
    cartcomm.Sendrecv(sendbuf, dest = dR, recvbuf = recvbuf, source = sR)
    # decompose recvbuf into channels and put them into f_iyx
    if boundary_conditions["left"] == None:
        f_iyx[1, :, 0] = recvbuf[:nysub_with_b].copy()
        f_iyx[5, :, 0] = recvbuf[nysub_with_b:2*nysub_with_b].copy()
        f_iyx[8, :, 0] = recvbuf[2*nysub_with_b:].copy()
    # elif rank == 0:
        # print(recvbuf[:nysub_with_b])

    # send and receive left
    # get relevant channels of f_iyx and flatten them
    i3 = f_iyx[3, :, 1].copy()
    i6 = f_iyx[6, :, 1].copy()
    i7 = f_iyx[7, :, 1].copy()
    sendbuf = np.array([i3, i6, i7]).flatten()
    recvbuf = np.zeros(nysub_with_b*3)
    cartcomm.Sendrecv(sendbuf, dest = dL, recvbuf = recvbuf, source = sL)
    # decompose recvbuf into channels and put them into f_iyx
    if boundary_conditions["right"] == None:
        f_iyx[3, :, -1] = recvbuf[:nysub_with_b].copy()
        f_iyx[6, :, -1] = recvbuf[nysub_with_b:2*nysub_with_b].copy()
        f_iyx[7, :, -1] = recvbuf[2*nysub_with_b:].copy()

    # send and receive up
    # get relevant channels of f_iyx and flatten them
    i2 = f_iyx[2, 1, :].copy()
    i5 = f_iyx[5, 1, :].copy()
    i6 = f_iyx[6, 1, :].copy()
    sendbuf = np.array([i2, i5, i6]).flatten()
    recvbuf = np.zeros(nxsub_with_b*3)
    cartcomm.Sendrecv(sendbuf, dest = dU, recvbuf = recvbuf, source = sU)
    # decompose recvbuf into channels and put them into f_iyx
    if boundary_conditions["bottom"] == None:
        f_iyx[2, -1, :] = recvbuf[:nxsub_with_b]
        f_iyx[5, -1, :] = recvbuf[nxsub_with_b:2*nxsub_with_b]
        f_iyx[6, -1, :] = recvbuf[2*nxsub_with_b:]

    # send and receive down
    # get relevant channels of f_iyx and flatten them
    i4 = f_iyx[4, -2, :].copy()
    i7 = f_iyx[7, -2, :].copy()
    i8 = f_iyx[8, -2, :].copy()
    sendbuf = np.array([i4, i7, i8]).flatten()
    recvbuf = np.zeros(nxsub_with_b*3)
    cartcomm.Sendrecv(sendbuf, dest = dD, recvbuf = recvbuf, source = sD)
    # decompose recvbuf into channels and put them into f_iyx
    if boundary_conditions["top"] == None:
        f_iyx[4, 0, :] = recvbuf[:nxsub_with_b]
        f_iyx[7, 0, :] = recvbuf[nxsub_with_b:2*nxsub_with_b]
        f_iyx[8, 0, :] = recvbuf[2*nxsub_with_b:]
    
    return f_iyx


def run_lbm_parallel(NX, NY, lbm_parameter, timesteps=100000):

    comm = MPI.COMM_WORLD      # start the communicator
    size = comm.Get_size()     # get the size (number of total processes)
    rank = comm.Get_rank()     # get the rank (number of current process)

    # start timer
    start_time = MPI.Wtime()

    NX = lbm_parameter["width"]
    NY = lbm_parameter["height"]


    print('NX = {} NY = {}, and sqrt(NX*NY) = {}'.format(NX,NY,np.sqrt(NX*NY)))
    print('NX/NY = {} and NY/NX = {}\n'.format(NX/NY,NY/NX))
    #

    sectsY = None
    sectsX = None
    if NX < NY:
        factor_sectY_larger = np.floornt(NY/NX)

        if factor_sectY_larger == 1:
            sectsY = size
            sectsX = 1

        else:
            size_Y_share = factor_sectY_larger / (1+factor_sectY_larger)
            sectsY = int(np.floor(size*size_Y_share))
            sectsX = int(np.floor(size/sectsY))

            if sectsX == 1:
                sectsY += 1

    elif NX > NY:
        factor_sectX_larger = np.floor(NX/NY)

        if factor_sectX_larger == 1:
            sectsX = size
            sectsY = 1
        
        else:
            size_X_share = factor_sectX_larger / (1+factor_sectX_larger)
            sectsX = int(np.floor(size*size_X_share))
            sectsY = int(np.floor(size/sectsX))

            if sectsY == 1:
                sectsX += 1


    elif NX==NY:
        sectsX=int(np.floor(np.sqrt(size)))
        sectsY=int(size/sectsX)
        if rank == 0: print('In the case of equal size we divide the processes as {} and {}'.format(sectsX,sectsY))

    print("Decomposition: y={}, x={}".format(sectsY, sectsX))
    print('Rank {}/{} is alive.'.format(rank, size))          


    # create cartesian communicator and get coordinates (rcoords) of rank in form (y,x)
    cartcomm=comm.Create_cart(dims=[sectsY, sectsX],periods=[False,False],reorder=False)
    rcoords = cartcomm.Get_coords(rank)
    print("Rank {} has coordinates {}".format(rank, rcoords))

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

    allrcoords = comm.gather(rcoords,root = 0)
    allDestSourBuf = np.zeros(size*8, dtype = int)
    comm.Gather(sd, allDestSourBuf, root = 0)

    if rank == 0:
        print_communication_stats(allrcoords, allDestSourBuf, sectsX, sectsY, size)


    # define actual grid size
    nxsub = NX//sectsX
    nysub = NY//sectsY
    print('Rank {}/{} has a subdomain of size x*y = {}x{}'.format(rank, size, nxsub, nysub))
    print("sectX = {}, sectY = {}".format(sectsX, sectsY))
    # add missing nodes in y direction to first row of processes
    if nysub * sectsY != NY and rcoords[0] == 0:
        nysub += NY - (nysub * sectsY)
        print('Rank {}/{} has a subdomain of size x*y = {}x{}'.format(rank, size, nxsub, nysub))
    
    # add missing nodes in x direction to first column of processes
    if nxsub * sectsX != NX and rcoords[1] == 0:
        nxsub += NX - (nxsub * sectsX)
        print('Rank {}/{} has a subdomain of size x*y = {}x{}'.format(rank, size, nxsub, nysub))


    # get lbm parameters
    reynolds_number = lbm_parameter["reynolds_number"]
    characteristic_length = lbm_parameter["characteristic_length"]
    characteristic_velocity = lbm_parameter["characteristic_velocity"]
    kinematic_viscosity = lbm_parameter["kinematic_viscosity"]
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
    if rcoords[1] == sectsX-1: boundary_conditions["right"] = boundary_conditions_full["right"]
    if rcoords[0] == 0: boundary_conditions["top"] = boundary_conditions_full["top"]
    if rcoords[0] == sectsY-1: boundary_conditions["bottom"] = boundary_conditions_full["bottom"]
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

    # initialize lbm per process
    lbm = LBM(nxsub, nysub, omega, boundary_conditions=boundary_conditions, boundary_velocities=boundary_velocities)

    for t in range(timesteps):
        # do communication
        lbm.f_iyx = communicate(lbm, cartcomm, sd)

        lbm.step()
        if rank == 0 and t%1000 == 0:
            print('Rank {} timestep {}/{}'.format(rank, t, timesteps))

    # gather all data and plot last velocity field
    lbm.update_velocity_field()
    velocity_field_size = lbm.velocity_field_Cyx.shape
    velocity_field_size_x = velocity_field_size[2] - 2
    velocity_field_size_y = velocity_field_size[1] - 2
    print("velocity_field_size x: {}, y: {}".format(velocity_field_size_x, velocity_field_size_y))
    local_velocity_field_size_x = np.array(comm.gather(velocity_field_size_x, root=0))
    local_velocity_field_size_y = np.array(comm.gather(velocity_field_size_y, root=0))
    local_velocity_field_size_full = None

    recvbuf = None
    if rank == 0:
        local_velocity_field_size_full = 2 * local_velocity_field_size_x * local_velocity_field_size_y
        recvbuf = np.zeros(np.sum(local_velocity_field_size_full))

    comm.Gatherv(lbm.get_velocity_field_Cyx(True).flatten(), recvbuf=(recvbuf, local_velocity_field_size_full), root=0)
    allDestSourBuf = np.array(recvbuf)

    # simulated_velocity_field_Cyx = lbm.get_velocity_field_Cyx(True)
    # print("Plotting...")
    # plot per rank velocity field
    # plot_name = "SlidingLidResults/PARALLEL_Sliding_Lid_Velocity_Streamplot_T_{}_RE_{}_RANK_{}.png".format(timesteps, reynolds_number, rank)
    # plot_velocity_field(simulated_velocity_field_Cyx, plot_name, timesteps-1, characteristic_velocity)
    # store velocity field data
    # np.save("SlidingLidResults/PARALLEL_Sliding_Lid_Velocity_Field_T_{}_RE_{}_RANK_{}.npy".format(timesteps, reynolds_number, rank), simulated_velocity_field_Cyx)

    if rank == 0:
        # merge single velocity fields to one big velocity field
        final_velocity_field_size_x = np.sum(local_velocity_field_size_x)
        final_velocity_field_size_y = np.sum(local_velocity_field_size_y)
        if sectsX == 1:
            final_velocity_field_size_x = velocity_field_size_x
        if sectsY == 1:
            final_velocity_field_size_y = velocity_field_size_y

        simulated_velocity_field_Cyx = np.zeros((2, final_velocity_field_size_y, final_velocity_field_size_x))
    
        # go row-wise through processes and put them into simulated_velocity_field_Cyx
        current_rank = 0
        last_end_index = 0
        print("ALLDESTSOURBUF shape: {}".format(allDestSourBuf.shape))
        print("Local velocity field size full shape: {}".format(local_velocity_field_size_full.shape))
        for j in np.arange(sectsY):
            for i in np.arange(sectsX):
                # multiply with 2 because we have 2 velocity channels
                grid_length = local_velocity_field_size_full[current_rank]
                rank_velocity_field_height = local_velocity_field_size_y[current_rank]
                rank_velocity_field_width = local_velocity_field_size_x[current_rank]
                start_index = last_end_index
                end_index = last_end_index + grid_length
                print("start_index: {}, end_index: {}".format(start_index, end_index))
                # velocity_field_part_data = allDestSourBuf[i+j*sectsX]
                velocity_field_part_data = allDestSourBuf[start_index:end_index]
                print("allDestSourBuf.shape: {}".format(allDestSourBuf.shape))
                print("velocity_field_part_data.shape: {}".format(velocity_field_part_data.shape))
                velocity_field_Cyx = velocity_field_part_data.reshape((2, rank_velocity_field_height, rank_velocity_field_width))

                
                simulated_velocity_field_Cyx[:, j*rank_velocity_field_height:(j+1)*rank_velocity_field_height, i*rank_velocity_field_width:(i+1)*rank_velocity_field_width] = velocity_field_Cyx
                last_end_index = end_index
                current_rank += 1
        print("Plotting...")
        plot_name = "SlidingLidResults/PARALLEL_Sliding_Lid_Velocity_Streamplot_T_{}_RE_{}_FULL.png".format(timesteps, reynolds_number)
        plot_velocity_field(simulated_velocity_field_Cyx, plot_name, timesteps-1, characteristic_velocity)
        # store velocity field data
        np.save("SlidingLidResults/PARALLEL_Sliding_Lid_Velocity_Field_T_{}_RE_{}_FULL.npy".format(timesteps, reynolds_number), simulated_velocity_field_Cyx)

        # print time elapsed
        print("Time elapsed: {}".format(MPI.Wtime() - start_time))


def print_communication_stats(allrcoords, allDestSourBuf, sectsX, sectsY, size):
    print(' ')
    cartarray = np.ones((sectsY,sectsX),dtype=int)
    allDestSour = np.array(allDestSourBuf).reshape((size,8))
    for i in np.arange(size):
        cartarray[allrcoords[i][0],allrcoords[i][1]] = i
        print('Rank {} all destinations and sources {}'.format(i,allDestSour[i,:]))
        sR_temp,dR_temp,sL_temp,dL_temp,sU_temp,dU_temp,sD_temp,dD_temp = allDestSour[i]
        print('Rank {} is at {}'.format(i,allrcoords[i]))
        print('sour/dest right {} {}'.format(sR_temp,dR_temp))
        print('sour/dest left  {} {}'.format(sL_temp,dL_temp))  
        print('sour/dest up    {} {}'.format(sU_temp,dU_temp))
        print('sour/dest down  {} {}'.format(sD_temp,dD_temp))
        #print('[stdout:',i,']',allDestSour[i])
    print('')
    print(cartarray)

def plot_velocity_field(velocity_field_Cyx, plot_name, timesteps, characteristic_velocity):
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

    print("velocity_field.shape: {}".format(velocity_field.shape))
    print("u_x.shape: {}".format(u_x.shape))
    print("u_y.shape: {}".format(u_y.shape))
    ax.streamplot(np.arange(0, u_x.shape[1]), np.arange(0, u_y.shape[0]), u_x, u_y, color=np.sqrt(u_x**2 + u_y**2), density=1.5, norm=plt.Normalize(0, characteristic_velocity))
    ax.plot([-0.5, width - 0.5], [height-1.5, height-1.5], color="red", label="Moving wall")
    ax.plot([-0.5, width - 0.5], [0.5, 0.5], color="black", label = "Fixed wall")
    # draw vertical boundaries
    ax.plot([-0.5, -0.5], [0.5, height- 1.5], color="black")
    ax.plot([width-0.5, width-0.5], [0.5, height - 1.5], color="black")

    plt.tight_layout()
    # save with rank number
    plt.savefig(plot_name)


if __name__ == "__main__":
    # define decomposition parameters
    nt = 20000  # timesteps to iterate
    NX = 100
    NY = 80


    reynolds_number = 1000
    characteristic_length = NX
    characteristic_velocity = 0.3
    kinematic_viscosity = characteristic_length * characteristic_velocity / reynolds_number
    omega = 1.0 / (3 * kinematic_viscosity + 0.5)
    boundary_velocities_full = {"bottom": 0.0, "top": characteristic_velocity, "left": 0.0, "right": 0.0}
    boundary_conditions_full = {
        "bottom" : "bounce_back",
        "top" : "moving_wall",
        "left" : "bounce_back",
        "right" : "bounce_back"
    }

    lbm_parameter_physical = {
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

    # lbm_parameter_lattice = convert_physical_to_lattice(lbm_parameter_physical)
    run_lbm_parallel(NX, NY, lbm_parameter_physical, timesteps=nt)

    # mpiexec -n 4 python3 Milestone7_Parallelization.py