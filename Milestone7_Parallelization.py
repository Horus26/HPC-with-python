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


def run_lbm_parallel(NX, NY, lbm_parameter, timesteps=100000):
    pass


#
# Start Intracommunicator and get what is out there i.t.o. size and rank
# Note that all these variables are known to all ranks where size is equal
# for all and rank is specific to the rank number of the process.
#
comm = MPI.COMM_WORLD      # start the communicator assign to comm
size = comm.Get_size()     # get the size (number of total processes) and assign to size
rank = comm.Get_rank()     # get the rank and assign to rank

# check for correct number of processes
if size % 2 != 0:
    if rank == 0: print("Please use an even number of processes.")
    exit()

# start timer
start_time = MPI.Wtime()

NX = 300
NY = 300


print('NX = {} NY = {}, and sqrt(NX*NY) = {}'.format(NX,NY,np.sqrt(NX*NY)))
print('NX/NY = {} and NY/NX = {}\n'.format(NX/NY,NY/NX))
#
if NX < NY:
    sectsX=int(np.floor(np.sqrt(size*NX/NY)))
    sectsY=int(np.floor(size/sectsX))
    print('We have {} fields in x-direction and {} in y-direction'.format(sectsX,sectsY))
    print('How do the fractions look like?')
    print('NX/NY={} and sectsX/sectsY = {}\n'.format(NX/NY,sectsX/sectsY))
elif NX > NY:
    sectsX=int(np.floor(np.sqrt(size*NY/NX)))
    sectsY=int(np.floor(size/sectsX))
    print('We have {} fields in x-direction and {} in y-direction'.format(sectsX,sectsY))
    print('How do the fractions look like?')
    print('NX/NY={} and sectsX/sectsY = {}\n'.format(NX/NY,sectsX/sectsY))
elif NX==NY:
    sectsX=int(np.floor(np.sqrt(size)))
    sectsY=int(size/sectsX)
    if rank == 0: print('In the case of equal size we divide the processes as {} and {}'.format(sectsX,sectsY))

print('Rank {}/{} is alive.'.format(rank, size))


nxsub = NX//sectsX
nysub = NY//sectsY
print('Rank {}/{} has a subdomain of size x*y = {}x{}'.format(rank, size, nxsub, nysub))
print("sectX = {}, sectY = {}".format(sectsX, sectsY))


cartcomm=comm.Create_cart(dims=[sectsY, sectsX],periods=[False,False],reorder=False)
rcoords = cartcomm.Get_coords(rank)
print("Rank {} has coordinates {}".format(rank, rcoords))
# boundary order is left, right, bottom, top
boundary_conditions = {
    "bottom" : None,
    "top" : None,
    "left" : None,
    "right" : None
}
if rcoords[1] == 0: boundary_conditions["left"] = "bounce_back"
if rcoords[1] == sectsX-1: boundary_conditions["right"] = "bounce_back"
if rcoords[0] == 0: boundary_conditions["top"] = "moving_wall"
if rcoords[0] == sectsY-1: boundary_conditions["bottom"] = "bounce_back"

print('Rank {} has boundary conditions {}'.format(rank, boundary_conditions))

# where to receive from and where send to 
# syntax: Shift(direction,displacement) --> direction = 0 is y-direction, 
#                                           direction = 1 is x-direction, 
#                                           displacement = -1 is left/up, 
#                                           displacement = 1 is right/down
# cartesian origin is top left with x-direction to the right and y-direction down	
# cartesian coordinates are given as (y,x)
sR,dR = cartcomm.Shift(1,1)
sL,dL = cartcomm.Shift(1,-1)
#sU,dU = cartcomm.Shift(0,1)
#sD,dD = cartcomm.Shift(0,-1)
sU,dU = cartcomm.Shift(0,-1)
sD,dD = cartcomm.Shift(0,1)
#
sd = np.array([sR,dR,sL,dL,sU,dU,sD,dD], dtype = int)

allrcoords = comm.gather(rcoords,root = 0)
allDestSourBuf = np.zeros(size*8, dtype = int)
comm.Gather(sd, allDestSourBuf, root = 0)
#
#print(sd)
if rank == 0:
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

reynolds_number = 1000
characteristic_length = NX
characteristic_velocity = 0.3
kinematic_viscosity = characteristic_length * characteristic_velocity / reynolds_number
omega = 1.0 / (3 * kinematic_viscosity + 0.5)
boundary_velocities = {"bottom": 0.0, "top": 0.0, "left": 0.0, "right": 0.0}
if boundary_conditions["top"] == "moving_wall":
    boundary_velocities = {"bottom": 0.0, "top": characteristic_velocity, "left": 0.0, "right": 0.0}
lbm = LBM(nxsub, nysub, omega, boundary_conditions=boundary_conditions, boundary_velocities=boundary_velocities)

nysub_with_b = nysub + 2
nxsub_with_b = nxsub + 2
timesteps = 100000
for t in range(timesteps):
    # do communication
    # cache f
    f_iyx = lbm.f_iyx

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
        lbm.f_iyx[1, :, 0] = recvbuf[:nysub_with_b].copy()
        lbm.f_iyx[5, :, 0] = recvbuf[nysub_with_b:2*nysub_with_b].copy()
        lbm.f_iyx[8, :, 0] = recvbuf[2*nysub_with_b:].copy()
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
        lbm.f_iyx[3, :, -1] = recvbuf[:nysub_with_b].copy()
        lbm.f_iyx[6, :, -1] = recvbuf[nysub_with_b:2*nysub_with_b].copy()
        lbm.f_iyx[7, :, -1] = recvbuf[2*nysub_with_b:].copy()

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
        lbm.f_iyx[2, -1, :] = recvbuf[:nxsub_with_b]
        lbm.f_iyx[5, -1, :] = recvbuf[nxsub_with_b:2*nxsub_with_b]
        lbm.f_iyx[6, -1, :] = recvbuf[2*nxsub_with_b:]

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
        lbm.f_iyx[4, 0, :] = recvbuf[:nxsub_with_b]
        lbm.f_iyx[7, 0, :] = recvbuf[nxsub_with_b:2*nxsub_with_b]
        lbm.f_iyx[8, 0, :] = recvbuf[2*nxsub_with_b:]


    lbm.step()
    if rank == 0 and t%1000 == 0:
        print('Rank {} timestep {}/{}'.format(rank, t, timesteps))

# gather all data and plot last velocity field
lbm.update_velocity_field()
# allDestSourBuf = np.zeros((size, 2, lbm.height-2, lbm.width-2)).flatten()
# comm.Gather(lbm.get_velocity_field_Cyx(True).flatten(), allDestSourBuf, root = 0)
allDestSourBuf = comm.gather(lbm.get_velocity_field_Cyx(True).flatten(), root=0)
allDestSourBuf = np.array(allDestSourBuf)
rcoords_x = comm.gather(rcoords[1], root=0)
rcoords_y = comm.gather(rcoords[0], root=0)


simulated_velocity_field_Cyx = lbm.get_velocity_field_Cyx(True)
print("Plotting...")
fig_velocity_field = plt.figure()
ax = fig_velocity_field.add_subplot(111)
cbar = fig_velocity_field.colorbar(ax=ax, mappable=matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, characteristic_velocity), cmap="viridis"))
cbar.set_label("Velocity magnitude")
cbar.set_ticks(np.linspace(0, characteristic_velocity, 5))
ax.clear()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.set_xlim(-10, (lbm.width-2) +10)
ax.set_ylim(-10, (lbm.height-2) +10)
ax.set_title("Velocity streamplot at t = " + str(timesteps))
lbm.update_velocity_field()
velocity_field = simulated_velocity_field_Cyx
velocity_field = np.flip(velocity_field, axis=1)
u_x = velocity_field[0]
u_y = velocity_field[1]


ax.streamplot(np.arange(0, u_x[1].shape[0]), np.arange(0, u_y[0].shape[0]), u_x, u_y, color=np.sqrt(u_x**2 + u_y**2), density=1.5, norm=plt.Normalize(0, characteristic_velocity))
ax.plot([-0.5, (lbm.width) - 0.5], [(lbm.height)-1.5, (lbm.height)-1.5], color="red", label="Moving wall")
ax.plot([-0.5, (lbm.width) - 0.5], [0.5, 0.5], color="black", label = "Fixed wall")
# draw vertical boundaries
ax.plot([-0.5, -0.5], [0.5, (lbm.height)- 1.5], color="black")
ax.plot([(lbm.width)-0.5, (lbm.width)-0.5], [0.5, (lbm.height) - 1.5], color="black")

plt.tight_layout()
# save with rank number
plt.savefig("SlidingLidResults/PARALLEL_Sliding_Lid_Velocity_Streamplot_T_{}_RE_{}_RANK_{}.png".format(timesteps, reynolds_number, rank))


xy = np.array([rcoords_x,rcoords_y]).T
simulated_velocity_field_Cyx = np.zeros((2, (lbm.height-2)*sectsY, (lbm.width-2)*sectsX))
if rank == 0:
    # go row-wise through processes and put them into simulated_velocity_field_Cyx
    for j in np.arange(sectsY):
        for i in np.arange(sectsX):
            # multiply with 2 because we have 2 velocity channels
            grid_length = 2*(lbm.height-2)*(lbm.width-2)
            start_index = j*sectsX*grid_length + i*grid_length
            end_index = start_index + grid_length
            print("start_index: {}, end_index: {}".format(start_index, end_index))
            velocity_field_part_data = allDestSourBuf[i+j*sectsX]
            print("allDestSourBuf.shape: {}".format(allDestSourBuf.shape))
            print("velocity_field_part_data.shape: {}".format(velocity_field_part_data.shape))
            velocity_field_Cyx = velocity_field_part_data.reshape((2, lbm.height-2, lbm.width-2))

            simulated_velocity_field_Cyx[:, j*(lbm.height-2):(j+1)*(lbm.height-2), i*(lbm.width-2):(i+1)*(lbm.width-2)] = velocity_field_Cyx
    
    # reshape allDestSourBuf to one velocity field
    # simulated_velocity_field_Cyx = allDestSourBuf.reshape((2, (lbm.height-2)*sectsY, (lbm.width-2)*sectsX))
    print("Plotting...")
    fig_velocity_field = plt.figure()
    ax = fig_velocity_field.add_subplot(111)
    cbar = fig_velocity_field.colorbar(ax=ax, mappable=matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, characteristic_velocity), cmap="viridis"))
    cbar.set_label("Velocity magnitude")
    cbar.set_ticks(np.linspace(0, characteristic_velocity, 5))
    ax.clear()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-10, (lbm.width-2)*sectsX +10)
    ax.set_ylim(-10, (lbm.height-2)*sectsY +10)
    ax.set_title("Velocity streamplot at t = " + str(timesteps))
    lbm.update_velocity_field()
    velocity_field = simulated_velocity_field_Cyx
    velocity_field = np.flip(velocity_field, axis=1)
    u_x = velocity_field[0]
    u_y = velocity_field[1]
    

    ax.streamplot(np.arange(0, u_x[1].shape[0]), np.arange(0, u_y[0].shape[0]), u_x, u_y, color=np.sqrt(u_x**2 + u_y**2), density=1.5, norm=plt.Normalize(0, characteristic_velocity))
    ax.plot([-0.5, (lbm.width * sectsX) - 0.5], [(lbm.height * sectsY)-1.5, (lbm.height * sectsY)-1.5], color="red", label="Moving wall")
    ax.plot([-0.5, (lbm.width * sectsX) - 0.5], [0.5, 0.5], color="black", label = "Fixed wall")
    # draw vertical boundaries
    ax.plot([-0.5, -0.5], [0.5, (lbm.height * sectsY)- 1.5], color="black")
    ax.plot([(lbm.width * sectsX)-0.5, (lbm.width * sectsX)-0.5], [0.5, (lbm.height * sectsY) - 1.5], color="black")

    plt.tight_layout()
    plt.savefig("SlidingLidResults/PARALLEL_Sliding_Lid_Velocity_Streamplot_T_{}_RE_{}_FULL.png".format(timesteps, reynolds_number))
    # store velocity field data
    np.save("SlidingLidResults/PARALLEL_Sliding_Lid_Velocity_Field_T_{}_RE_{}_FULL.npy".format(timesteps, reynolds_number), simulated_velocity_field_Cyx)

    # print time elapsed
    print("Time elapsed: {}".format(MPI.Wtime() - start_time))



    # mpiexec -n 4 python3 Milestone7_Parallelization.py



if __name__ == "__main__":
    # define decomposition parameters
    dx = 0.1     # = dy
    nt = 100000  # timesteps to iterate
    dt = 0.0001   # timestep length
    D = 1        # diffusion constant
    # phyiscal_size_x = 400
    # phyiscal_size_y = 400

    NX = 300
    NY = 300

    lbm_parameter_physical = {
        "reynolds_number": 1000,
        "width": NX,
        "height": NY,
        "characteristic_length": NX,
        "characteristic_velocity": 0.3,
        "kinematic_viscosity": 0.00009,
        "omega": 1.0 / (3 * 0.00009 + 0.5),
        "boundary_velocities": {"bottom": 0.0, "top": 0.0, "left": 0.0, "right": 0.0},
        "boundary_conditions": {"bottom": None, "top": None, "left": None, "right": None}
    }

    lbm_parameter_lattice = convert_physical_to_lattice(lbm_parameter_physical)
    run_lbm_parallel(NX, NY, lbm_parameter_physical, timesteps=nt)