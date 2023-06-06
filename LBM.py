import numpy as np

class LBM:
    """
    Lattice Boltzmann Method (LBM) for simulating fluid flow.
    D2Q9 model is used - 9 directions in 2D.
    0 --> no direction
    1 --> right
    2 --> up
    3 --> left
    4 --> down
    5 --> right-up
    6 --> left-up
    7 --> left-down
    8 --> right-down

    Grid origin is defined to be in the bottom left corner of the grid.
    Care because numpy origin is in the top left corner of the grid.

    Information about notation:
    - _yx: y is the row index, x is the column index.
        Every field is given in the form of (y,x) .
    - _i: i is the index of the lattice direction.
    - _C: cartesian coordinates.
    - f: probability density function.
    - f_eq: equilibrium distribution function.

    """
    def __init__(self,
                 width : int,
                 height : int,
                 omega : float= 1.0,
                 viscosity : float = None,
                 inital_density_field_yx : np.ndarray = None,
                 inital_velocity_field_Cyx : np.ndarray = None,
                 boundary_conditions : dict = None,
                 boundary_velocities : dict = None
                 ) -> None:
        """
        Initialize the LBM simulation.

        Parameters
        ----------
        width : int
            Width of the simulation grid.
        height : int
            Height of the simulation grid.
        omega : float, optional
            Relaxation parameter. The default is 1.0.
        viscosity : float, optional
            Viscosity of the fluid. The default is None.
        inital_density_field_yx : np.ndarray, optional
            Initial density field. The default is None. If not defined, the density field is initialized with ones.
        inital_velocity_field_Cyx : np.ndarray, optional
            Initial velocity field. The default is None. If not defined, the velocity field is initialized with zeros.
        boundary_conditions : dict, optional
            Boundary conditions. The default is None. If not defined, the boundary conditions are set to periodic.
        boundary_velocities : dict, optional
            Boundary velocities. The default is None. If not defined, the boundary velocities are set to zero.

        """
        self.width = width
        self.height = height
        self.boundary_conditions = boundary_conditions
        self.valid_boundary_conditions = ["periodic", "bounce_back", "moving_wall"]
        self.boundary_velocities = boundary_velocities
        if self.boundary_velocities is None:
            self.boundary_velocities = {
                "bottom" : 0.0,
                "top" : 0.0,
                "left" : 0.0,
                "right" : 0.0
            }

        # prepare valid boundary conditions and its implications on the grid size
        if self.boundary_conditions is None:
            self.boundary_conditions = {
                "bottom" : "periodic",
                "top" : "periodic",
                "left" : "periodic",
                "right" : "periodic"
            }
        else:
            # check if boundary conditions are valid           
            if self.boundary_conditions["bottom"] == "periodic" and self.boundary_conditions["top"] != "periodic":
                raise ValueError("Bottom boundary condition is periodic but top boundary condition is not. This is not possible.")
            if self.boundary_conditions["bottom"] != "periodic" and self.boundary_conditions["top"] == "periodic":
                raise ValueError("Top boundary condition is periodic but bottom boundary condition is not. This is not possible.")
            if self.boundary_conditions["left"] == "periodic" and self.boundary_conditions["right"] != "periodic":
                raise ValueError("Left boundary condition is periodic but right boundary condition is not. This is not possible.")
            if self.boundary_conditions["left"] != "periodic" and self.boundary_conditions["right"] == "periodic":
                raise ValueError("Right boundary condition is periodic but left boundary condition is not. This is not possible.")

            for key, value in  self.boundary_conditions.items():
                if value not in self.valid_boundary_conditions:
                    raise ValueError("Boundary condition is not valid. Given value: {}, expected: periodic, bounce_back or moving_wall".format(value))
                
                if value != "periodic" and (key == "bottom" or key == "top"):
                    self.height += 1
                if value != "periodic" and (key == "left" or key == "right"):
                    self.width += 1


        self.omega = omega
        self.viscosity = viscosity
        self.density_field_yx = inital_density_field_yx
        self.velocity_field_Cyx = inital_velocity_field_Cyx
        self.f_eq = np.zeros((9, self.height, self.width))
        self.f_iyx = np.zeros((9, self.height, self.width))


        # Initialize the lattice directions. Given in cartesian coordinates C=(x,y) for every lattice dimension i.
        self.lattice_directions_iC = np.array([
                [ 0, 0], # 0 (center)
                [ 1, 0], # 1 (right)
                [ 0, 1], # 2 (up)
                [-1, 0], # 3 (left)
                [ 0,-1], # 4 (down)
                [ 1, 1], # 5 (right-up)
                [-1, 1], # 6 (left-up)
                [-1,-1], # 7 (left-down)
                [ 1,-1]  # 8 (right-down)
             ])
        # Numpy origin is top left corner, so y axis is flipped compared to cartesian coordinate system
        self.lattice_directions_for_numpy = self.lattice_directions_iC * np.array([1, -1])
        self.inverse_direction_indices = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        # Initialize the lattice weights.
        self.lattice_weights_i = np.array([
                4/9, # 0 (center)
                1/9, # 1 (right)
                1/9, # 2 (up)
                1/9, # 3 (left)
                1/9, # 4 (down)
                1/36,# 5 (right-up)
                1/36,# 6 (left-up)
                1/36,# 7 (left-down)
                1/36 # 8 (right-down)
             ])


        # Initialize the density and velocity fields.
        if self.density_field_yx is None:
            self.density_field_yx = np.ones((height, width))
        # else check if dimension of density field is correct
        elif self.density_field_yx.shape != (height, width):
            raise ValueError("Dimension of density field is not correct. Given shape: {}, expected: {}".format(self.density_field_yx.shape, (height, width)))

        if self.velocity_field_Cyx is None:
            self.velocity_field_Cyx = np.zeros((2, height, width))
        # else check if dimension of velocity field is correct
        elif self.velocity_field_Cyx.shape != (2, height, width):
            raise ValueError("Dimension of velocity field is not correct. Given shape: {}, expected: {}".format(self.velocity_field_Cyx.shape, (2, height, width)))

        # update shape of density field and velocity field if boundary conditions are given, use parameter "boundary_conditions" instead of class instance variable because it remains None if not given
        if boundary_conditions is not None:
            # modify density field and velocity field grid according to boundary conditions (0 at dry nodes outside of border)
            # padding in format [[top, bottom], [left, right]]
            padding = [(0,0), (0,0)]
            # check only bottom and left because top and right are implicity given if bottom and left are defined not periodic
            if self.boundary_conditions["bottom"] != "periodic":
                padding[0] = (1, 1)
            if self.boundary_conditions["left"] != "periodic":
                padding[1] = (1, 1)

            padding = tuple(padding)
            # enlarge density field with padding of zeros
            self.density_field_yx = np.pad(self.density_field_yx, padding, mode="constant", constant_values=0)
            # enlarge velocity field with padding of zeros
            velocity_padding = ((0,0), padding[0], padding[1])
            self.velocity_field_Cyx = np.pad(self.velocity_field_Cyx, velocity_padding, mode="constant", constant_values=0)


        # if the viscosity is given, calculate omega else calculate viscosity from omega
        if self.viscosity is not None:
            self.omega = 1 / (3 * self.viscosity + 0.5)
        else:
            self.viscosity = (1 / 3) * (1 / self.omega - 0.5)

        # check that omega is in the correct range
        if self.omega < 0 or self.omega > 2:
            raise ValueError("Omega is not in the correct range. Given value: {}, expected: 0 <= omega <= 2".format(self.omega))

        # Initialize the equilibrium distribution function.
        self.update_equilibrium_distribution_function()

        #  Initialize the probability density function with the equilibrium distribution function.
        self.f_iyx = np.array(self.f_eq)

    def update_density_field(self):
        """
        Update the density field with current values of probability density function.
        """
        self.density_field_yx = np.sum(self.f_iyx, axis=0)

    def update_velocity_field(self):
        """
        Update the average velocity field with current values of probability density function.
        """
        # out and where parameters are used to handle division by zero
        self.velocity_field_Cyx = np.divide(np.einsum("iyx, iC->Cyx", self.f_iyx, self.lattice_directions_iC), self.density_field_yx[np.newaxis, ...], out=np.zeros_like(self.velocity_field_Cyx), where=self.density_field_yx != 0)


    def update_equilibrium_distribution_function(self):
        """
        Update the equilibrium distribution function with current values of velocity and density fields.
        """
        # precompute terms for equilibrium distribution
        u_norm_squared_yx = np.einsum("Cyx, Cyx -> yx", self.velocity_field_Cyx, self.velocity_field_Cyx)
        uc_iyx = np.einsum("Cyx, iC -> iyx", self.velocity_field_Cyx, self.lattice_directions_iC)

        # update equilibrium distribution
        self.f_eq = np.einsum('i, yx->iyx', self.lattice_weights_i, self.density_field_yx) * (1 + 3 * uc_iyx + 4.5 * uc_iyx**2 - 1.5 * u_norm_squared_yx)


    def boundary_handling_after_streaming(self):
        """
        Boundary handling after streaming step.
        """
        # Remember that grid has origin in the bottom left corner and numpy origin is in the top left corner.

        # apply bounce back boundary condition
        # care because numpy origin (top-left) is used here, not cartesian coordinate system
        if self.boundary_conditions["bottom"] != "periodic":
            self.f_iyx[self.inverse_direction_indices[4], -2, :] = self.f_iyx[4, -1, :]
            self.f_iyx[self.inverse_direction_indices[7], -2, :] = np.roll(self.f_iyx[7, -1, :], shift=1)
            self.f_iyx[self.inverse_direction_indices[8], -2, :] = np.roll(self.f_iyx[8, -1, :], shift=-1)
            self.f_iyx[4, -1, :] = 0
            self.f_iyx[7, -1, :] = 0
            self.f_iyx[8, -1, :] = 0
        
        if self.boundary_conditions["top"] != "periodic":
            self.f_iyx[self.inverse_direction_indices[2], 1, :] = self.f_iyx[2, 0, :]
            self.f_iyx[self.inverse_direction_indices[5], 1, :] = np.roll(self.f_iyx[5, 0, :], shift=-1)
            self.f_iyx[self.inverse_direction_indices[6], 1, :] = np.roll(self.f_iyx[6, 0, :], shift=1)
            self.f_iyx[2, 0, :] = 0
            self.f_iyx[5, 0, :] = 0
            self.f_iyx[6, 0, :] = 0
        
        if self.boundary_conditions["left"] != "periodic":
            self.f_iyx[self.inverse_direction_indices[3], :, 1] = self.f_iyx[3, :, 0]
            self.f_iyx[self.inverse_direction_indices[6], :, 1] = np.roll(self.f_iyx[6, :, 0], shift=1)
            self.f_iyx[self.inverse_direction_indices[7], :, 1] = np.roll(self.f_iyx[7, :, 0], shift=-1)
            self.f_iyx[3, :, 0] = 0
            self.f_iyx[6, :, 0] = 0
            self.f_iyx[7, :, 0] = 0

        if self.boundary_conditions["right"] != "periodic":
            self.f_iyx[self.inverse_direction_indices[1], :, -2] = self.f_iyx[1, :, -1]
            self.f_iyx[self.inverse_direction_indices[5], :, -2] = np.roll(self.f_iyx[5, :, -1], shift=1)
            self.f_iyx[self.inverse_direction_indices[8], :, -2] = np.roll(self.f_iyx[8, :, -1], shift=-1)
            self.f_iyx[1, :, -1] = 0
            self.f_iyx[5, :, -1] = 0
            self.f_iyx[8, :, -1] = 0
        
        
        self.update_density_field()
        # apply moving wall boundary condition
        # positive velocity direction is to right and up 
        if self.boundary_conditions["top"] == "moving_wall":
            self.f_iyx[7, 1, :] -= 6 * self.lattice_weights_i[5] * self.boundary_velocities["top"] * self.density_field_yx[1, :]
            # += because the direction introduces a minus sign and this cancles out the minus sign of the original formula
            self.f_iyx[8, 1, :] += 6 * self.lattice_weights_i[6] * self.boundary_velocities["top"] * self.density_field_yx[1, :]

        if self.boundary_conditions["bottom"] == "moving_wall":
            self.f_iyx[5, -2, :] += 6 * self.lattice_weights_i[7] * self.boundary_velocities["bottom"] * self.density_field_yx[-2, :]
            self.f_iyx[6, -2, :] -= 6 * self.lattice_weights_i[8] * self.boundary_velocities["bottom"] * self.density_field_yx[-2, :]

        if self.boundary_conditions["left"] == "moving_wall":
            self.f_iyx[8, :, 1] -= 6 * self.lattice_weights_i[6] * self.boundary_velocities["left"] * self.density_field_yx[:, 1]
            self.f_iyx[5, :, 1] += 6 * self.lattice_weights_i[7] * self.boundary_velocities["left"] * self.density_field_yx[:, 1]

        if self.boundary_conditions["right"] == "moving_wall":
            self.f_iyx[7, :, -2] -= 6 * self.lattice_weights_i[5] * self.boundary_velocities["right"] * self.density_field_yx[:, -2]
            self.f_iyx[6, :, -2] += 6 * self.lattice_weights_i[8] * self.boundary_velocities["right"] * self.density_field_yx[:, -2]

    def streaming(self):
        """
        Streaming step of the LBM.
        """
        # shift the probability density function in the direction of the lattice directions
        # start at index 1 because direction 0 (0,0) does not change anything with roll call, axis=(1,0) means first in x direction, then in y direction because f is defined by row, col (y,x) indices 
        # numpy origin is top left corner, so y axis is flipped compared to cartesian coordinate system --> therefore lattice_directions_iC in y direction is also flipped --> lattice_directions_for_numpy is being used
        for i in range(1, len(self.lattice_directions_iC)): self.f_iyx[i] = np.roll(self.f_iyx[i], shift=self.lattice_directions_for_numpy[i], axis=(1,0))

    def collision(self):
        """
        Collision step of the LBM.
        """
        self.update_density_field()
        self.update_velocity_field()
        self.update_equilibrium_distribution_function()
        # update the probability density function with the equilibrium distribution function
        self.f_iyx = (1 - self.omega) * self.f_iyx + self.omega * self.f_eq
        
    def step(self):
        """
        One step of the LBM.
        """
        self.streaming()
        self.boundary_handling_after_streaming()
        self.collision()
        

    def run_simulation(self, timesteps : int = 100):
        """
        Run the simulation for a given number of timesteps.
        """
        for i in range(timesteps):
            self.step()

if __name__ == "__main__":
    lbm = LBM(15, 10)
    print(lbm.f_iyx.shape)
    lbm.step()
    print(lbm.f_iyx.shape)
            