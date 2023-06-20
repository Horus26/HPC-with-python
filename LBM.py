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
                 boundary_velocities : dict = None,
                 boundary_pressure : dict = None
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

        self.c_s_squared = 1/3
        self.width = width
        self.height = height
        self.boundary_conditions = boundary_conditions
        self.boundary_pressure = boundary_pressure
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

            # check if boundary pressure is given correctly
            if self.boundary_pressure is not None:
                if (self.boundary_pressure["bottom"] is None) != (self.boundary_pressure["top"] is None):
                    raise ValueError("Bottom and top boundary pressure are different types. This is not possible.")
                if (self.boundary_pressure["left"] is None) != (self.boundary_pressure["right"] is None):
                    raise ValueError("Left and right boundary pressure are different types. This is not possible.")

            # # enlarge the grid depending on the boundary conditions
            # for key, value in  self.boundary_conditions.items():
            #     if value not in self.valid_boundary_conditions:
            #         raise ValueError("Boundary condition is not valid. Given value: {}, expected: periodic, bounce_back or moving_wall".format(value))
                
            #     # also change height if periodic and pressure boundary conditions are given
            #     if (value != "periodic" or ((key=="bottom" or key == "top") and self.boundary_pressure is not None and self.boundary_pressure[key] is not None)) and (key == "bottom" or key == "top"):
            #         self.height += 1
                
            #     # also change width if periodic and pressure boundary conditions are given
            #     if (value != "periodic" or ((key=="left" or key == "right") and self.boundary_pressure is not None and self.boundary_pressure[key] is not None)) and (key == "left" or key == "right"):
            #         self.width += 1

        # always enlarge the grid
        self.height += 2
        self.width += 2




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


        # Initialize the density and velocity fields. Width and height are used here on purpose instead of self.width and self.height because those are larger for handling boundary conditions.
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

        # modify density field and velocity field grid according to boundary conditions (0 at dry nodes outside of border)
        # padding in format [[top, bottom], [left, right]]
        padding = [(1,1), (1,1)]
        # boundary specific code not needed here anymore because all boundary conditions are handled with dry nodes outside of border
        # # check only bottom and left because top and right are implicity given if bottom and left are defined not periodic
        # if self.boundary_conditions["bottom"] != "periodic":
        #     padding[0] = (1, 1)
        # if self.boundary_conditions["left"] != "periodic":
        #     padding[1] = (1, 1)
        padding = tuple(padding)
        # enlarge density field with padding of zeros
        self.density_field_yx = np.pad(self.density_field_yx, padding, mode="constant", constant_values=0)
        # enlarge velocity field with padding of zeros. The (0,0) is there so that the cartesian coordinate dimension is not padded.
        velocity_padding = ((0,0), padding[0], padding[1])
        self.velocity_field_Cyx = np.pad(self.velocity_field_Cyx, velocity_padding, mode="constant", constant_values=0)

        # if the viscosity is given, calculate omega else calculate viscosity from omega
        if self.viscosity is not None:
            self.omega = 1 / ((1/self.c_s_squared) * self.viscosity + 0.5)
        else:
            self.viscosity = self.c_s_squared * (1 / self.omega - 0.5)

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


    def get_density_field_yx(self, without_boundary = True):
        """
        Get the density field.

        Parameters
        ----------
        without_boundary : bool, optional
            If true, the density field without the boundary is returned. The default is True.

        Returns
        -------
        np.ndarray
            The density field.

        """
        if without_boundary:
            return self.density_field_yx[1:-1, 1:-1]
        else:
            return self.density_field_yx
        
    def get_velocity_field_Cyx(self, without_boundary = True):
        """
        Get the velocity field.

        Parameters
        ----------
        without_boundary : bool, optional
            If true, the velocity field without the boundary is returned. The default is True.

        Returns
        -------
        np.ndarray
            The velocity field.

        """
        if without_boundary:
            return self.velocity_field_Cyx[:, 1:-1, 1:-1]
        else:
            return self.velocity_field_Cyx

    def calculate_equilibrium_distribution_function(self, density_field_yx : np.ndarray, velocity_field_Cyx : np.ndarray):
        """
        Calculate the equilibrium distribution function with given density and velocity fields.

        Parameters
        ----------
        density_field_yx : np.ndarray
            Density field.
        velocity_field_Cyx : np.ndarray
            Velocity field.

        Returns
        -------
        np.ndarray
            Equilibrium distribution function.

        """
        # precompute terms for equilibrium distribution
        u_norm_squared_yx = np.einsum("Cyx, Cyx -> yx", velocity_field_Cyx, velocity_field_Cyx)
        uc_iyx = np.einsum("Cyx, iC -> iyx", velocity_field_Cyx, self.lattice_directions_iC)

        # update equilibrium distribution
        return np.einsum('i, yx->iyx', self.lattice_weights_i, density_field_yx) * (1 + 3 * uc_iyx + 4.5 * uc_iyx**2 - 1.5 * u_norm_squared_yx)

    def calculate_equilibrium_distribution_for_pressure_boundary(self, density_value : np.ndarray, velocity_dimension : np.ndarray):
        """
        Calculate the equilibrium distribution function for a pressure boundary condition.

        Parameters
        ----------
        density_value : np.ndarray
            Density value.
        velocity_dimension : np.ndarray
            Velocity dimension.

        Returns
        -------
        np.ndarray
            Equilibrium distribution function.

        """
        u_norm_squared_yx = np.einsum("Cd, Cd -> yx", velocity_dimension, velocity_dimension)
        uc_iyx = np.einsum("Cd, iC -> id", velocity_dimension, self.lattice_directions_iC)

        return density_value * self.lattice_weights_i * (1 + 3 * uc_iyx + 4.5 * uc_iyx**2 - 1.5 * u_norm_squared_yx)


    def update_equilibrium_distribution_function(self):
        """
        Update the equilibrium distribution function with current values of velocity and density fields.
        """
        # precompute terms for equilibrium distribution
        u_norm_squared_yx = np.einsum("Cyx, Cyx -> yx", self.velocity_field_Cyx, self.velocity_field_Cyx)
        uc_iyx = np.einsum("Cyx, iC -> iyx", self.velocity_field_Cyx, self.lattice_directions_iC)

        # update equilibrium distribution
        self.f_eq = np.einsum('i, yx->iyx', self.lattice_weights_i, self.density_field_yx) * (1 + 3 * uc_iyx + 4.5 * uc_iyx**2 - 1.5 * u_norm_squared_yx)

    def boundary_handling_before_streaming(self):
        """
        Boundary handling before streaming step.
        """

        # only do boundary handling before streaming if boundary pressure is given
        if self.boundary_pressure is None:
            return
        
        # pressure boundary condition for horizontal pressure
        if self.boundary_conditions["left"] == "periodic" and self.boundary_pressure["left"] is not None:
            # get population of left boundary
            f_left_iy = self.f_iyx[:, :, 1]
            f_right_iy = self.f_iyx[:, :, -2]

            f_start_pipe = None
            f_end_pipe = None
            rho_input = None
            rho_output = None
            input_index = None
            output_index = None
            input_border_index = None
            output_border_index = None
            if self.boundary_pressure["input"] == "left":
                f_start_pipe = f_left_iy
                f_end_pipe = f_right_iy
                input_pressure = self.boundary_pressure["left"]
                output_pressure = self.boundary_pressure["right"]
                rho_input = (output_pressure + (output_pressure - input_pressure)) / self.c_s_squared
                rho_output = output_pressure / self.c_s_squared
                input_index = 1
                output_index = -2
                input_border_index = 0
                output_border_index = -1

            else:
                f_start_pipe = f_right_iy
                f_end_pipe = f_left_iy
                input_pressure = self.boundary_pressure["right"]
                output_pressure = self.boundary_pressure["left"]
                rho_input = (output_pressure + (output_pressure - input_pressure)) / self.c_s_squared
                rho_output = output_pressure / self.c_s_squared
                input_index = -2
                output_index = 1
                input_border_index = -1
                output_border_index = 0


            self.update_velocity_field()
            # calculate equilibrium distribution function of pipe with input pressure
            f_eq_pipe_input_pressure = self.calculate_equilibrium_distribution_for_pressure_boundary(rho_input, self.velocity_field_Cyx[:, :, output_index])
            # calculate equilibrium distribution function of pipe with output pressure
            f_eq_pipe_output_pressure = self.calculate_equilibrium_distribution_for_pressure_boundary(rho_output, self.velocity_field_Cyx[:, :, input_index])

            self.update_density_field()
            self.update_velocity_field()
            self.update_equilibrium_distribution_function()

            self.f_iyx[:, :, input_border_index] = f_eq_pipe_input_pressure + (f_end_pipe - self.f_eq[:, :, output_index])
            self.f_iyx[:, :, output_border_index] = f_eq_pipe_output_pressure + (f_start_pipe - self.f_eq[:, :, input_index])



    def boundary_handling_after_streaming(self):
        """
        Boundary handling after streaming step.
        """
        self.update_density_field()


        # Remember that grid has origin in the bottom left corner and numpy origin is in the top left corner.
        # care because numpy origin (top-left) is used here, not cartesian coordinate system

        # bounce back boundary condition, also used in moving wall case
        if self.boundary_conditions["bottom"] != "periodic":
            self.f_iyx[self.inverse_direction_indices[4], -2, :] = self.f_iyx[4, -1, :]
            self.f_iyx[self.inverse_direction_indices[7], -2, :] = np.roll(self.f_iyx[7, -1, :], shift=1)
            self.f_iyx[self.inverse_direction_indices[8], -2, :] = np.roll(self.f_iyx[8, -1, :], shift=-1)
            self.f_iyx[4, -1, :] = 0
            self.f_iyx[7, -1, :] = 0
            self.f_iyx[8, -1, :] = 0

        # basic periodic boundary condition.
        else:
            # explicit handling because layer of nodes around grid is also filled by periodic boundary condition and needs to be handled
            self.f_iyx[4, 1, :] = self.f_iyx[4, -1, :]
            self.f_iyx[7, 1, :] = self.f_iyx[7, -1, :]
            self.f_iyx[8, 1, :] = self.f_iyx[8, -1, :]
            self.f_iyx[4, -1, :] = 0
            self.f_iyx[7, -1, :] = 0
            self.f_iyx[8, -1, :] = 0

        
        # bounce back boundary condition, also used in moving wall case
        if self.boundary_conditions["top"] != "periodic":
            self.f_iyx[self.inverse_direction_indices[2], 1, :] = self.f_iyx[2, 0, :]
            self.f_iyx[self.inverse_direction_indices[5], 1, :] = np.roll(self.f_iyx[5, 0, :], shift=-1)
            self.f_iyx[self.inverse_direction_indices[6], 1, :] = np.roll(self.f_iyx[6, 0, :], shift=1)
            self.f_iyx[2, 0, :] = 0
            self.f_iyx[5, 0, :] = 0
            self.f_iyx[6, 0, :] = 0

        # basic periodic boundary condition.
        else:
            # explicit handling because additional boundary layer of nodes around grid is also filled by periodic boundary condition and needs to be handled
            self.f_iyx[2, -2, :] = self.f_iyx[2, 0, :]
            self.f_iyx[5, -2, :] = self.f_iyx[5, 0, :]
            self.f_iyx[6, -2, :] = self.f_iyx[6, 0, :]
            self.f_iyx[2, 0, :] = 0
            self.f_iyx[5, 0, :] = 0
            self.f_iyx[6, 0, :] = 0



        # bounce back boundary condition, also used in moving wall case
        if self.boundary_conditions["left"] != "periodic":
            self.f_iyx[self.inverse_direction_indices[3], :, 1] = self.f_iyx[3, :, 0]
            self.f_iyx[self.inverse_direction_indices[6], :, 1] = np.roll(self.f_iyx[6, :, 0], shift=1)
            self.f_iyx[self.inverse_direction_indices[7], :, 1] = np.roll(self.f_iyx[7, :, 0], shift=-1)
            self.f_iyx[3, :, 0] = 0
            self.f_iyx[6, :, 0] = 0
            self.f_iyx[7, :, 0] = 0

        # basic periodic boundary condition.
        else:
            # explicit handling because additional boundary layer of nodes around grid is also filled by periodic boundary condition and needs to be handled
            self.f_iyx[3, :, -2] = self.f_iyx[3, :, 0]
            self.f_iyx[6, :, -2] = self.f_iyx[6, :, 0]
            self.f_iyx[7, :, -2] = self.f_iyx[7, :, 0]
            self.f_iyx[3, :, 0] = 0
            self.f_iyx[6, :, 0] = 0
            self.f_iyx[7, :, 0] = 0


        # bounce back boundary condition, also used in moving wall case
        if self.boundary_conditions["right"] != "periodic":
            self.f_iyx[self.inverse_direction_indices[1], :, -2] = self.f_iyx[1, :, -1]
            self.f_iyx[self.inverse_direction_indices[5], :, -2] = np.roll(self.f_iyx[5, :, -1], shift=1)
            self.f_iyx[self.inverse_direction_indices[8], :, -2] = np.roll(self.f_iyx[8, :, -1], shift=-1)
            self.f_iyx[1, :, -1] = 0
            self.f_iyx[5, :, -1] = 0
            self.f_iyx[8, :, -1] = 0

        # basic periodic boundary condition.
        else:
            # explicit handling because additional boundary layer of nodes around grid is also filled by periodic boundary condition and needs to be handled
            self.f_iyx[1, :, 1] = self.f_iyx[1, :, -1]
            self.f_iyx[5, :, 1] = self.f_iyx[5, :, -1]
            self.f_iyx[8, :, 1] = self.f_iyx[8, :, -1]
            self.f_iyx[1, :, -1] = 0
            self.f_iyx[5, :, -1] = 0
            self.f_iyx[8, :, -1] = 0

        
        # apply moving wall boundary condition
        # positive velocity direction is to right and up 
        if self.boundary_conditions["top"] == "moving_wall":
            density_top = self.f_iyx[0, 1] + self.f_iyx[1, 1] + self.f_iyx[3, 1] + 2 * (self.f_iyx[2, 1] + self.f_iyx[6, 1] + self.f_iyx[5, 1])
            self.f_iyx[7, 1, :] += 0.5 * (self.f_iyx[1, 1, :] - self.f_iyx[3, 1, :]) - 0.5 * density_top * self.boundary_velocities["top"]
            self.f_iyx[8, 1, :] += 0.5 * (self.f_iyx[3, 1, :] - self.f_iyx[1, 1, :]) + 0.5 * density_top * self.boundary_velocities["top"]

        if self.boundary_conditions["bottom"] == "moving_wall":
            density_bottom = self.f_iyx[0, -2] + self.f_iyx[1, -2] + self.f_iyx[3, -2] + 2 * (self.f_iyx[4, -2] + self.f_iyx[7, -2] + self.f_iyx[8, -2])
            self.f_iyx[5, -2, :] += 0.5 * (self.f_iyx[3, -2, :] - self.f_iyx[1, -2, :]) + 0.5 * density_bottom * self.boundary_velocities["bottom"]
            self.f_iyx[6, -2, :] += 0.5 * (self.f_iyx[1, -2, :] - self.f_iyx[3, -2, :]) - 0.5 * density_bottom * self.boundary_velocities["bottom"]

        if self.boundary_conditions["left"] == "moving_wall":
            density_left = self.f_iyx[0, :, 1] + self.f_iyx[2, :, 1] + self.f_iyx[4, :, 1] + 2 * (self.f_iyx[3, :, 1] + self.f_iyx[7, :, 1] + self.f_iyx[6, :, 1])
            self.f_iyx[8, :, 1] += 0.5 * (self.f_iyx[2, :, 1] - self.f_iyx[4, :, 1]) - 0.5 * density_left * self.boundary_velocities["left"]
            self.f_iyx[5, :, 1] += 0.5 * (self.f_iyx[4, :, 1] - self.f_iyx[2, :, 1]) + 0.5 * density_left * self.boundary_velocities["left"]

        if self.boundary_conditions["right"] == "moving_wall":
            density_right = self.f_iyx[0, :, -2] + self.f_iyx[2, :, -2] + self.f_iyx[4, :, -2] + 2 * (self.f_iyx[1, :, -2] + self.f_iyx[5, :, -2] + self.f_iyx[8, :, -2])
            self.f_iyx[3, :, -2] += 0.5 * (self.f_iyx[4, :, -2] - self.f_iyx[2, :, -2]) + 0.5 * density_right * self.boundary_velocities["right"]
            self.f_iyx[7, :, -2] += 0.5 * (self.f_iyx[2, :, -2] - self.f_iyx[4, :, -2]) - 0.5 * density_right * self.boundary_velocities["right"]

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
    # print(lbm.f_iyx.shape)
    lbm.step()
    # print(lbm.f_iyx.shape)
            