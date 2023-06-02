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
                 ) -> None:
        """
        Initialize the LBM simulation.
        """
        self.width = width
        self.height = height
        self.omega = omega
        self.viscosity = viscosity
        self.density_field_yx = inital_density_field_yx
        self.velocity_field_Cyx = inital_velocity_field_Cyx
        self.f_eq = np.zeros((9, height, width))
        self.f_iyx = np.zeros((9, height, width))

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
        self.velocity_field_Cyx = np.divide(np.einsum("iyx, iC->Cyx", self.f_iyx, self.lattice_directions_iC), self.density_field_yx[np.newaxis, ...])


    def update_equilibrium_distribution_function(self):
        """
        Update the equilibrium distribution function with current values of velocity and density fields.
        """
        # precompute terms for equilibrium distribution
        u_norm_squared_yx = np.einsum("Cyx, Cyx -> yx", self.velocity_field_Cyx, self.velocity_field_Cyx)
        uc_iyx = np.einsum("Cyx, iC -> iyx", self.velocity_field_Cyx, self.lattice_directions_iC)

        # update equilibrium distribution
        self.f_eq = np.einsum('i, yx->iyx', self.lattice_weights_i, self.density_field_yx) * (1 + 3 * uc_iyx + 4.5 * uc_iyx**2 - 1.5 * u_norm_squared_yx)

    def streaming(self):
        """
        Streaming step of the LBM.
        """
        # shift the probability density function in the direction of the lattice directions
        # start at index 1 because direction 0 (0,0) does not change anything with roll call, axis=(1,0) means first in x direction, then in y direction because f is defined by row, col (y,x) indices 
        for i in range(1, len(self.lattice_directions_iC)): self.f_iyx[i] = np.roll(self.f_iyx[i], shift=self.lattice_directions_iC[i], axis=(1,0))

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
            