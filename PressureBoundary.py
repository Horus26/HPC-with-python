
class PressureBoundary:
    def __init__(self, rho_input, rho_output, input_border_update_channels, output_border_update_channels, input_index=1, output_index=-2, input_border_index=0, output_border_index=-1):
        """
        Stores the information about the pressure boundary.

        Parameters
        ----------
        rho_input : float
            The density at the inlet.
        rho_output : float
            The density at the outlet.
        input_border_update_channels : list
            The channels to update at the inlet.
        output_border_update_channels : list
            The channels to update at the outlet.
        input_index : int, optional
            The first index of the pipe. The default is 1.
        output_index : int, optional
            The last index of the pipe. The default is -2.
        input_border_index : int, optional
            The index of the inlet border. The default is 0.
        output_border_index : int, optional
            The index of the outlet border. The default is -1.
        """        
        
        self.rho_input = rho_input
        self.rho_output = rho_output
        self.input_border_update_channels = input_border_update_channels
        self.output_border_update_channels = output_border_update_channels
        self.input_index = input_index
        self.output_index = output_index
        self.input_border_index = input_border_index
        self.output_border_index = output_border_index
        