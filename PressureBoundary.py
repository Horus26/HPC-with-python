
class PressureBoundary:
    def __init__(self, f_start_pipe_index_yx, f_end_pipe_index_yx, input_pressure, output_pressure, rho_input, rho_output, input_border_update_channels, output_border_update_channels, input_index=1, output_index=-2, input_border_index=0, output_border_index=-1):
        """
        Parameters
        ----------
        f_start_pipe_index_yx : list
            The y and x indices of the first fluid cell in the pipe. Only one can be non None.
        f_end_pipe_index_yx : list
            The y and x indices of the last fluid cell in the pipe. Only one can be non None.
        input_pressure : float
            The pressure at the inlet.
        output_pressure : float
            The pressure at the outlet.
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
        
        self.f_start_pipe_index_yx = f_start_pipe_index_yx
        self.f_end_pipe_index_yx = f_end_pipe_index_yx
        self.input_pressure = input_pressure
        self.output_pressure = output_pressure
        self.rho_input = rho_input
        self.rho_output = rho_output
        self.input_border_update_channels = input_border_update_channels
        self.output_border_update_channels = output_border_update_channels
        self.input_index = input_index
        self.output_index = output_index
        self.input_border_index = input_border_index
        self.output_border_index = output_border_index

        # check valid inputs
        if self.f_start_pipe_index_yx[0] is None and self.f_start_pipe_index_yx[1] is None:
            raise ValueError("f_start_pipe_index_yx cannot be None for y and x.")
        if self.f_end_pipe_index_yx[0] is None and self.f_end_pipe_index_yx[1] is None:
            raise ValueError("f_end_pipe_index_yx cannot be None for y and x.")
        if self.f_start_pipe_index_yx[0] is not None and self.f_start_pipe_index_yx[1] is not None:
            raise ValueError("f_start_pipe_index_yx cannot be not None for y and x.")
        if self.f_end_pipe_index_yx[0] is not None and self.f_end_pipe_index_yx[1] is not None:
            raise ValueError("f_end_pipe_index_yx cannot be not None for y and x.")
        