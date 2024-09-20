import torch
import torch.nn as nn

class DendriticLinear(nn.Module):
    
    def __init__(self, 
                 in_features, 
                 out_features,
                 max_constant_threshold=2,
                 constant_init_std=0.1,
                 resolution=10, 
                 dt=0.001,
                 decay_per_neuron=True):

        super(DendriticLinear, self).__init__()

        self.dt = dt
        self.in_features = in_features
        self.out_features = out_features
        self.resolution = resolution
        self.max_constant = max_constant_threshold
        self.init_std = constant_init_std

        ### Initialize Dendrite Constants ###
        dendrite_currents = torch.zeros(1, out_features, in_features)
        input_currents = torch.zeros(1, out_features, in_features)
        soma_potential = torch.zeros(1, out_features, 1)

        self.register_buffer("dendrite_currents", dendrite_currents)
        self.register_buffer("input_currents", input_currents)
        self.register_buffer("soma_potential", soma_potential)

        ### Define Learnable Parameters ###
        self.dendrite_weights = nn.Parameter(torch.randn(out_features, in_features), requires_grad=True)
        self.time_constants = nn.Parameter(torch.normal(0., self.init_std, size=(out_features, in_features)), requires_grad=True)
        self.space_constants = nn.Parameter(torch.normal(0., self.init_std, size=(out_features, in_features)), requires_grad=True)
        self.dend_decay = nn.Parameter(torch.normal(0, self.init_std, size=(out_features if decay_per_neuron else 1, 1)), requires_grad=True)
        

    
    def forward_dendrite(self, x):

        ### Check input shape ###
        assert x.shape[-1] == self.in_features, f"Number of inputs in tensor x: f{x.shape} does not match in_features {self.in_features}"
        
        ### Get Batch Size ###
        if len(x.shape) == 2:
            batch_size, n_inputs = x.shape
            reshape_output = None
        elif len(x.shape) == 3:
            orig_batch_size, num_patches, n_inputs = x.shape
            batch_size = orig_batch_size * num_patches
            x = x.reshape(batch_size, n_inputs)
            reshape_output = (orig_batch_size, num_patches, self.out_features)
            
        ### Create Clones of Initialized Constant Variables ###
        soma_potential = self.soma_potential.repeat(batch_size, 1, 1)

        ### Ensure Time/Space/Decay constatants are scaled btwn 0 and max_time_space_constant with sigmoid ###
        time_constants = self.time_constants.sigmoid() * self.max_constant
        space_constants = self.space_constants.sigmoid() * self.max_constant
        dend_decay = (self.dend_decay.sigmoid() * self.max_constant).repeat(1,self.in_features)
        
        dendrite_currents = torch.zeros(size=(batch_size, self.out_features, self.in_features), device=x.device)
        input_currents = torch.zeros(size=(batch_size, self.out_features, self.in_features), device=x.device)

        ### We will push data x through multiple dendrites, to vectorize this we need to make copies of x ###
        x = x.unsqueeze(1).repeat(1, self.out_features, 1)


        ### Couple the Resolution step with timesteps on the Leaky ###
        for t in range(self.resolution):

            ### Propagate Dendrite Current to Next Resolution Step ###
            updated_dendrite_currents = dendrite_currents * time_constants 
            
            ### Update Neighboring Current ###
            neighbor_currents = dendrite_currents * space_constants 
    
            ### Send Current Through Soma ###
            input_currents[:, :, 1:] = input_currents[:, :, 1:] + neighbor_currents[:, :, :-1]
            updated_dendrite_currents[:, :, :-1] = updated_dendrite_currents[:, :, :-1] - neighbor_currents[:, :, :-1]

            ### Send Currents Away from Soma ###
            input_currents[:, :, :-1] = input_currents[:, :, :-1] + neighbor_currents[:, :, 1:]
            updated_dendrite_currents[:, :, 1:] = updated_dendrite_currents[:, :, 1:] - neighbor_currents[:, :, 1:]

            ### Push Current "Data" into the Input Currents ###
            input_currents = input_currents + (x * self.dendrite_weights * self.dt)
           
            ### Update Dendrite Currents with new Input Currents ###
            updated_dendrite_currents = updated_dendrite_currents + input_currents

            ### Accumulate Updated Dendrite Currents over Space ###
            accumulated_updated_dendrite_currents = (updated_dendrite_currents * space_constants).sum(axis=-1, keepdim=True)
            
            ### Accumulate Soma Potential over Micro-Timestep ###
            soma_potential = soma_potential + accumulated_updated_dendrite_currents
            
            ### Dendrite Current Leakage ###
            dendrite_currents = updated_dendrite_currents * dend_decay * self.dt

        if reshape_output is None:
            return soma_potential.squeeze()
        else:
            return soma_potential.squeeze().reshape(*reshape_output)
        
    def forward(self, x):
        
        ### Iterate Through Sequence Dim of Data ###
        soma_potential = self.forward_dendrite(x)
        
        return soma_potential

class DendriticConv2d(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=0, 
                 max_constant_threshold=2,
                 constant_init_std=0.1, 
                 resolution=10, 
                 dt=0.001, 
                 decay_per_neuron=True):
        
        super(DendriticConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.filter_weights = DendriticLinear(in_channels*kernel_size**2, 
                                              out_channels,
                                              max_constant_threshold=max_constant_threshold,
                                              constant_init_std=constant_init_std,
                                              resolution=resolution, 
                                              dt=dt,
                                              decay_per_neuron=decay_per_neuron)

    def _compute_output_size(self, in_size):
        return int(((in_size - self.kernel_size + 2*self.padding) / self.stride) + 1)
        
    def forward(self, x):
        
        batch, channels, height, width = x.shape

        ### Compute Output Shape of Convolution ###
        h_out = self._compute_output_size(height)
        w_out = self._compute_output_size(width)

        ### Unfold Data ###
        unfolded = torch.nn.functional.unfold(x, 
                                              kernel_size=self.kernel_size, 
                                              stride=self.stride, 
                                              padding=self.padding)
        unfolded = unfolded.transpose(1,2)

        ### Multiply by Filter Coefficients ###
        out_unfolded = self.filter_weights(unfolded).transpose(1,2)

        ### Fold Data ###
        out = torch.nn.functional.fold(out_unfolded, (h_out, w_out), (1, 1))

        return out
        