import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DendriticLinear


class DendFCNetRegressor(nn.Module):
    def __init__(self, resolution=50):
        """
        Initializes the DendFCNetRegressor model.

        Parameters:
        resolution (int): The resolution parameter for the DendriticLinear layers. Default is 50.
        """
        super().__init__()
        self.fc1 = DendriticLinear(1, 20, resolution=resolution)
        self.fc2 = DendriticLinear(20, 40, resolution=resolution)
        self.fc3 = DendriticLinear(40, 20, resolution=resolution)
        self.fc4 = DendriticLinear(20, 1, resolution=resolution)

    def forward(self, x):
        """
        Defines the forward pass of the DendFCNetRegressor model.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor after passing through the network.
        """
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x).reshape(-1, 1)
        return x


class DendFCNetClassifier(nn.Module):
    def __init__(self, in_features, num_classes, resolution=50, dt=0.001):
        """
        Initializes the DendFCNetClassifier model.

        Parameters:
        in_features (int): The number of input features.
        num_classes (int): The number of output classes.
        resolution (int): The resolution parameter for the DendriticLinear layers. Default is 50.
        dt (float): The time step parameter for the DendriticLinear layers. Default is 0.001.
        """
        super().__init__()
        self.fc1 = DendriticLinear(in_features, 20, resolution=resolution, dt=dt)
        self.fc2 = DendriticLinear(20, 40, resolution=resolution, dt=dt)
        self.fc3 = DendriticLinear(40, 20, resolution=resolution, dt=dt)
        self.fc4 = DendriticLinear(20, num_classes, resolution=resolution, dt=dt)

    def forward(self, x):
        """
        Defines the forward pass of the DendFCNetClassifier model.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor after passing through the network.
        """
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x


class MNISTDendFCNet(nn.Module):
    def __init__(self, resolution=50, dt=0.001):
        super().__init__()
        self.fc1 = DendriticLinear(28 * 28, 64, resolution=resolution, dt=dt)
        self.fc2 = DendriticLinear(64, 64, resolution=resolution, dt=dt)
        self.fc3 = DendriticLinear(64, 64, resolution=resolution, dt=dt)
        self.fc4 = DendriticLinear(64, 10, resolution=resolution, dt=dt)

    def forward(self, x):
        """
        Defines the forward pass of the MNISTDendFCNet model.

        Parameters:
        x (torch.Tensor): The input tensor, expected to be of shape (batch_size, 1, 28, 28).

        Returns:
        torch.Tensor: The output tensor after passing through the network, of shape (batch_size, 10).
        """
        x = x.view(-1, 28 * 28)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x


class MNISTDendFCLeakyNet(nn.Module):
    def __init__(self, resolution=50, beta=0.95, num_steps=20):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = DendriticLinear(28 * 28, 64, resolution=resolution)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = DendriticLinear(64, 64, resolution=resolution)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = DendriticLinear(64, 64, resolution=resolution)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc4 = DendriticLinear(64, 10, resolution=resolution)
        self.lif4 = snn.Leaky(beta=beta)

    def forward(self, x):
        """
        Defines the forward pass of the MNISTDendFCLeakyNet model.

        This model is a variant of the MNISTDendFCNet that incorporates leaky integration
        and thresholding (LIF) units between the dendritic linear layers.

        Parameters:
        x (torch.Tensor): The input tensor, expected to be of shape (batch_size, 1, 28, 28).

        Returns:
        tuple: A tuple containing:

        - spk_rec (torch.Tensor): A tensor of shape (num_steps, batch_size, 10) containing the
          spike recordings from the output layer for each time step.
        - mem_rec (torch.Tensor): A tensor of shape (num_steps, batch_size, 10) containing the
          membrane potential from the output layer for each time step.
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_rec = []
        mem_rec = []

        # Reshape the input tensor to the expected shape for fully-connected layers
        x = x.view(-1, 28 * 28)

        # Iterate over the specified number of time steps
        for step in range(self.num_steps):
            # Forward pass through the first layer
            cur1 = self.fc1(x)
            # Apply LIF function to get spikes and updated membrane potential
            spk1, mem1 = self.lif1(cur1, mem1)
            # Forward pass through the second layer
            cur2 = self.fc2(spk1)
            # Apply LIF function
            spk2, mem2 = self.lif2(cur2, mem2)
            # Forward pass through the third layer
            cur3 = self.fc3(spk2)
            # Apply LIF function
            spk3, mem3 = self.lif3(cur3, mem3)
            # Forward pass through the output layer
            cur4 = self.fc4(spk3)
            # Apply LIF function
            spk4, mem4 = self.lif4(cur4, mem4)

            # Append the spikes and membrane potentials to the recordings
            spk_rec.append(spk4)
            mem_rec.append(mem4)

        # Convert the list of spikes and membrane potentials to tensors
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
