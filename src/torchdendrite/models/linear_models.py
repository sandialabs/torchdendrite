from .modules import DendriticLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn

class DendFCNetRegressor(nn.Module):
    def __init__(self, resolution=50):
        super().__init__()
        self.fc1 = DendriticLinear(1, 20, resolution=resolution)
        self.fc2 = DendriticLinear(20, 40, resolution=resolution)
        self.fc3 = DendriticLinear(40, 20, resolution=resolution)
        self.fc4 = DendriticLinear(20, 1, resolution=resolution)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x).reshape(-1,1)
        return x

class DendFCNetClassifier(nn.Module):
    def __init__(self, in_features, num_classes, resolution=50, dt=0.001):
        super().__init__()
        self.fc1 = DendriticLinear(in_features, 20, resolution=resolution, dt=dt)
        self.fc2 = DendriticLinear(20, 40, resolution=resolution, dt=dt)
        self.fc3 = DendriticLinear(40, 20, resolution=resolution, dt=dt)
        self.fc4 = DendriticLinear(20, num_classes, resolution=resolution, dt=dt)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

class MNISTDendFCNet(nn.Module):
    def __init__(self, resolution=50, dt=0.001):
        super().__init__() 
        self.fc1 = DendriticLinear(28*28, 64, resolution=resolution, dt=dt) 
        self.fc2 = DendriticLinear(64, 64, resolution=resolution, dt=dt)
        self.fc3 = DendriticLinear(64, 64, resolution=resolution, dt=dt)
        self.fc4 = DendriticLinear(64, 10, resolution=resolution, dt=dt) 
        
    def forward(self, x):
        """
        Activation Function: ReLU
        """
        x = x.view(-1, 28*28)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

class MNISTDendFCLeakyNet(nn.Module):
    def __init__(self, resolution=50, beta=0.95, num_steps=20):
        super().__init__() 

        self.num_steps = num_steps
        
        self.fc1 = DendriticLinear(28*28, 64, resolution=resolution)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = DendriticLinear(64, 64, resolution=resolution)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = DendriticLinear(64, 64, resolution=resolution)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc4 = DendriticLinear(64, 10, resolution=resolution)
        self.lif4 = snn.Leaky(beta=beta)
        
    def forward(self, x):
        """
        Activation Function: ReLU
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_rec = []
        mem_rec = []

        x = x.view(-1, 28*28)
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            
            spk_rec.append(spk4)
            mem_rec.append(mem4)
            
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)





        
        