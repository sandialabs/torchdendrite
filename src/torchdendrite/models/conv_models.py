from .modules import DendriticConv2d, DendriticLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDendConv(nn.Module):
    def __init__(self, resolution=30, dt=0.001, in_channels=1):
        super(SimpleDendConv,self).__init__()
        self.conv1 = DendriticConv2d(in_channels,10,kernel_size=5,stride=1, resolution=resolution, dt=dt)
        self.conv2 = DendriticConv2d(10,10,kernel_size=5,stride=1, resolution=resolution, dt=dt)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2) #2x2 maxpool
        self.fc1 = DendriticLinear(4*4*10,100, resolution=resolution, dt=dt)
        self.fc2 = DendriticLinear(100,10, resolution=resolution, dt=dt)
      
    def forward(self,x):
        x = F.relu(self.conv1(x)) #24x24x10
        x = self.pool(x) #12x12x10
        x = F.relu(self.conv2(x)) #8x8x10
        x = self.pool(x) #4x4x10    
        x = x.view(-1, 4*4*10) #flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CifarDendConv(nn.Module):
    def __init__(self, resolution=30, dt=0.001, in_channels=3):
        super().__init__()
        self.conv1 = DendriticConv2d(in_channels,6,kernel_size=5,stride=1, resolution=resolution, dt=dt)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = DendriticConv2d(6,16,kernel_size=5,stride=1, resolution=resolution, dt=dt)
        self.fc1 = DendriticLinear(16*5*5, 120, resolution=resolution, dt=dt)
        self.fc2 = DendriticLinear(120, 84, resolution=resolution, dt=dt)
        self.fc3 = DendriticLinear(84, 10, resolution=resolution, dt=dt)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, downsample=None, middle_conv_stride=1, resolution=30, residual=True, dt=0.001):
        """
        This residual block will be reused multiple times to define our model. It consists of 3 convolutional layers, 
        along with Batch Normalization and ReLU. If a downsample is needed, it will also accept a downsampling convolution
        that will ensure our identity is equal to the output before returning. 
        
        in_planes: Expected Number of Input Planes
        planes: Number of Planes to Map to in the Intermediate before expansion
        downsample: Pass in a downsampling function to ensure Identity shape matches X
        middle_conv_stride: The first block in every set of N blocks has a stride of 2 on the second convolution
        residual: Turn the residual sum on or off
        """
        super(ResidualBlock, self).__init__()
        ### Set Convolutional Layers ###
        self.conv1 = DendriticConv2d(in_planes, planes, kernel_size=1, stride=1, resolution=resolution, dt=dt)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DendriticConv2d(planes, planes, kernel_size=3, stride=middle_conv_stride, padding=1, resolution=resolution, dt=dt)
        self.bn2 = nn.BatchNorm2d(planes)
        
        ### Output to planes * 4 as our expansion ###
        self.conv3 = DendriticConv2d(planes, planes*4, kernel_size=1, stride=1, resolution=resolution, dt=dt)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU()
        
        ### This Will Exist if a Downsample Is Needed ###
        self.downsample = downsample
        self.residual = residual
        
    def forward(self, x):
        identity = x # Store the identity function

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.residual:
            if self.downsample is not None: # If our identity function has less channels or larger size we remap it
                identity = self.downsample(identity)
                
            x  = x + identity

        x = self.relu(x)
        return x
    

class DendResNet(nn.Module):
    def __init__(self, 
                 layer_counts, 
                 num_channels=3, 
                 num_classes=2,
                 resolution=30,
                 dt=0.001,
                 residual=True):
        """
        ResNet Implementation (Inspired by PyTorch torchvision.models implementation)
        
        layer_counts: Number of blocks in each set of blocks passed as a list
        num_channels: Number of input channels to model
        num_classes: Number of outputs for classification
        residual: Turn on or off residual connections
        """
        super(DendResNet, self).__init__()
        self.residual = residual # Store if we want residual connections
        self.inplanes = 64 # Starting number of planes to map to from input channels
        self.resolution = resolution
        self.dt = dt
        
        ### INITIAL SET OF CONVOLUTIONS ###
        self.conv1 = DendriticConv2d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3, resolution=resolution, dt=dt)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        ### DEFINE LAYERS ###
        self.layer1 = self._make_layers(layer_counts[0], planes=64, stride=1)
        self.layer2 = self._make_layers(layer_counts[1], planes=128, stride=2)
        self.layer3 = self._make_layers(layer_counts[2], planes=256, stride=2)
        self.layer4 = self._make_layers(layer_counts[3], planes=512, stride=2)
        
        ### AVERAGE POOLING AND MAP TO CLASSIFIER ###
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = DendriticLinear(512*4, num_classes, resolution=resolution, dt=dt)
    
    def _make_layers(self, num_residual_blocks, planes, stride):
        downsample = None # Initialize downsampling as None
        layers = nn.ModuleList() # Create a Module list to store all our convolutions
        
        # If we have a stride of 2, or the number of planes dont match. This condition will ALWAYS BE MET only 
        #on the first block of every set of blocks
        
        if stride != 1 or self.inplanes != planes*4: 
            ### Map to the number of wanted planes with a stride of 2 to map identity to X
            downsample = nn.Sequential(DendriticConv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, resolution=self.resolution, dt=self.dt),
                                       nn.BatchNorm2d(planes*4))

        ### Append this First Block with the Downsample Layer ###
        layers.append(ResidualBlock(in_planes=self.inplanes,
                                    planes=planes, 
                                    downsample=downsample,
                                    middle_conv_stride=stride,
                                    residual=self.residual,
                                    resolution=self.resolution,
                                    dt=self.dt))
        
        ### Set our InPlanes to be expanded by 4 ###
        self.inplanes = planes * 4
        
        ### The remaining layers shouldnt have any issues so we can just append all of the blocks on ###
        for _ in range(num_residual_blocks - 1):
            layers.append(
                ResidualBlock(
                    in_planes=self.inplanes, 
                    planes = planes,
                    residual=self.residual,
                    resolution=self.resolution,
                    dt=self.dt
                )
            )
        
        return nn.Sequential(*layers)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    model = DendResNet(layer_counts=[2,2,2,2], num_channels=3, num_classes=2, resolution=10, dt=0.01)
    print(model)
    

