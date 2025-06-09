import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Get_Correlation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduction_channel = channels//16 # Reduce the number of channels to 1/16 to reduce computation cost
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)

        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,1,1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,2,2), dilation=(1,2,2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,3,3), dilation=(1,3,3), groups=reduction_channel)
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

    def forward(self, x):

        x2 = self.down_conv2(x)

        # Calculate the correlation with h,w is spatial position in current frame and s,d is spatial position in the next frame
        # [:,:,1:] is a three dimensional tensor with the first frame removed, [:,:,-1:] is the last frame

        affinities = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,1:], x2[:,:,-1:]], 2))  # Calculate with the next frame 
        affinities2 = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,:1], x2[:,:,:-1]], 2))  # Calculate with the previous frame

        # Calculate the correlation features
        features = torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,1:], x2[:,:,-1:]], 2), F.sigmoid(affinities)-0.5 )* self.weights2[0] + \
            torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,:1], x2[:,:,:-1]], 2), F.sigmoid(affinities2)-0.5 ) * self.weights2[1]

        x = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(x)*self.weights[0] + self.spatial_aggregation2(x)*self.weights[1] \
                    + self.spatial_aggregation3(x)*self.weights[2]
        aggregated_x = self.conv_back(aggregated_x)

        return features * (F.sigmoid(aggregated_x)-0.5)


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__() # Initialize ResNet with BasicBlock

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        """
        Why use a 1x7x7 kernel size in the first convolutional layer?
        
        1) Temporal-Spatial Processing: 
           - The (1,7,7) kernel allows the model to process video data as a sequence of frames
           - The first dimension (1) spans the temporal axis but only processes one frame at a time
           - The second and third dimensions (7,7) capture spatial features within each frame
        
        2) Model Architecture Benefits:
           - Enables leveraging pretrained 2D ResNet weights while handling temporal data
           - Preserves temporal ordering by not mixing information across frames
           - Creates a pseudo-3D architecture while maintaining computational efficiency
        
        3) Dimension Handling:
           - Input: N (batch) x C (channels) x T (time steps) x H (height) x W (width)
           - This conv layer preserves the temporal dimension (T) while reducing spatial dimensions
           - Subsequent temporal correlations (via Get_Correlation) will model frame relationships
        """

        self.bn1 = nn.BatchNorm3d(64) # Batch normalization for the first conv layer
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) # Max pooling to reduce spatial dimensions with kernel size (1,3,3) meaning no temporal pooling, only spatial pooling
        self.layer1 = self._make_layer(block, 64, layers[0]) # First layer with 64 output channels

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) 
        self.corr1 = Get_Correlation(self.inplanes)
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.corr2 = Get_Correlation(self.inplanes)
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True) # Learnable parameters to control the contribution of correlation features

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.corr3 = Get_Correlation(self.inplanes)

        self.avgpool = nn.AvgPool2d(7, stride=1) # Average pooling to reduce the spatial dimensions to 1x1 before the final fully connected layer
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes) # Final fully connected layer to output class scores


        for m in self.modules(): # Initialize weights for the model
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1): # Create a layer of blocks with specified planes and stride
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): # Forward pass through the ResNet model
        N, C, T, H, W = x.size() # N is batch size, C is number of channels, T is number of frames, H is height, W is width
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = x + self.corr1(x) * self.alpha[0]
        x = self.layer3(x)
        x = x + self.corr2(x) * self.alpha[1]
        x = self.layer4(x)
        x = x + self.corr3(x) * self.alpha[2]
        x = x.transpose(1,2).contiguous()
        x = x.view((-1,)+x.size()[2:]) #bt,c,h,w: # Reshape the tensor from [N, T, C, H, W] to (batch_size * T, channels, height, width)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) #bt,c: (batch_size * T, channels dimension)
        x = self.fc(x) #bt,c # Final fully connected layer to output class scores

        return x

def resnet18(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model

def test():
    net = resnet18()
    y = net(torch.randn(1,3,224,224))
    print(y.size())

#test()