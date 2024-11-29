import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    
    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))

class DepthWiseBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        self.conv1 = ConvBlock(in_c, groups, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = ConvBlock(groups, groups, kernel=kernel, stride=stride, padding=padding, groups=groups)
        self.conv3 = ConvBlock(groups, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))

class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        
        self.conv_23 = DepthWiseBlock(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = DepthWiseBlock(128, 128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
        
        self.conv_34 = DepthWiseBlock(128, 256, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = DepthWiseBlock(256, 256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
        
        self.conv_45 = DepthWiseBlock(256, 512, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = DepthWiseBlock(512, 512, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
        
        self.conv6_sep = ConvBlock(512, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv7_dw = ConvBlock(512, 512, kernel=(7, 7), stride=(1, 1), padding=(0, 0), groups=512)
        self.conv7_sep = ConvBlock(512, embedding_size, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        
        self.linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        
        out = self.conv_23(out)
        out = self.conv_3(out)
        
        out = self.conv_34(out)
        out = self.conv_4(out)
        
        out = self.conv_45(out)
        out = self.conv_5(out)
        
        out = self.conv6_sep(out)
        out = self.conv7_dw(out)
        out = self.conv7_sep(out)
        
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        out = self.bn(out)
        
        return F.normalize(out, p=2, dim=1)

def MainModel(nOut=512, **kwargs):
    """
    Creates a lightweight MobileFaceNet model
    Args:
        nOut: Size of the face embedding (default: 512)
    Returns:
        MobileFaceNet model
    """
    model = MobileFaceNet(embedding_size=nOut)
    return model

def inspect_model():
    model = MobileFaceNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {params:,} parameters ({params/total_params*100:.2f}%)")

if __name__ == "__main__":
    inspect_model()
