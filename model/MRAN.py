from torch import nn
import torch
from model.layers import *
import random
import numpy as np
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2023)

class ECA(nn.Module):
    def __init__(self, kernel_size=3):
        super(ECA, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=self.padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1)
        y = y.expand_as(x)
        return x * y

class MRANModule(nn.Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        super(MRANModule, self).__init__()
        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        self.bottleneck = Conv1d_new_padding(ni, nf, 1, bias=False) if bottleneck else noop
        self.convs = nn.ModuleList([SeparableConv1d(nf if bottleneck else ni, nf, k, bias=False) for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), Conv1d_new_padding(ni, nf, 1, bias=False)])
        self.concat = Concat()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.concat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)])
        return x

    
class MRANBlock(nn.Module):
    def __init__(self, ni, nf, residual=True, **kwargs):
        super(MRANBlock, self).__init__()
        self.residual = residual
        self.xception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for i in range(4):
            if self.residual and (i-1) % 2 == 0: self.shortcut.append(nn.BatchNorm1d(n_in) if n_in == n_out else ConvBlock(n_in, n_out * 4 * 2, 1, act=None))
            n_out = nf * 2 ** i
            n_in = ni if i == 0 else n_out * 2
            self.xception.append(MRANModule(n_in, n_out, **kwargs))
        self.add = Add()
        self.act = nn.ReLU()
        
    def forward(self, x):
        res = x
        for i in range(4):
            x = self.xception[i](x)
            if self.residual and (i + 1) % 2 == 0: res = x = self.act(self.add(x, self.shortcut[i//2](res)))
        return x
    
    
class MRAN(nn.Module):
    def __init__(self, c_in=2, c_out=10, nf=16, nb_filters=None, adaptive_size=50, **kwargs):
        super(MRAN, self).__init__()
        nf = ifnone(nf, nb_filters)
        self.block = MRANBlock(c_in, nf, **kwargs)
        self.ECA = ECA()
        self.head_nf = nf * 32
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(adaptive_size),
                                  ConvBlock(self.head_nf, self.head_nf//2, 1),
                                  ConvBlock(self.head_nf//2, self.head_nf//4, 1),
                                  ConvBlock(self.head_nf//4, c_out, 1),
                                  GAP1d(1))

    def forward(self, x):
        embedding_output = self.block(x)
        embedding_output = self.ECA(embedding_output)
        cls_output = self.head(embedding_output)
        return embedding_output, cls_output