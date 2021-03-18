import argparse
import os
import shutil
import time
import math

import torch
from torch.autograd import gradcheck
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

# def l2_normalize(tensor, eps=1e-12):
#     norm = float(torch.sqrt(torch.sum(tensor * tensor)))
#     norm = max(norm, eps)
#     ans = tensor / norm
#     return ans

# def conv_power_iteration(conv_filter, u_list=None, v_list=None, num_iters=50):
#     start_time = time.time()
#     out_ch, in_ch, h, w = conv_filter.shape
#     if u_list is None:
#         u1 = torch.randn((1, in_ch, 1, w), device='cuda', requires_grad=False)
#         u1.data = l2_normalize(u1.data)
        
#         u2 = torch.randn((1, in_ch, h, 1), device='cuda', requires_grad=False)
#         u2.data = l2_normalize(u2.data)

#         u3 = torch.randn((1, in_ch, h, w), device='cuda', requires_grad=False)
#         u3.data = l2_normalize(u3.data)

#         u4 = torch.randn((out_ch, 1, h, w), device='cuda', requires_grad=False)
#         u4.data = l2_normalize(u4.data)
        
#     if v_list is None:
#         v1 = torch.randn((out_ch, 1, h, 1), device='cuda', requires_grad=False)
#         v1.data = l2_normalize(v1.data)
        
#         v2 = torch.randn((out_ch, 1, 1, w), device='cuda', requires_grad=False)
#         v2.data = l2_normalize(v2.data)

#         v3 = torch.randn((out_ch, 1, 1, 1), device='cuda', requires_grad=False)
#         v3.data = l2_normalize(v3.data)

#         v4 = torch.randn((1, in_ch, 1, 1), device='cuda', requires_grad=False)
#         v4.data = l2_normalize(v4.data)

#     for i in range(num_iters):
#         v1.data = l2_normalize((conv_filter*u1).sum((1, 3), keepdim=True).data)
#         u1.data = l2_normalize((conv_filter*v1).sum((0, 2), keepdim=True).data)
        
#         v2.data = l2_normalize((conv_filter*u2).sum((1, 2), keepdim=True).data)
#         u2.data = l2_normalize((conv_filter*v2).sum((0, 3), keepdim=True).data)
        
#         v3.data = l2_normalize((conv_filter*u3).sum((1, 2, 3), keepdim=True).data)
#         u3.data = l2_normalize((conv_filter*v3).sum(0, keepdim=True).data)
        
#         v4.data = l2_normalize((conv_filter*u4).sum((0, 2, 3), keepdim=True).data)
#         u4.data = l2_normalize((conv_filter*v4).sum(1, keepdim=True).data)
#     return u1, v1, u2, v2, u3, v3, u4, v4
    
# class ConvFilterNorm(nn.Module):
#     def __init__(self, conv_filter, name='weight', init_iters=50, num_iters=1):
#         super(ConvFilterNorm, self).__init__()
#         self.name = name
#         self.num_iters = num_iters
        
#         with torch.no_grad():
#             u1, u2, u3, u4, v1, v2, v3, v4 = conv_power_iteration(conv_filter, num_iters=init_iters)
            
#             self.u1 = u1
#             self.v1 = v1            
#             self.u2 = u2
#             self.v2 = v2
#             self.u3 = u3
#             self.v3 = v3
#             self.u4 = u4
#             self.v4 = v4

#     def forward(self, conv_filter):
#         _, _, h, w = conv_filter.shape
        
#         with torch.no_grad():
#             self.v1.data = l2_normalize((conv_filter*self.u1).sum((1, 3), keepdim=True).data)
#             self.u1.data = l2_normalize((conv_filter*self.v1).sum((0, 2), keepdim=True).data)

#             self.v2.data = l2_normalize((conv_filter*self.u2).sum((1, 2), keepdim=True).data)
#             self.u2.data = l2_normalize((conv_filter*self.v2).sum((0, 3), keepdim=True).data)

#             self.v3.data = l2_normalize((conv_filter*self.u3).sum((1, 2, 3), keepdim=True).data)
#             self.u3.data = l2_normalize((conv_filter*self.v3).sum(0, keepdim=True).data)

#             self.v4.data = l2_normalize((conv_filter*self.u4).sum((0, 2, 3), keepdim=True).data)
#             self.u4.data = l2_normalize((conv_filter*self.v4).sum(1, keepdim=True).data)
        
#         sigma1 = torch.sum((conv_filter*self.u1)*self.v1)
#         sigma2 = torch.sum((conv_filter*self.u2)*self.v2)
#         sigma3 = torch.sum((conv_filter*self.u3)*self.v3)
#         sigma4 = torch.sum((conv_filter*self.u4)*self.v4)
        
#         sigma = math.sqrt(h*w)*torch.min(torch.min(torch.min(sigma1, sigma2), sigma3), sigma4)
#         return sigma

def power_iteration(W, u=None, v=None, num_iters=1):
    if u is None:
        u = torch.randn(W.shape[1], device='cuda', requires_grad=False)
        u.data = F.normalize(u.data, dim=0)
        
        v = torch.randn(W.shape[0], device='cuda', requires_grad=False)
        v.data = F.normalize(v.data, dim=0)
    for i in range(num_iters):
        v.data = F.normalize(torch.mv(W.data, u.data), dim=0)
        u.data = F.normalize(torch.mv(torch.t(W.data), v.data), dim=0)
    return u, v
    
class ConvFilterNorm(nn.Module):
    def __init__(self, conv_filter, name='weight', init_iters=50, num_iters=1):
        super(ConvFilterNorm, self).__init__()
        self.name = name
        self.num_iters = num_iters
        self.init_filter = conv_filter.clone().detach()
        self.conv_filter = conv_filter
        
        with torch.no_grad():
            matrix1, matrix2, matrix3, matrix4 = self._conv_matrices(conv_filter)
            
            u1, v1 = power_iteration(matrix1, num_iters=init_iters)
            self.u1 = u1
            self.v1 = v1
            
            u2, v2 = power_iteration(matrix2, num_iters=init_iters)
            self.u2 = u2
            self.v2 = v2
            
            u3, v3 = power_iteration(matrix3, num_iters=init_iters)
            self.u3 = u3
            self.v3 = v3

            u4, v4 = power_iteration(matrix4, num_iters=init_iters)
            self.u4 = u4
            self.v4 = v4
            
    def _conv_matrices(self, conv_filter):
        out_ch, in_ch, h, w = conv_filter.shape
        
        transpose1 = torch.transpose(conv_filter, 1, 2)
        matrix1 = transpose1.reshape(out_ch*h, in_ch*w)
        
        transpose2 = torch.transpose(conv_filter, 1, 3)
        matrix2 = transpose2.reshape(out_ch*w, in_ch*h)

        matrix3 = conv_filter.view(out_ch, in_ch*h*w)

        transpose4 = torch.transpose(conv_filter, 0, 1)
        matrix4 = transpose4.reshape(in_ch, out_ch*h*w)

        return matrix1, matrix2, matrix3, matrix4

    def forward(self):
        conv_filter = self.conv_filter
        _, _, h, w = conv_filter.shape
        
        matrix1, matrix2, matrix3, matrix4 = self._conv_matrices(conv_filter)
        
        with torch.no_grad():
            self.v1.data = F.normalize(torch.mv(matrix1.data, self.u1.data), dim=0)
            self.u1.data = F.normalize(torch.mv(torch.t(matrix1.data), self.v1.data), dim=0)

            self.v2.data = F.normalize(torch.mv(matrix2.data, self.u2.data), dim=0)
            self.u2.data = F.normalize(torch.mv(torch.t(matrix2.data), self.v2.data), dim=0)

            self.v3.data = F.normalize(torch.mv(matrix3.data, self.u3.data), dim=0)
            self.u3.data = F.normalize(torch.mv(torch.t(matrix3.data), self.v3.data), dim=0)
            
            self.v4.data = F.normalize(torch.mv(matrix4.data, self.u4.data), dim=0)
            self.u4.data = F.normalize(torch.mv(torch.t(matrix4.data), self.v4.data), dim=0)
        
        sigma1 = torch.mv(self.v1.unsqueeze(0), torch.mv(matrix1, self.u1))
        sigma2 = torch.mv(self.v2.unsqueeze(0), torch.mv(matrix2, self.u2))
        sigma3 = torch.mv(self.v3.unsqueeze(0), torch.mv(matrix3, self.u3)) 
        sigma4 = torch.mv(self.v4.unsqueeze(0), torch.mv(matrix4, self.u4))
        
        sigma = math.sqrt(h*w)*torch.min(torch.min(torch.min(sigma1, sigma2), sigma3), sigma4)
        return sigma