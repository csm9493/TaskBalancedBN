import torch
from torch import nn
from copy import deepcopy
from torchvision.models import resnet
from functools import partial

import torch.nn.functional as F
from copy import deepcopy
import pickle

import math

from torch.nn.modules.batchnorm import _BatchNorm

class TaskBalancedBN(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, batch_ratio = 3, B_c = 48, B_p = 16):
        super(TaskBalancedBN, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
        self.batch_ratio = batch_ratio
        self.num_splits = 1
        self.B_c = B_c
        self.B_p = B_p
        
    # Function to find common divisors of two numbers
    def common_divisors(self, B_c, B_p):
        divisors = []
        gcd_value = math.gcd(B_c, B_p)  # Get the greatest common divisor
        for i in range(1, gcd_value + 1):
            if B_c % i == 0 and B_p % i == 0:
                divisors.append(i)
        return divisors

    # Function to calculate feasible r*
    def calculate_feasible_r(self, B_c, B_p, r):
        # Get the common divisors
        common_divs = self.common_divisors(B_c, B_p)

        # Check if r is in the set of common divisors
        if r in common_divs:
            return r
        else:
            # If r is not in common divisors, find the maximum feasible r* from the common divisors less than r
            feasible_r = max([d for d in common_divs if d < r], default=None)
            return feasible_r
        
    def set_number_of_task(self, T):
        
        self.T = T
        
        if self.T == 0:
            r = 1
        else:
            r = self.batch_ratio * (self.T)
        
        
        feasible_r = self.calculate_feasible_r(self.B_c, self.B_p, r)
        
        print ('Task Number : ', self.T, ' Feasible r : ', feasible_r)
        
        self.num_splits = feasible_r

    def forward(self, input):
        
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
                
        # if T == 0 : general BN
        if self.T == 0:
            
            # calculate running estimates
            if self.training:
                
                running_mean_split = self.running_mean.repeat(self.num_splits)
                running_var_split = self.running_var.repeat(self.num_splits)

                N,C,H,W = input.shape

                input = input.view(-1, C * self.num_splits, H, W)

                mean = input.mean([0, 2, 3])
                var = input.var([0, 2, 3], unbiased=False)

                n = input.numel() / input.size(1)

                with torch.no_grad():

                    running_mean_split = exponential_average_factor * mean\
                        + (1 - exponential_average_factor) * running_mean_split

                    running_var_split = exponential_average_factor * var*n / (n - 1)\
                        + (1 - exponential_average_factor) * running_var_split

                input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
                if self.affine:
                    input = input * self.weight.repeat(self.num_splits)[None, :, None, None] + self.bias.repeat(self.num_splits)[None, :, None, None]

                input = input.view(N, C, H, W)

                self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
                self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))

            else:

                mean = self.running_mean
                var = self.running_var

                input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
                if self.affine:
                    input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
                
            return input
        
        else:

            if self.training:
                
                N,C,H,W = input.shape
                curr_dataset = (N // (self.batch_ratio + 1)) * self.batch_ratio

                running_mean_split = self.running_mean
                running_var_split = self.running_var

                curr_batch = input[:curr_dataset,:,:,:]
                mem_batch = input[curr_dataset:,:,:,:]

                curr_batch = curr_batch.view(-1, C * self.num_splits, H, W)
                mem_batch_repeat = mem_batch.repeat(1, self.num_splits, 1, 1)

                concat_batch = torch.cat([curr_batch, mem_batch_repeat], dim = 0)

                repeat_mean = concat_batch.mean([0, 2, 3])
                repeat_var = concat_batch.var([0, 2, 3], unbiased=False)

#                 input = input.view(-1, C * self.num_splits, H, W)

                concat_batch = (concat_batch - repeat_mean[None, :, None, None]) / (torch.sqrt(repeat_var[None, :, None, None] + self.eps))
                if self.affine:
                    concat_batch = concat_batch * self.weight.repeat(self.num_splits)[None, :, None, None] + self.bias.repeat(self.num_splits)[None, :, None, None]

                reshaped_curr_batch = concat_batch[:curr_dataset//self.num_splits].view(-1, C, H, W)
                reshaped_mem_batch = torch.mean(concat_batch[curr_dataset//self.num_splits:].view(-1, self.num_splits, C, H, W),  dim = 1)

                input = torch.cat([reshaped_curr_batch, reshaped_mem_batch], dim = 0)

                repeat_mean = input.mean([0, 2, 3])
                repeat_var = input.var([0, 2, 3], unbiased=False)

                n = input.numel() / input.size(1)

                with torch.no_grad():

                    running_mean_split = exponential_average_factor * repeat_mean\
                        + (1 - exponential_average_factor) * running_mean_split

                    running_var_split = exponential_average_factor * repeat_var* n / (n - 1)\
                        + (1 - exponential_average_factor) * running_var_split

                self.running_mean.data.copy_(running_mean_split)
                self.running_var.data.copy_(running_var_split)
                
            else:
                
                mean = self.running_mean
                var = self.running_var

                input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
                if self.affine:
                    input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

            return input    