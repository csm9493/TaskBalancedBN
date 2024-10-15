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


# class TaskBalancedBN(nn.BatchNorm2d):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1,
#                  affine=True, track_running_stats=True, batch_ratio = 3, num_splits_arr = [1,2,4,8,8,8,16,16,16,16,16], ablation = 'TBBN2'):
#         super(TaskBalancedBN, self).__init__(
#             num_features, eps, momentum, affine, track_running_stats)
        
#         self.batch_ratio = batch_ratio
#         self.num_splits_arr = num_splits_arr
#         self.ablation = ablation
        
#     def set_number_of_task(self, T):
        
#         self.T = T
#         print ('set_number_of_task : ', T)

#     def forward(self, input):
        
#         self._check_input_dim(input)

#         exponential_average_factor = 0.0

#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum
                
#         # if T == 0 : general BN
#         if self.T == 0:
            
#             # calculate running estimates
#             if self.training:
                
#                 self.num_splits = self.num_splits_arr[self.T]

#                 running_mean_split = self.running_mean.repeat(self.num_splits)
#                 running_var_split = self.running_var.repeat(self.num_splits)

#                 N,C,H,W = input.shape

#                 input = input.view(-1, C * self.num_splits, H, W)

#                 mean = input.mean([0, 2, 3])
#                 var = input.var([0, 2, 3], unbiased=False)

#                 n = input.numel() / input.size(1)

#                 with torch.no_grad():

#                     running_mean_split = exponential_average_factor * mean\
#                         + (1 - exponential_average_factor) * running_mean_split

#                     running_var_split = exponential_average_factor * var*n / (n - 1)\
#                         + (1 - exponential_average_factor) * running_var_split

#                 input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
#                 if self.affine:
#                     input = input * self.weight.repeat(self.num_splits)[None, :, None, None] + self.bias.repeat(self.num_splits)[None, :, None, None]

#                 input = input.view(N, C, H, W)

#                 self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
#                 self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))

#             else:

#                 mean = self.running_mean
#                 var = self.running_var

#                 input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
#                 if self.affine:
#                     input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
                
#             return input
        
#         # else : SplitRepeatBN with Adaptive num_splits
#         else:

#             if self.training:
                
#                 if self.ablation == 'TBBN2':

#                     N,C,H,W = input.shape
#                     curr_dataset = (N // (self.batch_ratio + 1)) * self.batch_ratio

#                     self.num_splits = self.num_splits_arr[self.T]

#                     running_mean_split = self.running_mean.repeat(self.num_splits)
#                     running_var_split = self.running_var.repeat(self.num_splits)

#                     curr_batch = input[:curr_dataset,:,:,:]
#                     mem_batch = input[curr_dataset:,:,:,:]

#                     curr_batch = curr_batch.view(-1, C * self.num_splits, H, W)
#                     mem_batch_repeat = mem_batch.repeat(1, self.num_splits, 1, 1)

#                     concat_batch = torch.cat([curr_batch, mem_batch_repeat], dim = 0)

#                     repeat_mean = concat_batch.mean([0, 2, 3])
#                     repeat_var = concat_batch.var([0, 2, 3], unbiased=False)

#                     n = concat_batch.numel() / concat_batch.size(1)

#                     with torch.no_grad():

#                         running_mean_split = exponential_average_factor * repeat_mean\
#                             + (1 - exponential_average_factor) * running_mean_split

#                         running_var_split = exponential_average_factor * repeat_var* n / (n - 1)\
#                             + (1 - exponential_average_factor) * running_var_split

#                     self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
#                     self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))

#     #                 input = input.view(-1, C * self.num_splits, H, W)

#                     concat_batch = (concat_batch - repeat_mean[None, :, None, None]) / (torch.sqrt(repeat_var[None, :, None, None] + self.eps))
#                     if self.affine:
#                         concat_batch = concat_batch * self.weight.repeat(self.num_splits)[None, :, None, None] + self.bias.repeat(self.num_splits)[None, :, None, None]

#                     reshaped_curr_batch = concat_batch[:curr_dataset//self.num_splits].view(-1, C, H, W)
#                     reshaped_mem_batch = torch.mean(concat_batch[curr_dataset//self.num_splits:].view(-1, self.num_splits, C, H, W),  dim = 1)

#                     input = torch.cat([reshaped_curr_batch, reshaped_mem_batch], dim = 0)
                    
#                 elif self.ablation == 'case1':

#                     N,C,H,W = input.shape
#                     curr_dataset = (N // (self.batch_ratio + 1)) * self.batch_ratio

#                     self.num_splits = self.num_splits_arr[self.T]

#                     running_mean_split = self.running_mean.repeat(self.num_splits)
#                     running_var_split = self.running_var.repeat(self.num_splits)

#                     curr_batch = input[:curr_dataset,:,:,:]
#                     mem_batch = input[curr_dataset:,:,:,:]

#                     curr_batch = curr_batch.view(-1, C * self.num_splits, H, W)
#                     mem_batch_repeat = mem_batch.repeat(1, self.num_splits, 1, 1)

#                     concat_batch = torch.cat([curr_batch, mem_batch_repeat], dim = 0)

#                     repeat_mean = concat_batch.mean([0, 2, 3])
#                     repeat_var = concat_batch.var([0, 2, 3], unbiased=False)

#                     n = concat_batch.numel() / concat_batch.size(1)

#                     with torch.no_grad():

#                         running_mean_split = exponential_average_factor * repeat_mean\
#                             + (1 - exponential_average_factor) * running_mean_split

#                         running_var_split = exponential_average_factor * repeat_var* n / (n - 1)\
#                             + (1 - exponential_average_factor) * running_var_split

#                     self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
#                     self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
                    
#                     repeat_mean = input.mean([0, 2, 3]).repeat(self.num_splits)
#                     repeat_var = input.var([0, 2, 3], unbiased=False).repeat(self.num_splits)

#     #                 input = input.view(-1, C * self.num_splits, H, W)

#                     concat_batch = (concat_batch - repeat_mean[None, :, None, None]) / (torch.sqrt(repeat_var[None, :, None, None] + self.eps))
#                     if self.affine:
#                         concat_batch = concat_batch * self.weight.repeat(self.num_splits)[None, :, None, None] + self.bias.repeat(self.num_splits)[None, :, None, None]

#                     reshaped_curr_batch = concat_batch[:curr_dataset//self.num_splits].view(-1, C, H, W)
#                     reshaped_mem_batch = torch.mean(concat_batch[curr_dataset//self.num_splits:].view(-1, self.num_splits, C, H, W),  dim = 1)

#                     input = torch.cat([reshaped_curr_batch, reshaped_mem_batch], dim = 0)
                    
#                 elif self.ablation == 'case2':

#                     N,C,H,W = input.shape
#                     curr_dataset = (N // (self.batch_ratio + 1)) * self.batch_ratio

#                     self.num_splits = self.num_splits_arr[self.T]

#                     running_mean_split = self.running_mean
#                     running_var_split = self.running_var

#                     curr_batch = input[:curr_dataset,:,:,:]
#                     mem_batch = input[curr_dataset:,:,:,:]

#                     curr_batch = curr_batch.view(-1, C * self.num_splits, H, W)
#                     mem_batch_repeat = mem_batch.repeat(1, self.num_splits, 1, 1)

#                     concat_batch = torch.cat([curr_batch, mem_batch_repeat], dim = 0)

#                     repeat_mean = concat_batch.mean([0, 2, 3])
#                     repeat_var = concat_batch.var([0, 2, 3], unbiased=False)

#     #                 input = input.view(-1, C * self.num_splits, H, W)

#                     concat_batch = (concat_batch - repeat_mean[None, :, None, None]) / (torch.sqrt(repeat_var[None, :, None, None] + self.eps))
#                     if self.affine:
#                         concat_batch = concat_batch * self.weight.repeat(self.num_splits)[None, :, None, None] + self.bias.repeat(self.num_splits)[None, :, None, None]

#                     reshaped_curr_batch = concat_batch[:curr_dataset//self.num_splits].view(-1, C, H, W)
#                     reshaped_mem_batch = torch.mean(concat_batch[curr_dataset//self.num_splits:].view(-1, self.num_splits, C, H, W),  dim = 1)

#                     input = torch.cat([reshaped_curr_batch, reshaped_mem_batch], dim = 0)
                    
#                     repeat_mean = input.mean([0, 2, 3])
#                     repeat_var = input.var([0, 2, 3], unbiased=False)
                    
#                     n = input.numel() / input.size(1)

#                     with torch.no_grad():

#                         running_mean_split = exponential_average_factor * repeat_mean\
#                             + (1 - exponential_average_factor) * running_mean_split

#                         running_var_split = exponential_average_factor * repeat_var* n / (n - 1)\
#                             + (1 - exponential_average_factor) * running_var_split

#                     self.running_mean.data.copy_(running_mean_split)
#                     self.running_var.data.copy_(running_var_split)
                    
#                 elif self.ablation == 'case3':

#                     N,C,H,W = input.shape
#                     curr_dataset = (N // (self.batch_ratio + 1)) * self.batch_ratio

#                     self.num_splits = self.num_splits_arr[self.T]

#                     running_mean_split = self.running_mean.repeat(self.num_splits)
#                     running_var_split = self.running_var.repeat(self.num_splits)

#                     curr_batch = input[:curr_dataset,:,:,:]
#                     mem_batch = input[curr_dataset:,:,:,:]

#                     curr_batch = curr_batch.view(-1, C * self.num_splits, H, W)
#                     mem_batch_repeat = mem_batch.repeat(1, self.num_splits, 1, 1)

#                     concat_batch = torch.cat([curr_batch, mem_batch_repeat], dim = 0)

#                     repeat_mean = concat_batch.mean([0, 2, 3])
#                     repeat_var = concat_batch.var([0, 2, 3], unbiased=False)

#                     n = concat_batch.numel() / concat_batch.size(1)

#                     with torch.no_grad():

#                         running_mean_split = exponential_average_factor * repeat_mean\
#                             + (1 - exponential_average_factor) * running_mean_split

#                         running_var_split = exponential_average_factor * repeat_var* n / (n - 1)\
#                             + (1 - exponential_average_factor) * running_var_split

#                     self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
#                     self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))

#     #                 input = input.view(-1, C * self.num_splits, H, W)
    
#                     repeat_mean = repeat_mean.view(self.num_splits, C).mean(dim=0)
#                     repeat_var = repeat_var.view(self.num_splits, C).mean(dim=0)
    

#                     input = (input - repeat_mean[None, :, None, None]) / (torch.sqrt(repeat_var[None, :, None, None] + self.eps))
#                     if self.affine:
#                         input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

# #                     reshaped_curr_batch = concat_batch[:curr_dataset//self.num_splits].view(-1, C, H, W)
# #                     reshaped_mem_batch = torch.mean(concat_batch[curr_dataset//self.num_splits:].view(-1, self.num_splits, C, H, W),  dim = 1)

# #                     input = torch.cat([reshaped_curr_batch, reshaped_mem_batch], dim = 0)
                
#                 if self.ablation == 'case4':

#                     N,C,H,W = input.shape
#                     curr_dataset = (N // (self.batch_ratio + 1)) * self.batch_ratio

#                     self.num_splits = self.num_splits_arr[self.T]

#                     running_mean_split = self.running_mean.repeat(self.num_splits)
#                     running_var_split = self.running_var.repeat(self.num_splits)

#                     curr_batch = input[:curr_dataset,:,:,:]
#                     mem_batch = input[curr_dataset:,:,:,:]

#                     curr_batch = curr_batch.view(-1, C * self.num_splits, H, W)
#                     mem_batch_repeat = mem_batch.repeat(1, self.num_splits, 1, 1)

#                     concat_batch = torch.cat([curr_batch, mem_batch_repeat], dim = 0)

#                     repeat_mean = input.mean([0, 2, 3]).repeat(self.num_splits)
#                     repeat_var = input.var([0, 2, 3], unbiased=False).repeat(self.num_splits)

#                     n = concat_batch.numel() / concat_batch.size(1)

#                     with torch.no_grad():

#                         running_mean_split = exponential_average_factor * repeat_mean\
#                             + (1 - exponential_average_factor) * running_mean_split

#                         running_var_split = exponential_average_factor * repeat_var* n / (n - 1)\
#                             + (1 - exponential_average_factor) * running_var_split

#                     self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
#                     self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))

#     #                 input = input.view(-1, C * self.num_splits, H, W)

#                     concat_batch = (concat_batch - repeat_mean[None, :, None, None]) / (torch.sqrt(repeat_var[None, :, None, None] + self.eps))
#                     if self.affine:
#                         concat_batch = concat_batch * self.weight.repeat(self.num_splits)[None, :, None, None] + self.bias.repeat(self.num_splits)[None, :, None, None]

#                     reshaped_curr_batch = concat_batch[:curr_dataset//self.num_splits].view(-1, C, H, W)
#                     reshaped_mem_batch = torch.mean(concat_batch[curr_dataset//self.num_splits:].view(-1, self.num_splits, C, H, W),  dim = 1)

#                     input = torch.cat([reshaped_curr_batch, reshaped_mem_batch], dim = 0)
                
#                 elif self.ablation == 'case5':

#                     N,C,H,W = input.shape
#                     curr_dataset = (N // (self.batch_ratio + 1)) * self.batch_ratio

#                     self.num_splits = self.num_splits_arr[self.T]

#                     running_mean_split = self.running_mean
#                     running_var_split = self.running_var

#                     curr_batch = input[:curr_dataset,:,:,:]
#                     mem_batch = input[curr_dataset:,:,:,:]

#                     curr_batch = curr_batch
#                     mem_batch_repeat = mem_batch.repeat(self.num_splits, 1, 1, 1)

#                     concat_batch = torch.cat([curr_batch, mem_batch_repeat], dim = 0)

#                     repeat_mean = concat_batch.mean([0, 2, 3])
#                     repeat_var = concat_batch.var([0, 2, 3], unbiased=False)

#                     n = concat_batch.numel() / concat_batch.size(1)

#                     with torch.no_grad():

#                         running_mean_split = exponential_average_factor * repeat_mean\
#                             + (1 - exponential_average_factor) * running_mean_split

#                         running_var_split = exponential_average_factor * repeat_var* n / (n - 1)\
#                             + (1 - exponential_average_factor) * running_var_split

#                     self.running_mean.data.copy_(running_mean_split)
#                     self.running_var.data.copy_(running_var_split)

#     #                 input = input.view(-1, C * self.num_splits, H, W)

#                     concat_batch = (concat_batch - repeat_mean[None, :, None, None]) / (torch.sqrt(repeat_var[None, :, None, None] + self.eps))
#                     if self.affine:
#                         concat_batch = concat_batch * self.weight[None, :, None, None] + self.bias[None, :, None, None]

#                     reshaped_curr_batch = concat_batch[:curr_dataset]
#                     reshaped_mem_batch = torch.mean(concat_batch[curr_dataset:].view(-1, self.num_splits, C, H, W),  dim = 1)

#                     input = torch.cat([reshaped_curr_batch, reshaped_mem_batch], dim = 0)
                
#                 elif self.ablation == 'case6':

#                     N,C,H,W = input.shape
#                     curr_dataset = (N // (self.batch_ratio + 1)) * self.batch_ratio

#                     self.num_splits = self.num_splits_arr[self.T]

#                     running_mean_split = self.running_mean
#                     running_var_split = self.running_var

#                     curr_batch = input[:curr_dataset,:,:,:]
#                     mem_batch = input[curr_dataset:,:,:,:]

#                     curr_batch = curr_batch
#                     mem_batch_repeat = mem_batch.repeat(self.num_splits, 1, 1, 1)

#                     concat_batch = torch.cat([curr_batch, mem_batch_repeat], dim = 0)

#                     repeat_mean = concat_batch.mean([0, 2, 3])
#                     repeat_var = concat_batch.var([0, 2, 3], unbiased=False)

#                     n = concat_batch.numel() / concat_batch.size(1)

#                     with torch.no_grad():

#                         running_mean_split = exponential_average_factor * repeat_mean\
#                             + (1 - exponential_average_factor) * running_mean_split

#                         running_var_split = exponential_average_factor * repeat_var* n / (n - 1)\
#                             + (1 - exponential_average_factor) * running_var_split

#                     self.running_mean.data.copy_(running_mean_split)
#                     self.running_var.data.copy_(running_var_split)

#                     input = (input - repeat_mean[None, :, None, None]) / (torch.sqrt(repeat_var[None, :, None, None] + self.eps))
#                     if self.affine:
#                         input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

                
#                 elif self.ablation == 'case7':

#                     N,C,H,W = input.shape
#                     curr_dataset = (N // (self.batch_ratio + 1)) * self.batch_ratio

#                     self.num_splits = self.num_splits_arr[self.T]

#                     running_mean_split = self.running_mean
#                     running_var_split = self.running_var

#                     curr_batch = input[:curr_dataset,:,:,:]
#                     mem_batch = input[curr_dataset:,:,:,:]

#                     curr_batch = curr_batch
#                     mem_batch_repeat = mem_batch.repeat(self.num_splits, 1, 1, 1)

#                     concat_batch = torch.cat([curr_batch, mem_batch_repeat], dim = 0)

#                     repeat_mean = input.mean([0, 2, 3])
#                     repeat_var = input.var([0, 2, 3], unbiased=False)

#                     n = input.numel() / input.size(1)

#                     with torch.no_grad():

#                         running_mean_split = exponential_average_factor * repeat_mean\
#                             + (1 - exponential_average_factor) * running_mean_split

#                         running_var_split = exponential_average_factor * repeat_var* n / (n - 1)\
#                             + (1 - exponential_average_factor) * running_var_split

#                     self.running_mean.data.copy_(running_mean_split)
#                     self.running_var.data.copy_(running_var_split)

#     #                 input = input.view(-1, C * self.num_splits, H, W)

#                     concat_batch = (concat_batch - repeat_mean[None, :, None, None]) / (torch.sqrt(repeat_var[None, :, None, None] + self.eps))
#                     if self.affine:
#                         concat_batch = concat_batch * self.weight[None, :, None, None] + self.bias[None, :, None, None]

#                     reshaped_curr_batch = concat_batch[:curr_dataset]
#                     reshaped_mem_batch = torch.mean(concat_batch[curr_dataset:].view(-1, self.num_splits, C, H, W),  dim = 1)

#                     input = torch.cat([reshaped_curr_batch, reshaped_mem_batch], dim = 0)
                
#             else:
                
#                 mean = self.running_mean
#                 var = self.running_var

#                 input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
#                 if self.affine:
#                     input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

#             return input    
        
class LLL_Net(nn.Module):
    def __init__(self, model, remove_existing_head=False, num_splits = 1):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()
        
        self.dim = 512
        self.K = 4096
        self.m = 0.99
        self.T = 0.1
        self.symmetric = False
        

        # create the encoders
        self.model = model
            
        last_layer = getattr(self.model, head_var)
        
        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features
            
        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()
        
        self.fc = None
        


    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]
    
    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])
        
    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        x = self.model(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        for head in self.heads:
            y.append(head(x))
        if return_features:
            return y, x
        else:
            return y

        
        
    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        print("freeze_backbone")
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        print("freeze_bn")
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def log_running_parameters(self, trn_loader, exp_name, t):
        print("logging running parameters")
        with torch.no_grad():
            # zero-intialize all mean/var stored in bn
            for m in self.model.modules():
                if isinstance(m, SplitBatchNorm):
                    m.mean = m.mean * 0
                    m.var = m.var * 0
                    print("m.mean",m.mean)
                    print("m.var",m.var)
            
            for m in self.model.modules():
                if isinstance(m, BatchNorm):
                    m.mean = m.mean * 0
                    m.var = m.var * 0
                    print("m.mean",m.mean)
                    print("m.var",m.var)


            self.eval()
            for k in range(t+1):
                for i, data in enumerate(trn_loader[k]):
                    images, targets = data
                    images = images.cuda()
                    #print("i", i)
                    #print("task:",k)
                    output = self.model(images)
            
            for m in self.model.modules():
                if isinstance(m, SplitBatchNorm):
                    m.mean = m.mean / m.num_batches_tracked
                    m.var = m.var / m.num_batches_tracked
                    m.var = m.var - m.mean ** 2
                    print("Cal. m.mean:", m.mean)
                    print("Cal. m.var:", m.var)
                    
            for m in self.model.modules():
                if isinstance(m, BatchNorm):
                    m.mean = m.mean / m.num_batches_tracked
                    m.var = m.var / m.num_batches_tracked
                    m.var = m.var - m.mean ** 2
                    print("Cal. m.mean:", m.mean)
                    print("Cal. m.var:", m.var)
                    
                    
            # zero-intialize all mean/var stored in bn
            for m in self.model.modules():
                if isinstance(m, SplitBatchNorm):
                    m.mean = m.mean * 0
                    m.var = m.var * 0
                    print("m.mean",m.mean)
                    print("m.var",m.var)
            
            for m in self.model.modules():
                if isinstance(m, BatchNorm):
                    m.mean = m.mean * 0
                    m.var = m.var * 0
                    print("m.mean",m.mean)
                    print("m.var",m.var)
                    
            """

            mean_1 = self.model.bn1.mean
            mean_1 = mean_1 / self.model.bn1.num_batches_tracked
            var_1 = self.model.bn1.var
            var_1 = var_1 / self.model.bn1.num_batches_tracked
            var_1 = var_1 - mean_1 ** 2
            """
            """

            with open("../results/" + exp_name + "/bn_parameters/"+"mean_1.txt", 'wb') as fp:
                pickle.dump(mean_1, fp)
            with open("../results/" + exp_name + "/bn_parameters/"+"var_1.txt", 'wb') as fp:
                pickle.dump(var_1, fp)
            """
        """
        self.mean_1.append(deepcopy(mean_1))
        self.var_1.append(deepcopy(var_1))
        """
        
        
        self.mean_1.append(deepcopy(self.model.bn1.mean))
        self.var_1.append(deepcopy(self.model.bn1.var))
        
        
        with open("../results/" + exp_name + "/bn_parameters/"+"mean_1.txt", 'wb') as fp:
            pickle.dump(self.mean_1, fp)
        with open("../results/" + exp_name + "/bn_parameters/"+"var_1.txt", 'wb') as fp:
            pickle.dump(self.var_1, fp)
        
    
    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        
        # for saving gamma, beta, running_mean, running_var, initialize structure
        
        self.weight_1, self.bias_1, self.running_mean_1, self.running_var_1, self.mean_1, self.var_1 = [], [], [], [], [], []
        self.weight_2, self.bias_2, self.running_mean_2, self.running_var_2, self.mean_2, self.var_2 = [], [], [], [], [], []
        self.weight_3, self.bias_3, self.running_mean_3, self.running_var_3, self.mean_3, self.var_3 = [], [], [], [], [], []
        self.weight_4, self.bias_4, self.running_mean_4, self.running_var_4, self.mean_4, self.var_4 = [], [], [], [], [], []
        self.weight_5, self.bias_5, self.running_mean_5, self.running_var_5, self.mean_5, self.var_5 = [], [], [], [], [], []
        self.weight_6, self.bias_6, self.running_mean_6, self.running_var_6, self.mean_6, self.var_6 = [], [], [], [], [], []
        self.weight_7, self.bias_7, self.running_mean_7, self.running_var_7, self.mean_7, self.var_7 = [], [], [], [], [], []
        self.weight_8, self.bias_8, self.running_mean_8, self.running_var_8, self.mean_8, self.var_8 = [], [], [], [], [], []
        self.weight_9, self.bias_9, self.running_mean_9, self.running_var_9, self.mean_9, self.var_9 = [], [], [], [], [], []
        self.weight_10, self.bias_10, self.running_mean_10, self.running_var_10, self.mean_10, self.var_10 = [], [], [], [], [], []
        self.weight_11, self.bias_11, self.running_mean_11, self.running_var_11, self.mean_11, self.var_11 = [], [], [], [], [], []
        self.weight_12, self.bias_12, self.running_mean_12, self.running_var_12, self.mean_12, self.var_12 = [], [], [], [], [], []
        self.weight_13, self.bias_13, self.running_mean_13, self.running_var_13, self.mean_13, self.var_13 = [], [], [], [], [], []
        self.weight_14, self.bias_14, self.running_mean_14, self.running_var_14, self.mean_14, self.var_14 = [], [], [], [], [], []
        self.weight_15, self.bias_15, self.running_mean_15, self.running_var_15, self.mean_15, self.var_15 = [], [], [], [], [], []
        self.weight_16, self.bias_16, self.running_mean_16, self.running_var_16, self.mean_16, self.var_16 = [], [], [], [], [], []
        self.weight_17, self.bias_17, self.running_mean_17, self.running_var_17, self.mean_17, self.var_17 = [], [], [], [], [], []
        self.weight_18, self.bias_18, self.running_mean_18, self.running_var_18, self.mean_18, self.var_18 = [], [], [], [], [], []
        self.weight_19, self.bias_19, self.running_mean_19, self.running_var_19, self.mean_19, self.var_19 = [], [], [], [], [], []
        self.weight_20, self.bias_20, self.running_mean_20, self.running_var_20, self.mean_20, self.var_20 = [], [], [], [], [], []
    
    def save_bn_parameters(self):
        print("Copying BN parameters...")
        self.weight_1.append(deepcopy(self.model.bn1.weight))
        self.bias_1.append(deepcopy(self.model.bn1.bias))
        self.running_mean_1.append(deepcopy(self.model.bn1.running_mean))
        self.running_var_1.append(deepcopy(self.model.bn1.running_var))
        
        self.weight_2.append(deepcopy(self.model.layer1[0].bn1.weight))
        self.bias_2.append(deepcopy(self.model.layer1[0].bn1.bias))
        self.running_mean_2.append(deepcopy(self.model.layer1[0].bn1.running_mean))
        self.running_var_2.append(deepcopy(self.model.layer1[0].bn1.running_var))
        
        self.weight_3.append(deepcopy(self.model.layer1[0].bn2.weight))
        self.bias_3.append(deepcopy(self.model.layer1[0].bn2.bias))
        self.running_mean_3.append(deepcopy(self.model.layer1[0].bn2.running_mean))
        self.running_var_3.append(deepcopy(self.model.layer1[0].bn2.running_var))
        
        self.weight_4.append(deepcopy(self.model.layer1[1].bn1.weight))
        self.bias_4.append(deepcopy(self.model.layer1[1].bn1.bias))
        self.running_mean_4.append(deepcopy(self.model.layer1[1].bn1.running_mean))
        self.running_var_4.append(deepcopy(self.model.layer1[1].bn1.running_var))
        
        self.weight_5.append(deepcopy(self.model.layer1[1].bn2.weight))
        self.bias_5.append(deepcopy(self.model.layer1[1].bn2.bias))
        self.running_mean_5.append(deepcopy(self.model.layer1[1].bn2.running_mean))
        self.running_var_5.append(deepcopy(self.model.layer1[1].bn2.running_var))
        
        self.weight_6.append(deepcopy(self.model.layer2[0].bn1.weight))
        self.bias_6.append(deepcopy(self.model.layer2[0].bn1.bias))
        self.running_mean_6.append(deepcopy(self.model.layer2[0].bn1.running_mean))
        self.running_var_6.append(deepcopy(self.model.layer2[0].bn1.running_var))
        
        self.weight_7.append(deepcopy(self.model.layer2[0].bn2.weight))
        self.bias_7.append(deepcopy(self.model.layer2[0].bn2.bias))
        self.running_mean_7.append(deepcopy(self.model.layer2[0].bn2.running_mean))
        self.running_var_7.append(deepcopy(self.model.layer2[0].bn2.running_var))
        
        self.weight_8.append(deepcopy(self.model.layer2[0].downsample[1].weight))
        self.bias_8.append(deepcopy(self.model.layer2[0].downsample[1].bias))
        self.running_mean_8.append(deepcopy(self.model.layer2[0].downsample[1].running_mean))
        self.running_var_8.append(deepcopy(self.model.layer2[0].downsample[1].running_var))
        
        self.weight_9.append(deepcopy(self.model.layer2[1].bn1.weight))
        self.bias_9.append(deepcopy(self.model.layer2[1].bn1.bias))
        self.running_mean_9.append(deepcopy(self.model.layer2[1].bn1.running_mean))
        self.running_var_9.append(deepcopy(self.model.layer2[1].bn1.running_var))
        
        self.weight_10.append(deepcopy(self.model.layer2[1].bn2.weight))
        self.bias_10.append(deepcopy(self.model.layer2[1].bn2.bias))
        self.running_mean_10.append(deepcopy(self.model.layer2[1].bn2.running_mean))
        self.running_var_10.append(deepcopy(self.model.layer2[1].bn2.running_var))
        
        self.weight_11.append(deepcopy(self.model.layer3[0].bn1.weight))
        self.bias_11.append(deepcopy(self.model.layer3[0].bn1.bias))
        self.running_mean_11.append(deepcopy(self.model.layer3[0].bn1.running_mean))
        self.running_var_11.append(deepcopy(self.model.layer3[0].bn1.running_var))
        
        self.weight_12.append(deepcopy(self.model.layer3[0].bn2.weight))
        self.bias_12.append(deepcopy(self.model.layer3[0].bn2.bias))
        self.running_mean_12.append(deepcopy(self.model.layer3[0].bn2.running_mean))
        self.running_var_12.append(deepcopy(self.model.layer3[0].bn2.running_var))
        
        self.weight_13.append(deepcopy(self.model.layer3[0].downsample[1].weight))
        self.bias_13.append(deepcopy(self.model.layer3[0].downsample[1].bias))
        self.running_mean_13.append(deepcopy(self.model.layer3[0].downsample[1].running_mean))
        self.running_var_13.append(deepcopy(self.model.layer3[0].downsample[1].running_var))
        
        self.weight_14.append(deepcopy(self.model.layer3[1].bn1.weight))
        self.bias_14.append(deepcopy(self.model.layer3[1].bn1.bias))
        self.running_mean_14.append(deepcopy(self.model.layer3[1].bn1.running_mean))
        self.running_var_14.append(deepcopy(self.model.layer3[1].bn1.running_var))
        
        self.weight_15.append(deepcopy(self.model.layer3[1].bn2.weight))
        self.bias_15.append(deepcopy(self.model.layer3[1].bn2.bias))
        self.running_mean_15.append(deepcopy(self.model.layer3[1].bn2.running_mean))
        self.running_var_15.append(deepcopy(self.model.layer3[1].bn2.running_var))
        
        self.weight_16.append(deepcopy(self.model.layer4[0].bn1.weight))
        self.bias_16.append(deepcopy(self.model.layer4[0].bn1.bias))
        self.running_mean_16.append(deepcopy(self.model.layer4[0].bn1.running_mean))
        self.running_var_16.append(deepcopy(self.model.layer4[0].bn1.running_var))
        
        self.weight_17.append(deepcopy(self.model.layer4[0].bn2.weight))
        self.bias_17.append(deepcopy(self.model.layer4[0].bn2.bias))
        self.running_mean_17.append(deepcopy(self.model.layer4[0].bn2.running_mean))
        self.running_var_17.append(deepcopy(self.model.layer4[0].bn2.running_var))
        
        self.weight_18.append(deepcopy(self.model.layer4[0].downsample[1].weight))
        self.bias_18.append(deepcopy(self.model.layer4[0].downsample[1].bias))
        self.running_mean_18.append(deepcopy(self.model.layer4[0].downsample[1].running_mean))
        self.running_var_18.append(deepcopy(self.model.layer4[0].downsample[1].running_var))
        
        self.weight_19.append(deepcopy(self.model.layer4[1].bn1.weight))
        self.bias_19.append(deepcopy(self.model.layer4[1].bn1.bias))
        self.running_mean_19.append(deepcopy(self.model.layer4[1].bn1.running_mean))
        self.running_var_19.append(deepcopy(self.model.layer4[1].bn1.running_var))
        
        self.weight_20.append(deepcopy(self.model.layer4[1].bn2.weight))
        self.bias_20.append(deepcopy(self.model.layer4[1].bn2.bias))
        self.running_mean_20.append(deepcopy(self.model.layer4[1].bn2.running_mean))
        self.running_var_20.append(deepcopy(self.model.layer4[1].bn2.running_var))
        
    def log_bn_parameters(self, full_exp_name=None):
        print("Logging BN parameters")
        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_1.txt", 'wb') as fp:
            pickle.dump(self.weight_1, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_1.txt", 'wb') as fp:
            pickle.dump(self.bias_1, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_1.txt", 'wb') as fp:
            pickle.dump(self.running_mean_1, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_1.txt", 'wb') as fp:
            pickle.dump(self.running_var_1, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_2.txt", 'wb') as fp:
            pickle.dump(self.weight_2, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_2.txt", 'wb') as fp:
            pickle.dump(self.bias_2, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_2.txt", 'wb') as fp:
            pickle.dump(self.running_mean_2, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_2.txt", 'wb') as fp:
            pickle.dump(self.running_var_2, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_3.txt", 'wb') as fp:
            pickle.dump(self.weight_3, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_3.txt", 'wb') as fp:
            pickle.dump(self.bias_3, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_3.txt", 'wb') as fp:
            pickle.dump(self.running_mean_3, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_3.txt", 'wb') as fp:
            pickle.dump(self.running_var_3, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_4.txt", 'wb') as fp:
            pickle.dump(self.weight_4, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_4.txt", 'wb') as fp:
            pickle.dump(self.bias_4, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_4.txt", 'wb') as fp:
            pickle.dump(self.running_mean_4, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_4.txt", 'wb') as fp:
            pickle.dump(self.running_var_4, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_5.txt", 'wb') as fp:
            pickle.dump(self.weight_5, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_5.txt", 'wb') as fp:
            pickle.dump(self.bias_5, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_5.txt", 'wb') as fp:
            pickle.dump(self.running_mean_5, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_5.txt", 'wb') as fp:
            pickle.dump(self.running_var_5, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_6.txt", 'wb') as fp:
            pickle.dump(self.weight_6, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_6.txt", 'wb') as fp:
            pickle.dump(self.bias_6, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_6.txt", 'wb') as fp:
            pickle.dump(self.running_mean_6, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_6.txt", 'wb') as fp:
            pickle.dump(self.running_var_6, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_7.txt", 'wb') as fp:
            pickle.dump(self.weight_7, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_7.txt", 'wb') as fp:
            pickle.dump(self.bias_7, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_7.txt", 'wb') as fp:
            pickle.dump(self.running_mean_7, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_7.txt", 'wb') as fp:
            pickle.dump(self.running_var_7, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_8.txt", 'wb') as fp:
            pickle.dump(self.weight_8, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_8.txt", 'wb') as fp:
            pickle.dump(self.bias_8, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_8.txt", 'wb') as fp:
            pickle.dump(self.running_mean_8, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_8.txt", 'wb') as fp:
            pickle.dump(self.running_var_8, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_9.txt", 'wb') as fp:
            pickle.dump(self.weight_9, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_9.txt", 'wb') as fp:
            pickle.dump(self.bias_9, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_9.txt", 'wb') as fp:
            pickle.dump(self.running_mean_9, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_9.txt", 'wb') as fp:
            pickle.dump(self.running_var_9, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_10.txt", 'wb') as fp:
            pickle.dump(self.weight_10, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_10.txt", 'wb') as fp:
            pickle.dump(self.bias_10, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_10.txt", 'wb') as fp:
            pickle.dump(self.running_mean_10, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_10.txt", 'wb') as fp:
            pickle.dump(self.running_var_10, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_11.txt", 'wb') as fp:
            pickle.dump(self.weight_11, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_11.txt", 'wb') as fp:
            pickle.dump(self.bias_11, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_11.txt", 'wb') as fp:
            pickle.dump(self.running_mean_11, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_11.txt", 'wb') as fp:
            pickle.dump(self.running_var_11, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_12.txt", 'wb') as fp:
            pickle.dump(self.weight_12, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_12.txt", 'wb') as fp:
            pickle.dump(self.bias_12, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_12.txt", 'wb') as fp:
            pickle.dump(self.running_mean_12, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_12.txt", 'wb') as fp:
            pickle.dump(self.running_var_12, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_13.txt", 'wb') as fp:
            pickle.dump(self.weight_13, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_13.txt", 'wb') as fp:
            pickle.dump(self.bias_13, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_13.txt", 'wb') as fp:
            pickle.dump(self.running_mean_13, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_13.txt", 'wb') as fp:
            pickle.dump(self.running_var_13, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_14.txt", 'wb') as fp:
            pickle.dump(self.weight_14, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_14.txt", 'wb') as fp:
            pickle.dump(self.bias_14, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_14.txt", 'wb') as fp:
            pickle.dump(self.running_mean_14, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_14.txt", 'wb') as fp:
            pickle.dump(self.running_var_14, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_15.txt", 'wb') as fp:
            pickle.dump(self.weight_15, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_15.txt", 'wb') as fp:
            pickle.dump(self.bias_15, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_15.txt", 'wb') as fp:
            pickle.dump(self.running_mean_15, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_15.txt", 'wb') as fp:
            pickle.dump(self.running_var_15, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_16.txt", 'wb') as fp:
            pickle.dump(self.weight_16, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_16.txt", 'wb') as fp:
            pickle.dump(self.bias_16, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_16.txt", 'wb') as fp:
            pickle.dump(self.running_mean_16, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_16.txt", 'wb') as fp:
            pickle.dump(self.running_var_16, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_17.txt", 'wb') as fp:
            pickle.dump(self.weight_17, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_17.txt", 'wb') as fp:
            pickle.dump(self.bias_17, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_17.txt", 'wb') as fp:
            pickle.dump(self.running_mean_17, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_17.txt", 'wb') as fp:
            pickle.dump(self.running_var_17, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_18.txt", 'wb') as fp:
            pickle.dump(self.weight_18, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_18.txt", 'wb') as fp:
            pickle.dump(self.bias_18, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_18.txt", 'wb') as fp:
            pickle.dump(self.running_mean_18, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_18.txt", 'wb') as fp:
            pickle.dump(self.running_var_18, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_19.txt", 'wb') as fp:
            pickle.dump(self.weight_19, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_19.txt", 'wb') as fp:
            pickle.dump(self.bias_19, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_19.txt", 'wb') as fp:
            pickle.dump(self.running_mean_19, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_19.txt", 'wb') as fp:
            pickle.dump(self.running_var_19, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_20.txt", 'wb') as fp:
            pickle.dump(self.weight_20, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_20.txt", 'wb') as fp:
            pickle.dump(self.bias_20, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_20.txt", 'wb') as fp:
            pickle.dump(self.running_mean_20, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_20.txt", 'wb') as fp:
            pickle.dump(self.running_var_20, fp)

        
        
    def load_bn_parameters(self, full_exp_name=None, gamma_beta=False, task=0):

        if gamma_beta:
            print("Load BN parameters")
        else:
            print("Load mu, var")

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_1.txt", 'rb') as fp:
                self.model.bn1.weight  = pickle.load( fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_1.txt", 'rb') as fp:
                self.model.bn1.bias  = pickle.load( fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_1.txt", 'rb') as fp:
            self.model.bn1.running_mean  = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_1.txt", 'rb') as fp:
            self.model.bn1.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_2.txt", 'rb') as fp:
                self.model.layer1[0].bn1.weight  = pickle.load( fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_2.txt", 'rb') as fp:
                self.model.layer1[0].bn1.bias = pickle.load( fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_2.txt", 'rb') as fp:
            self.model.layer1[0].bn1.running_mean = pickle.load( fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_2.txt", 'rb') as fp:
            self.model.layer1[0].bn1.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_3.txt", 'rb') as fp:
                self.model.layer1[0].bn2.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_3.txt", 'rb') as fp:
                self.model.layer1[0].bn2.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_3.txt", 'rb') as fp:
            self.model.layer1[0].bn2.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_3.txt", 'rb') as fp:
            self.model.layer1[0].bn2.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_4.txt", 'rb') as fp:
                self.model.layer1[1].bn1.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_4.txt", 'rb') as fp:
                self.model.layer1[1].bn1.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_4.txt", 'rb') as fp:
            self.model.layer1[1].bn1.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_4.txt", 'rb') as fp:
            self.model.layer1[1].bn1.running_var= pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_5.txt", 'rb') as fp:
                self.model.layer1[1].bn2.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_5.txt", 'rb') as fp:
                self.model.layer1[1].bn2.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_5.txt", 'rb') as fp:
            self.model.layer1[1].bn2.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_5.txt", 'rb') as fp:
            self.model.layer1[1].bn2.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_6.txt", 'rb') as fp:
                self.model.layer2[0].bn1.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_6.txt", 'rb') as fp:
                self.model.layer2[0].bn1.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_6.txt", 'rb') as fp:
            self.model.layer2[0].bn1.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_6.txt", 'rb') as fp:
            self.model.layer2[0].bn1.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_7.txt", 'rb') as fp:
                self.model.layer2[0].bn2.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_7.txt", 'rb') as fp:
                self.model.layer2[0].bn2.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_7.txt", 'rb') as fp:
            self.model.layer2[0].bn2.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_7.txt", 'rb') as fp:
            self.model.layer2[0].bn2.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_8.txt", 'rb') as fp:
                self.model.layer2[0].downsample[1].weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_8.txt", 'rb') as fp:
                self.model.layer2[0].downsample[1].bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_8.txt", 'rb') as fp:
            self.model.layer2[0].downsample[1].running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_8.txt", 'rb') as fp:
            self.model.layer2[0].downsample[1].running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_9.txt", 'rb') as fp:
                self.model.layer2[1].bn1.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_9.txt", 'rb') as fp:
                self.model.layer2[1].bn1.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_9.txt", 'rb') as fp:
            self.model.layer2[1].bn1.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_9.txt", 'rb') as fp:
            self.model.layer2[1].bn1.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_10.txt", 'rb') as fp:
                self.model.layer2[1].bn2.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_10.txt", 'rb') as fp:
                self.model.layer2[1].bn2.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_10.txt", 'rb') as fp:
            self.model.layer2[1].bn2.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_10.txt", 'rb') as fp:
            self.model.layer2[1].bn2.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_11.txt", 'rb') as fp:
                self.model.layer3[0].bn1.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_11.txt", 'rb') as fp:
                self.model.layer3[0].bn1.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_11.txt", 'rb') as fp:
            self.model.layer3[0].bn1.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_11.txt", 'rb') as fp:
            self.model.layer3[0].bn1.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_12.txt", 'rb') as fp:
                self.model.layer3[0].bn2.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_12.txt", 'rb') as fp:
                self.model.layer3[0].bn2.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_12.txt", 'rb') as fp:
            self.model.layer3[0].bn2.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_12.txt", 'rb') as fp:
            self.model.layer3[0].bn2.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_13.txt", 'rb') as fp:
                self.model.layer3[0].downsample[1].weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_13.txt", 'rb') as fp:
                self.model.layer3[0].downsample[1].bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_13.txt", 'rb') as fp:
            self.model.layer3[0].downsample[1].running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_13.txt", 'rb') as fp:
            self.model.layer3[0].downsample[1].running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_14.txt", 'rb') as fp:
                self.model.layer3[1].bn1.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_14.txt", 'rb') as fp:
                self.model.layer3[1].bn1.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_14.txt", 'rb') as fp:
            self.model.layer3[1].bn1.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_14.txt", 'rb') as fp:
            self.model.layer3[1].bn1.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_15.txt", 'rb') as fp:
                self.model.layer3[1].bn2.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_15.txt", 'rb') as fp:
                self.model.layer3[1].bn2.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_15.txt", 'rb') as fp:
            self.model.layer3[1].bn2.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_15.txt", 'rb') as fp:
            self.model.layer3[1].bn2.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_16.txt", 'rb') as fp:
                self.model.layer4[0].bn1.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_16.txt", 'rb') as fp:
                self.model.layer4[0].bn1.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_16.txt", 'rb') as fp:
            self.model.layer4[0].bn1.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_16.txt", 'rb') as fp:
            self.model.layer4[0].bn1.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_17.txt", 'rb') as fp:
                self.model.layer4[0].bn2.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_17.txt", 'rb') as fp:
                self.model.layer4[0].bn2.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_17.txt", 'rb') as fp:
            self.model.layer4[0].bn2.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_17.txt", 'rb') as fp:
            self.model.layer4[0].bn2.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_18.txt", 'rb') as fp:
                self.model.layer4[0].downsample[1].weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_18.txt", 'rb') as fp:
                self.model.layer4[0].downsample[1].bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_18.txt", 'rb') as fp:
            self.model.layer4[0].downsample[1].running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_18.txt", 'rb') as fp:
            self.model.layer4[0].downsample[1].running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_19.txt", 'rb') as fp:
                self.model.layer4[1].bn1.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_19.txt", 'rb') as fp:
                self.model.layer4[1].bn1.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_19.txt", 'rb') as fp:
            self.model.layer4[1].bn1.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_19.txt", 'rb') as fp:
            self.model.layer4[1].bn1.running_var = pickle.load(fp)[task]

        if gamma_beta:
            with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_20.txt", 'rb') as fp:
                self.model.layer4[1].bn2.weight = pickle.load(fp)[task]
            with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_20.txt", 'rb') as fp:
                self.model.layer4[1].bn2.bias = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_20.txt", 'rb') as fp:
            self.model.layer4[1].bn2.running_mean = pickle.load(fp)[task]
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_20.txt", 'rb') as fp:
            self.model.layer4[1].bn2.running_var= pickle.load(fp)[task]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
