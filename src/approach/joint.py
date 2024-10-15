import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, lr_scheduler = 'multisteplr', lr=0.05, lr_min=1e-4, lr_factor=3, 
                 lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                     logger=None, exemplars_dataset=None, freeze_after=-1, con_temp = 0.1, con_alpha = 0.0, con_strategy = 'SimCLR', batch_size=64, amp=False, fix_batch=False, batch_ratio=3, seperate_batch = False, bias_analysis='plain', model_freeze=False, change_mu=False, noise=0, fix_bn_parameters=False, cn=8, split_group=False):
        super(Appr, self).__init__(model, device, nepochs, lr_scheduler, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, con_temp, con_alpha, con_strategy, batch_size, amp, fix_batch, batch_ratio, seperate_batch, bias_analysis, model_freeze, change_mu, noise, fix_bn_parameters,cn, split_group)
        self.trn_datasets = []
        self.val_datasets = []
        self.freeze_after = freeze_after
        #print("self.freeze_after", freeze_after)

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        #assert (have_exemplars == 0), 'Warning: Joint does not use exemplars. Comment this line to force it.'

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset
        
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--freeze-after', default=-1, type=int, required=False,
                            help='Freeze model except heads after the specified task'
                                 '(-1: normal Incremental Joint Training, no freeze) (default=%(default)s)')
        return parser.parse_known_args(args)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        if self.freeze_after > -1 and t >= self.freeze_after:
            self.model.freeze_all()
            for head in self.model.heads:
                for param in head.parameters():
                    param.requires_grad = True

    def train_loop(self, t, trn_loader, val_loader, unsup_loader = None):
        """Contains the epochs loop"""

        # add new datasets to existing cumulative ones
        self.trn_datasets.append(trn_loader.dataset)
        self.val_datasets.append(val_loader.dataset)
        trn_dset = JointDataset(self.trn_datasets)
        val_dset = JointDataset(self.val_datasets)
        trn_loader = DataLoader(trn_dset,
                                batch_size=trn_loader.batch_size,
                                shuffle=True,
                                num_workers=trn_loader.num_workers,
                                pin_memory=trn_loader.pin_memory, worker_init_fn=np.random.seed(0))
        val_loader = DataLoader(val_dset,
                                batch_size=val_loader.batch_size,
                                shuffle=False,
                                num_workers=val_loader.num_workers,
                                pin_memory=val_loader.pin_memory, worker_init_fn=np.random.seed(0))
        if self.change_mu and t < 9:
                return
        # continue training as usual
        super().train_loop(t, trn_loader, val_loader, unsup_loader, exemplars_loader=None)

    def train_epoch(self, t, trn_loader, unsup_loader = None, exemplars_loader=None):
        """Runs a single epoch"""
        if self.freeze_after < 0 or t <= self.freeze_after:
            self.model.train()
            if self.fix_bn and t > 0:
                self.model.freeze_bn()
        else:
            self.model.eval()
            for head in self.model.heads:
                head.train()
        i = 1
        iteration = 0
        num_tracked = 0
        for images, targets in trn_loader:
#             if i==1:
#                 print("targets",targets)
                #print("list(self.model.model.layer1[0].conv1.parameters())[0][0][0]", list(self.model.model.layer1[0].conv1.parameters())[0][0][0])
                #print("self.model.model.bn1.weight", self.model.model.bn1.weight)
                #print("self.model.model.bn1.bias", self.model.model.bn1.bias)
                #print("self.model.model.bn1.running_mean", self.model.model.bn1.running_mean)
                #print("self.model.heads[0]",list(self.model.heads[0].parameters()))
                #print("self.model.model.bn1.running_var", self.model.model.bn1.running_var)
                #print("images[0][0][0]",images[0][0][0])
         
            if images.shape[0] != self.batch_size:
                #print("images.shape[0]", images.shape[0])
                continue
            
            #num_tracked += images.shape[0]
            
            # Forward current model
            outputs = self.model(images.to(self.device))
            if True:
                loss = self.criterion(t, outputs, targets.to(self.device))
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.optimizer.step()
            else:
                if i==1:
                    print("No backward pass")
            if i==1:
                i+=1
                #print("after list(self.model.model.layer1[0].conv1.parameters())[0][0][0]", list(self.model.model.layer1[0].conv1.parameters())[0][0][0])
                #print("after self.model.model.bn1.weight", self.model.model.bn1.weight)
                #print("after self.model.model.bn1.bias", self.model.model.bn1.bias)
                #print("after self.model.model.bn1.running_mean", self.model.model.bn1.running_mean)
                #print("after self.model.model.bn1.running_var", self.model.model.bn1.running_var)
                
        #print("num_tracked", num_tracked)
        
#             break

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)


class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                x, y = d[index]
                return x, y
