import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr_scheduler = 'multisteplr', lr=0.05, lr_min=1e-4, lr_factor=3, 
                 lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, all_outputs=False, con_temp = 0.1, con_alpha = 1.0, con_strategy = 'SimCLR', batch_size=64, amp=False, fix_batch=False, batch_ratio=3, seperate_batch = False, bias_analysis = 'plain', model_freeze = False, change_mu=False, noise=0, fix_bn_parameters=False, cn=8, split_group=False, ablation_balanced_aug = False, eval_train_mode = False):
        super(Appr, self).__init__(model, device, nepochs, lr_scheduler, lr, lr_min, lr_factor, lr_patience, clipgrad, 
                                   momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, con_temp, con_alpha, con_strategy, batch_size, amp, fix_batch, batch_ratio, seperate_batch, bias_analysis, model_freeze, change_mu, noise, fix_bn_parameters, cn, split_group)
        self.all_out = all_outputs
        self.trn_datasets = []
        self.val_datasets = []

        self.balanced_aug = ablation_balanced_aug
        self.eval_train_mode = eval_train_mode
        
    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        parser.add_argument('--ablation-balanced-aug', action='store_true', required=False,
                            help='Ablations study for balanced aug')
        parser.add_argument('--eval-train-mode', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        
        if torch.cuda.device_count() > 1:
        
            if len(self.exemplars_dataset) == 0 and len(self.model.module.heads) > 1 and not self.all_out:
                # if there are no exemplars, previous heads are not modified
                params = list(self.model.module.model.parameters()) + list(self.model.module.heads[-1].parameters())
            else:
                params = self.model.module.parameters()
                
        else:
            
            if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 and not self.all_out:
                # if there are no exemplars, previous heads are not modified
                params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
            else:
                params = self.model.parameters()
                
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader, unsup_loader = None):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        exemplars_loader = None
        current_loader = None
        
        #if self.change_mu:
        #    self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
        
        #if len(self.exemplars_dataset) > 0 and t > 0:
        if t>0:
            
            """
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory,
                                                     worker_init_fn=np.random.seed(0))
            """
            if self.bias_analysis == 'previous':
                print("self.bias_analysis",self.bias_analysis)
                trn_dset = JointDataset(self.trn_datasets)
                val_dset = JointDataset(self.val_datasets)
                self.trn_datasets.append(trn_loader.dataset)
                self.val_datasets.append(val_loader.dataset)
                trn_loader_new = DataLoader(trn_dset,
                                    batch_size=trn_loader.batch_size,
                                    shuffle=True,
                                    num_workers=trn_loader.num_workers,
                                    pin_memory=trn_loader.pin_memory, worker_init_fn=np.random.seed(0))
                val_loader_new = DataLoader(val_dset,
                                batch_size=val_loader.batch_size,
                                shuffle=False,
                                num_workers=val_loader.num_workers,
                                pin_memory=val_loader.pin_memory, worker_init_fn=np.random.seed(0))
                trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory,
                                                     worker_init_fn=np.random.seed(0))
            elif self.bias_analysis == 'current':
                trn_loader = torch.utils.data.DataLoader(trn_loader.dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory,
                                                     worker_init_fn=np.random.seed(0))
            else:
                if self.fix_batch:
                    trn_loader = torch.utils.data.DataLoader(trn_loader.dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory,
                                                     worker_init_fn=np.random.seed(0))
                else: trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory,
                                                     worker_init_fn=np.random.seed(0))
            if self.fix_batch:
                current_batch = int(self.batch_size * self.batch_ratio / (self.batch_ratio + 1))
                current_loader = torch.utils.data.DataLoader(trn_loader.dataset,
                                                     batch_size=current_batch,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory,
                                                     worker_init_fn=np.random.seed(0))
                exemplars_batch = int(self.batch_size / (self.batch_ratio + 1))
                print ('current_batch : ', current_batch, ' exemplars_batch : ', exemplars_batch) 
                exemplars_loader = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                     batch_size=exemplars_batch,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory,
                                                     worker_init_fn=np.random.seed(0))
            

        # FINETUNING TRAINING -- contains the epochs loop
        if len(self.exemplars_dataset) > 0 and t > 0:
            if self.fix_batch:
                print("fix_batch")
                super().train_loop(t, current_loader, val_loader, unsup_loader, exemplars_loader)
            else:
                print("else")
                if self.bias_analysis == 'previous':
                    print("previous")
                    super().train_loop(t, trn_loader_new, val_loader_new, unsup_loader)
                else:
                    print("plain")
                    super().train_loop(t, trn_loader, val_loader, unsup_loader)
                    
        else:
            super().train_loop(t, trn_loader, val_loader, unsup_loader)
            
        if t==0 and self.bias_analysis =='previous':
            self.trn_datasets.append(trn_loader.dataset)
            self.val_datasets.append(val_loader.dataset)
        
        if self.bias_analysis != 'current':
            # EXEMPLAR MANAGEMENT -- select training subset
            print("collect exemplars")
            
            if self.fix_batch:
                trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory,
                                                     worker_init_fn=np.random.seed(0))
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""   
        if self.all_out or len(self.exemplars_dataset) > 0 or t == 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)

        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
    
    
    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num, total_forg, total_forg_1, total_forg_2, total_forg_3, total_forg_4 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            if self.eval_train_mode:
                self.model.train()
            else:
                self.model.eval()
            i = 1
            for images, targets in val_loader:
                # Forward current model
                #if i==1:
                    #print("valid images[0][0][0]", images[0][0][0])
                    #i+=1
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                forg_1, forg_2, forg_3, forg_4 = self.calculate_forgetting_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_forg += forg_1 + forg_2 + forg_3 + forg_4
                total_forg_1 += forg_1
                total_forg_2 += forg_2
                total_forg_3 += forg_3
                total_forg_4 += forg_4
                total_num += len(targets)
            
            """
            print("total_acc_tag",total_acc_tag)
            print("total_forg", total_forg)
            print("total_forg_1", total_forg_1)
            print("total_forg_2", total_forg_2)
            print("total_forg_3", total_forg_3)
            print("total_forg_4", total_forg_4)
            """
            #print("valid total_num", total_num)
#             assert total_acc_tag + total_forg == total_num
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, total_forg_1, total_forg_2, total_forg_3, total_forg_4
    
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

