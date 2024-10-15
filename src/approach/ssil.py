import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from copy import deepcopy
from itertools import cycle
import torch.nn.functional as F

class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr_scheduler = 'multisteplr', lr=0.05, lr_min=1e-4, lr_factor=3, 
                 lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, all_outputs=False, con_temp = 0.1, con_alpha = 1.0, con_strategy = 'SimCLR', batch_size=64, amp=False, fix_batch=False, batch_ratio=3, seperate_batch = False, bias_analysis = 'plain', model_freeze = False, change_mu=False, noise=0, fix_bn_parameters=False, cn=8, split_group=False, eval_train_mode = False):
        super(Appr, self).__init__(model, device, nepochs, lr_scheduler, lr, lr_min, lr_factor, lr_patience, clipgrad, 
                                   momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, con_temp, con_alpha, con_strategy, batch_size, amp, fix_batch, batch_ratio, seperate_batch, bias_analysis, model_freeze, change_mu, noise, fix_bn_parameters, cn, split_group)
        self.all_out = all_outputs
        self.trn_datasets = []
        self.val_datasets = []
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
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
        
        if t>0:
            if self.fix_batch:
                trn_loader = torch.utils.data.DataLoader(trn_loader.dataset,
                                                 batch_size=trn_loader.batch_size,
                                                 shuffle=True,
                                                 num_workers=trn_loader.num_workers,
                                                 pin_memory=trn_loader.pin_memory,
                                                 worker_init_fn=np.random.seed(0))
            else: 
                trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
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
                exemplars_loader = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                     batch_size=exemplars_batch,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory,
                                                     worker_init_fn=np.random.seed(0))
            

        # FINETUNING TRAINING -- contains the epochs loop
        if len(self.exemplars_dataset) > 0 and t > 0:
            if self.fix_batch:
                super().train_loop(t, current_loader, val_loader, unsup_loader, exemplars_loader)
            else:
                super().train_loop(t, trn_loader, val_loader, unsup_loader)    
        else:
            super().train_loop(t, trn_loader, val_loader, unsup_loader)
            
        
        self.model_old = deepcopy(self.model)
        self.model_old.to(self.device)
        self.model_old.eval()
        if torch.cuda.device_count() > 1:
            self.model_old.module.freeze_all()
        else:
            self.model_old.freeze_all()
        
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

    def train_epoch(self, t, trn_loader, unsup_loader=None, exemplars_loader = None):
        """Runs a single epoch"""
        
        self.model.train()
        
        # if fix_batch used and t>0
        if exemplars_loader != None:
            if len(trn_loader) >= len(exemplars_loader):
                    exemplars_loader = cycle(exemplars_loader)
            else:
                trn_loader = cycle(trn_loader)
            
            start = 0
            
            if torch.cuda.device_count() > 1:

                mid = self.model.module.task_offset[t]
                end = mid + self.model.module.task_cls[t]
                
            else:
                
                mid = self.model.task_offset[t]
                end = mid + self.model.task_cls[t]
                
            #print("mid", mid)
            #print("end", end)
            i=1
            for i , data in enumerate(zip(trn_loader, exemplars_loader)):
                
                (current_images, current_targets), (previous_images, previous_targets) = data

                if (current_images.shape[0] + previous_images.shape[0]) != self.batch_size:
                    continue

                if torch.cuda.device_count() > 1:

                    gpu_numbers = torch.cuda.device_count()

                    images = torch.cat((torch.stack(current_images.chunk(gpu_numbers, dim=0), dim = 1), torch.stack(previous_images.chunk(gpu_numbers, dim=0), dim = 1)), dim = 0)
                    targets = torch.cat((torch.stack(current_targets.chunk(gpu_numbers, dim=0), dim = 1), torch.stack(previous_targets.chunk(gpu_numbers, dim=0), dim = 1)), dim = 0)

                    images = torch.cat([images[:,i,:] for i in range (gpu_numbers)])
                    targets = torch.cat([targets[:,i] for i in range (gpu_numbers)])

                else:

                    images = torch.cat((current_images, previous_images),dim=0)
                    targets = torch.cat((current_targets, previous_targets) , dim=0)
                
                current_targets = current_targets % (end-mid)
                cur_batch = current_images.shape[0]
                prev_batch = previous_images.shape[0]
                
                outputs = self.model(images.to(self.device))
                outputs = torch.cat(outputs, dim=1)
                
                loss_KD = 0
                
                loss_CE_curr = 0
                loss_CE_prev = 0
                
                
                curr = outputs[:cur_batch, mid:end]

                loss_CE_curr = self.loss(curr, current_targets.to(self.device))
                
                if t>0 :
                    prev = outputs[cur_batch:images.shape[0], start:mid]
                    loss_CE_prev = self.loss(prev, previous_targets.to(self.device))
                    loss_CE = (loss_CE_curr + loss_CE_prev)/ (cur_batch + prev_batch)
                    
                    score = self.model_old(images.to(self.device))
                    score = torch.cat(score, dim=1)[:,:mid]
                    
                    if torch.cuda.device_count() > 1:
                        tasknum = len(self.model.module.task_cls)
                    else:
                        tasknum = len(self.model.task_cls)
                    loss_KD = torch.zeros(tasknum).to(self.device)
                    
                    for task in range(tasknum-1):
                        
                        if torch.cuda.device_count() > 1:
                            start_KD = self.model.module.task_offset[task]
                            end_KD = start_KD + self.model.module.task_cls[task]
                        else:
                            start_KD = self.model.task_offset[task]
                            end_KD = start_KD + self.model.task_cls[task]
                            
                        
                        T = 2
                        soft_target = F.softmax(score[:, start_KD:end_KD] / T, dim=1)
                        output_log = F.log_softmax(outputs[:, start_KD:end_KD] / T, dim=1)
                        
                        loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                    loss_KD = loss_KD.sum()
                    
                else:
                    loss_CE = loss_CE_curr / cur_batch
                
                
                self.optimizer.zero_grad()
                (loss_CE + loss_KD).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.optimizer.step()
                
        else:
            if t>0:
                print("error, fix-batch should be used")
                return
            i=1
            for data in trn_loader:
                
                images, targets = data
                if i==1:
                    i+=1
#                     print("targets",targets)
                if images.shape[0] != self.batch_size:
                    continue
                
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device))
                
                self.optimizer.zero_grad()
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.optimizer.step()
                
            
                
    
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
        
    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0 or t == 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)

        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
    