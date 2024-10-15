import torch
import warnings
from copy import deepcopy
from argparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset

from itertools import cycle

class Appr(Inc_Learning_Appr):
    """Class implementing the End-to-end Incremental Learning (EEIL) approach described in
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.pdf
    Original code available at https://github.com/fmcp/EndToEndIncrementalLearning
    Helpful code from https://github.com/arthurdouillard/incremental_learning.pytorch
    """

    def __init__(self, model, device, nepochs=90, lr_scheduler = 'multisteplr', lr=0.1, lr_min=1e-6, lr_factor=10, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0001, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=1.0, T=2, lr_finetuning_factor=0.1,
                 nepochs_finetuning=40, noise_grad=False, 
                 con_temp = 0.1, con_alpha = 1.0, con_strategy = 'SimCLR', batch_size=64, amp=False, fix_batch=False, batch_ratio=3, seperate_batch=False, bias_analysis=None, model_freeze=False, change_mu=False, noise=0, fix_bn_parameters=False, cn=8, split_group=False):
        super(Appr, self).__init__(model, device, nepochs, lr_scheduler, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, con_temp, con_alpha, con_strategy, batch_size, amp, fix_batch, batch_ratio, seperate_batch, bias_analysis, model_freeze, change_mu, noise, fix_bn_parameters,cn, split_group)
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.lr_finetuning_factor = lr_finetuning_factor
        self.nepochs_finetuning = nepochs_finetuning
        self.noise_grad = noise_grad

        self._train_epoch = 0
        self._finetuning_balanced = None

        # EEIL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: EEIL is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Added trade-off between the terms of Eq. 1 -- L = L_C + lamb * L_D
        parser.add_argument('--lamb', default=1.0, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 6: "Based on our empirical results, we set T to 2 for all our experiments"
        parser.add_argument('--T', default=2.0, type=float, required=False,
                            help='Temperature scaling (default=%(default)s)')
        # "The same reduction is used in the case of fine-tuning, except that the starting rate is 0.01."
        parser.add_argument('--lr-finetuning-factor', default=0.01, type=float, required=False,
                            help='Finetuning learning rate factor (default=%(default)s)')
        # Number of epochs for balanced training
        parser.add_argument('--nepochs-finetuning', default=40, type=int, required=False,
                            help='Number of epochs for balanced training (default=%(default)s)')
        # the addition of noise to the gradients
        parser.add_argument('--noise-grad', action='store_true',
                            help='Add noise to gradients (default=%(default)s)')
        return parser.parse_known_args(args)

    def _train_unbalanced(self, t, current_loader, val_loader, unsup_loader = None, exemplars_loader = None):
        """Unbalanced training"""
        self._finetuning_balanced = False
        self._train_epoch = 0
        loader = self._get_train_loader(current_loader, False)
        
        if exemplars_loader != None:
            super().train_loop(t, current_loader, val_loader, unsup_loader, exemplars_loader)
        else:
            
            super().train_loop(t, trn_loader, val_loader, unsup_loader)
            
        return loader

    def _train_balanced(self, t, current_loader, val_loader, unsup_loader = None, exemplars_loader = None):
        """Balanced finetuning"""
        self._finetuning_balanced = True
        self._train_epoch = 0
        orig_lr = self.lr
        self.lr *= self.lr_finetuning_factor
        orig_nepochs = self.nepochs
        self.nepochs = self.nepochs_finetuning
        loader = self._get_train_loader(current_loader, True)
#         super().train_loop(t, loader, exemplars_loader, val_loader)

        if exemplars_loader != None:
            super().train_loop(t, current_loader, val_loader, unsup_loader, exemplars_loader)
        else:
            
            super().train_loop(t, trn_loader, val_loader, unsup_loader)
            
            
        self.lr = orig_lr
        self.nepochs = orig_nepochs

    def _get_train_loader(self, trn_loader, balanced=False):
        """Modify loader to be balanced or unbalanced"""
        exemplars_ds = self.exemplars_dataset
        trn_dataset = trn_loader.dataset
        if balanced:
            indices = torch.randperm(len(trn_dataset))
            trn_dataset = torch.utils.data.Subset(trn_dataset, indices[:len(exemplars_ds)])
        ds = exemplars_ds + trn_dataset
        return DataLoader(ds, batch_size=trn_loader.batch_size,
                              shuffle=True,
                              num_workers=trn_loader.num_workers,
                              pin_memory=trn_loader.pin_memory, drop_last=True)

    def _noise_grad(self, parameters, iteration, eta=0.3, gamma=0.55):
        """Add noise to the gradients"""
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        variance = eta / ((1 + iteration) ** gamma)
        for p in parameters:
            p.grad.add_(torch.randn(p.grad.shape, device=p.grad.device) * variance)
            
    def train_loop(self, t, trn_loader, val_loader, unsup_loader = None):
        """Contains the epochs loop"""

        exemplars_loader = None
        current_loader = None
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory, drop_last=True)
            if self.fix_batch:
                current_batch = int(self.batch_size * self.batch_ratio / (self.batch_ratio + 1))
                current_loader = torch.utils.data.DataLoader(trn_loader.dataset,
                                                     batch_size=current_batch,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory, drop_last=True)
                exemplars_batch = int(self.batch_size / (self.batch_ratio + 1))
                exemplars_loader = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                     batch_size=exemplars_batch,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory, drop_last=True)



        # FINETUNING TRAINING -- contains the epochs loop
        if len(self.exemplars_dataset) > 0 and t > 0:
            if self.fix_batch:
#                 super().train_loop(t, current_loader, val_loader, unsup_loader, exemplars_loader)
                
                current_loader = self._train_unbalanced(t, current_loader, val_loader, unsup_loader, exemplars_loader)
                # Balanced fine-tunning (new + old)
                self._train_balanced(t, current_loader, val_loader, unsup_loader, exemplars_loader)
                
            else:
#                 super().train_loop(t, trn_loader, val_loader, unsup_loader)
                
                loader = self._train_unbalanced(t, current_loader, val_loader, unsup_loader, exemplars_loader)
                # Balanced fine-tunning (new + old)
                self._train_balanced(t, current_loader, val_loader, unsup_loader, exemplars_loader)
        else:
            super().train_loop(t, trn_loader, val_loader, unsup_loader)
            
        
        # After task training： update exemplars
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)


#     def train_loop(self, t, trn_loader, val_loader):
#         """Contains the epochs loop"""
#         if t == 0:  # First task is simple training
#             super().train_loop(t, trn_loader, val_loader)
#             loader = trn_loader
#         else:
#             # Page 4: "4. Incremental Learning" -- Only modification is that instead of preparing examplars before
#             # training, we do it online using the stored old model.

#             # Training process (new + old) - unbalanced training
#             loader = self._train_unbalanced(t, trn_loader, val_loader)
#             # Balanced fine-tunning (new + old)
#             self._train_balanced(t, trn_loader, val_loader)

#         # After task training： update exemplars
#         self.exemplars_dataset.collect_exemplars(self.model, loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        if torch.cuda.device_count() > 1:
            self.model_old.module.freeze_all()
        else:
            self.model_old.freeze_all()
            

#     def train_epoch(self, t, trn_loader):
#         """Runs a single epoch"""
#         self.model.train()
#         if self.fix_bn and t > 0:
#             self.model.freeze_bn()
#         for images, targets in trn_loader:
#             images = images.to(self.device)
#             # Forward old model
#             outputs_old = None
#             if t > 0:
#                 outputs_old = self.model_old(images)
#             # Forward current model
#             outputs = self.model(images)
#             loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
#             # Backward
#             self.optimizer.zero_grad()
#             loss.backward()
#             # Page 8: "We apply L2-regularization and random noise [21] (with parameters eta = 0.3, gamma = 0.55)
#             # on the gradients to minimize overfitting"
#             # https://github.com/fmcp/EndToEndIncrementalLearning/blob/master/cnn_train_dag_exemplars.m#L367
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
#             if self.noise_grad:
#                 self._noise_grad(self.model.parameters(), self._train_epoch)
#             self.optimizer.step()
#         self._train_epoch += 1


    def train_epoch(self, t, trn_loader, unsup_loader = None, exemplars_loader = None):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        
        if exemplars_loader != None:
                
            if len(trn_loader) >= len(exemplars_loader):
                exemplars_loader = cycle(exemplars_loader)
            else:
                trn_loader = cycle(trn_loader)

            for i, data in enumerate(zip(trn_loader, exemplars_loader)):

                (current_images, current_targets), (previous_images, previous_targets) = data
#                     print("iteration : ", i)
#                     print("current targets : ", current_targets, "shape ", current_targets.shape)
#                     print("previous targets : ", previous_targets, "shape ", previous_targets.shape)


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


        
                images, targets = images.to(self.device), targets.to(self.device)

                utputs_old = None
                if t > 0:
                    outputs_old = self.model_old(images)
                # Forward current model
                outputs = self.model(images)
                loss = self.criterion(t, outputs, targets, outputs_old)
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                if self.noise_grad:
                    self._noise_grad(self.model.parameters(), self._train_epoch)
                self.optimizer.step()
            
            self._train_epoch += 1
            
        else:

            for images, targets in trn_loader:
                # Forward old model
                outputs_old = None
                if t > 0:
                    outputs_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                if self.noise_grad:
                        self._noise_grad(self.model.parameters(), self._train_epoch)
                self.optimizer.step()
            
            self._train_epoch += 1

    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""

        # Classification loss for new classes
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        # Distilation loss
        if t > 0 and outputs_old:
            # take into account current head when doing balanced finetuning
            last_head_idx = t if self._finetuning_balanced else (t - 1)
            for i in range(last_head_idx):
                loss += self.lamb * F.binary_cross_entropy(F.softmax(outputs[i] / self.T, dim=1),
                                                           F.softmax(outputs_old[i] / self.T, dim=1))
        return loss
