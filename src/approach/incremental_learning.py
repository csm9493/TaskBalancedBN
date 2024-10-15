import time
import torch
import torch.optim as optim

import numpy as np
from argparse import ArgumentParser

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset

from losses import SupConLoss, MoCoLoss, SwavLoss
from itertools import cycle

class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, nepochs=100, lr_scheduler = 'multisteplr', lr=0.05, lr_min=1e-4, lr_factor=3, 
                 lr_patience=5, clipgrad=10000,momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None, 
                 con_temp = 0.1, con_alpha = 1.0, con_strategy = 'SimCLR', batch_size=64, amp=False, fix_batch = False, batch_ratio=3, seperate_batch=False, bias_analysis='plain', model_freeze = False, change_mu = False, noise=0, fix_bn_parameters=False,cn=8, split_group=False):
        self.model = model
        
        self.device = device
        print("device:",self.device)
        self.nepochs = nepochs
        self.lr_scheduler = lr_scheduler
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.optimizer = None
        self.criterion_supconloss = SupConLoss(temperature=con_temp)
        self.criterion_mocoloss = MoCoLoss()
        self.criterion_swavloss = SwavLoss()
        self.con_alpha = con_alpha
        self.m = 0.99
        self.con_strategy = con_strategy
        self.queue = None #Swav
        self.batch_size = batch_size
        self.amp = amp
        self.fix_batch = fix_batch
        self.batch_ratio = batch_ratio
        self.seperate_batch = seperate_batch
        self.bias_analysis = bias_analysis
        self.model_freeze = model_freeze
        self.change_mu = change_mu
        self.fix_bn_parameters=fix_bn_parameters
        self.schedule = [30, 60, 80, 90]
        self.balanced_aug = False
        print("self.bias_analysis", self.bias_analysis)


    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
    
    def set_schedule(self, schedule):
        
        print ('set schedule : ', schedule)
        self.schedule = schedule
    
    def set_hyperparameters(self, t, num_epochs=100, lr=0.1, schedule = [40, 80], ep_factor = 4, decay_lr = False):
        
        if t == 0:
            
            self.lr = lr
            self.nepochs = num_epochs
            self.schedule = np.array([40, 80])
            
        else:
            
            
            if decay_lr:
                self.lr = lr / (t+1)
            else:
                self.lr = lr
            self.nepochs = num_epochs // ep_factor
            self.schedule = np.array([40, 80]) // ep_factor
            
        print ('set hyperparameters [lr, nepochs, schedule]: ', self.lr, self.nepochs, self.schedule)
        
        return 

    def train(self, t, trn_loader, val_loader, unsup_loader = None):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader, unsup_loader)
        self.post_train_process(t, trn_loader)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""

        # Warm-up phase
        if self.warmup_epochs and t > 0:
            self.optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                self.model.heads[-1].train()
                for images, targets in trn_loader:
                    outputs = self.model(images.to(self.device))
                    loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.heads[-1].parameters(), self.clipgrad)
                    self.optimizer.step()
                warmupclock1 = time.time()
                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for images, targets in trn_loader:
                        outputs = self.model(images.to(self.device))
                        loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                        pred = torch.zeros_like(targets.to(self.device))
                        for m in range(len(pred)):
                            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
                        hits_taw = (pred == targets.to(self.device)).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(trn_loader.dataset.labels)
                trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")

    def train_loop(self, t, trn_loader, val_loader, unsup_loader = None, exemplars_loader = None):
        """Contains the epochs loop"""
        if self.change_mu and t<9:
            print("loop passed")
        else:
            lr = self.lr
            best_loss = np.inf
            patience = self.lr_patience
            #best_model = self.model.get_copy()

            self.optimizer = self._get_optimizer()
    #         print ('self.lr_scheduler : ' , self.lr_scheduler)
            if 'multisteplr' in self.lr_scheduler :
                scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=1/self.lr_factor)

            # Loop epochs
            for e in range(self.nepochs):
                # Train
                clock0 = time.time()

                # optionally starts a queue
                if e >= 15 and self.queue is None:
                    self.queue = torch.zeros(
                        1, # len(args.crops_for_assign)
                        #args.queue_length // args.world_size,
                        3840, # queue_length = 3840
                        128, # args.feat_dim
                    ).cuda()
                    #print("queue.shape", queue.shape) #

                self.train_epoch(t, trn_loader, unsup_loader, exemplars_loader)
                clock1 = time.time()
                if self.eval_on_train:
                    train_loss, train_acc, _ , _, _, _, _= self.eval(t, trn_loader)
                    clock2 = time.time()
                    print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                        e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                    self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                    self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
                else:
                    print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

                # Valid
                clock3 = time.time()
                valid_loss, valid_acc, _, _, _, _, _ = self.eval(t, val_loader)
                clock4 = time.time()
                print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    clock4 - clock3, valid_loss, 100 * valid_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")


                if 'adaptive' in self.lr_scheduler:

                    # Adapt learning rate - patience scheme - early stopping regularization
                    if valid_loss < best_loss:
                        # if the loss goes down, keep it as the best model and end line with a star ( * )
                        best_loss = valid_loss
                        best_model = self.model.get_copy()
                        patience = self.lr_patience
                        print(' *', end='')
                    else:
                        # if the loss does not go down, decrease patience
                        patience -= 1
                        if patience <= 0:
                            # if it runs out of patience, reduce the learning rate
                            lr /= self.lr_factor
                            print(' lr={:.1e}'.format(lr), end='')
                            if lr < self.lr_min:
                                # if the lr decreases below minimum, stop the training session
                                print()
                                break
                            # reset patience and recover best model so far to continue training
                            patience = self.lr_patience
                            self.optimizer.param_groups[0]['lr'] = lr
                            self.model.set_state_dict(best_model)

                else:
                    if valid_loss < best_loss:
                        # if the loss goes down, keep it as the best model and end line with a star ( * )
                        best_loss = valid_loss
                        
                        if torch.cuda.device_count() > 1:
                        
                            best_model = self.model.module.get_copy()
                            
                        else:
                            
                            best_model = self.model.get_copy()
                            
                        print(' *', end='')
    #                 else:
    #                     self.model.set_state_dict(best_model)
                    scheduler.step()
        


                self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
                print()
            #self.model.log_running_parameter(trn_loader)
            
            if torch.cuda.device_count() > 1:
                self.model.module.set_state_dict(best_model)
            else:
                self.model.set_state_dict(best_model)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader, unsup_loader = None, exemplars_loader = None):
        """Runs a single epoch"""
        
        self.model.train()
#         avg_loss = []
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
            
        if unsup_loader != None:
            if len(trn_loader) >= len(unsup_loader):
                unsup_loader = cycle(unsup_loader)
            else:
                trn_loader = cycle(trn_loader)
                
            
            iteration = 400
            for i, data in enumerate(zip(trn_loader, unsup_loader)):
                
                if t == 0:
                    iteration = i
                
                (images, targets), unsup_images = data

                if images.shape[0] != self.batch_size:
                    continue
                if unsup_images[0].shape[0] != self.batch_size or unsup_images[1].shape[0] != self.batch_size:
                    continue


                # Forward current model with current + mem dataset
                outputs = self.model(images.to(self.device),return_features=False)
                loss = self.criterion(t, outputs, targets.to(self.device))
                
                # Forward current model with unsuperivsed dataset (Contrastive learning)
                unsup_images = torch.cat([unsup_images[0], unsup_images[1]], dim=0)
                bsz = unsup_images.shape[0]//2
                
                outputs, f1, f2 = self.model(unsup_images.to(self.device), return_features = True) #( 128, 3000 ) ( batch*2 , prototypes)
                #f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                #features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                
                if self.con_strategy == 'MoCo':
                    loss_cont = self.criterion_mocoloss(f1, f2)
                elif self.con_strategy == 'SimCLR':
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    loss_cont = self.criterion_supconloss(features)
                elif self.con_strategy == 'Swav':
                    use_the_queue = False
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1).detach()
                    loss_cont = self.criterion_swavloss(f1, f2, outputs, self.queue, use_the_queue, self.model)
                
                loss = loss + loss_cont * self.con_alpha

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
    #             avg_loss.append(loss.item())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
        
                if iteration < 313: # 313 : args.freeze_prototypes_niters:
                    for name, p in self.model.named_parameters():
                        if "prototypes" in name:
                            p.grad = None
                            
                self.optimizer.step()
            
        else:
            # if fix_batch used
            if exemplars_loader != None:
                if len(trn_loader) >= len(exemplars_loader):
                    exemplars_loader = cycle(exemplars_loader)
                else:
                    trn_loader = cycle(trn_loader)
                i=1
                for i, data in enumerate(zip(trn_loader, exemplars_loader)):
                    
                    (current_images, current_targets), (previous_images, previous_targets) = data
                    
                    if self.balanced_aug == True:
                        
                        previous_images = torch.cat([previous_images[aug_i] for aug_i in range(t*3)], dim = 0)
                        previous_targets = torch.cat([previous_targets for aug_i in range(t*3)], dim = 0)
                        current_images = current_images[0]
                        
                        print (previous_images.shape, previous_targets.shape)
                    
                    if self.seperate_batch:
                        
                        # Split_ver2 : Curr-wise norm, Prev-wise norm
                        if True:
                            print("Test for Split-ver2")
                            if current_images.shape[0] != 48:
                                continue
                            if previous_images.shape[0] != 15:
                                continue
                            splits = 16
                            C, H, W = current_images.shape[1], current_images.shape[2], current_images.shape[3]
                            images = torch.cat([current_images.view(3, C*16, H, W), previous_images.view(3, C*5, H, W)], dim=1).view(63, C, H, W)
                            targets = torch.cat([current_target.view(3, 16), previous_targets.view(3, 5)], dim=1).view(63)
                            
                            outputs = self.model(images.to(self.device))
                            loss = self.criterion(t, outputs, targets.to(self.device))
                        else:

                            if current_images.shape[0] != 48:
                                continue
                            if previous_images.shape[0] != 16:
                                continue
                            if i==1:
                                print("Seperate batch")
                                print("Current targets", current_targets)
                                print("Previous targets", previous_targets)

                            current_outputs = self.model(current_images.to(self.device))
                            loss_current = self.criterion(t, current_outputs, current_targets.to(self.device))
                            previous_outputs = self.model(previous_images.to(self.device))
                            loss_previous = self.criterion(t, previous_outputs, previous_targets.to(self.device))

                            loss = loss_current + loss_previous
                    else:
                        
                        if self.balanced_aug == False:
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


                        outputs = self.model(images.to(self.device))
                        loss = self.criterion(t, outputs, targets.to(self.device))

                    self.optimizer.zero_grad()
                    
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                    
                    self.optimizer.step()
            else:
                
                
                i=1
                iteration = 0
                #total_sum = torch.tensor([0,0,0], dtype=torch.double)
                #num_tracked = 0
                for images, targets in trn_loader:

                    # Forward current model
                    if self.balanced_aug == True:
                        
                        images = images[0]
                    
                    if images.shape[0] != self.batch_size:
                        continue
                    
                    
                    #num_tracked += images.shape[0]
                    #i += 1

                    outputs = self.model(images.to(self.device))

                    if True:
                        loss = self.criterion(t, outputs, targets.to(self.device))

                        self.optimizer.zero_grad()

                    
                        loss.backward()
                        

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                        self.optimizer.step()
                    else:
                        if i==1:
                            print("No backward pass")
                    if i==1:
                        i+=1

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num, total_forg, total_forg_1, total_forg_2, total_forg_3, total_forg_4 = 0, 0, 0, 0, 0, 0, 0, 0, 0
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

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        
        # Task-Aware Multi-Head
        if torch.cuda.device_count() > 1:

            for m in range(len(pred)):
                this_task = (self.model.module.task_cls.cumsum(0) <= targets[m]).sum()
                pred[m] = outputs[this_task][m].argmax() + self.model.module.task_offset[this_task]
            hits_taw = (pred == targets.to(self.device)).float()
            
        else:
            
            for m in range(len(pred)):
                this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
            hits_taw = (pred == targets.to(self.device)).float()
            
        
        
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
    
    def calculate_forgetting_metrics(self, outputs, targets):
        
        forg_1, forg_2, forg_3, forg_4 = 0, 0, 0, 0
        
        if torch.cuda.device_count() > 1:
            task_cumsum = self.model.module.task_cls.cumsum(0)
        else:
            task_cumsum = self.model.task_cls.cumsum(0)
            
        if len(task_cumsum) == 1:
            prev_cls = 0
            cur_cls = 10
        else:
            prev_cls = task_cumsum[-2]
            cur_cls = task_cumsum[-1]
        
        pred = torch.cat(outputs, dim=1).argmax(1) #Tag case
        
        for m in range(len(pred)):
            if pred[m] != targets[m]:
                if targets[m] >= prev_cls : # label : current class
                    if pred[m] >= prev_cls:  # prediction : current class
                        forg_4 += 1
                    else:                # prediction : previous class
                        forg_3 += 1
                elif targets[m] < prev_cls: # label : previous class
                    if pred[m] >= prev_cls: # prediction : current class
                        forg_2 += 1
                    else:                  # prediction : previous class
                        forg_1 += 1
        
        return forg_1, forg_2, forg_3, forg_4
    
    
    
    
    
    
    
    
    
    