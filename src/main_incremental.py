import os
import time
import torch
import argparse
import importlib
import numpy as np
from functools import reduce
from functools import partial
import pickle
import random
import numpy as np

import torch.nn as nn
import utils
import approach
from loggers.exp_logger import MultiLogger
from datasets.data_loader import get_loaders, get_loaders_unsup
from datasets.dataset_config import dataset_config
from last_layer_analysis import last_layer_analysis
from networks import tvmodels, allmodels, set_tvmodel_head_var
from networks.norm_layers import ContinualNorm, CustomSplitBatchNorm
from networks.norm_layers import BatchRenorm2d, GroupNorm2d, SwitchNorm2d
from networks.tbbn import TaskBalancedBN
from copy import deepcopy

from scipy.io import savemat

def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--last-layer-analysis', action='store_true',
                        help='Plot last layer analysis (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    parser.add_argument('--split', default=False, type=bool, required=False,
                        help='Split Batch Norm (default=%(default)s)')
    parser.add_argument('--amp', default=False, type=bool, required=False,
                        help='Automatic Mixed Precision (default=%(default)s)')
    parser.add_argument('--fix-batch', default=True, type=bool,
                        help='whether to use fixed current-memory ratio batch (default=%(default)s)')
    parser.add_argument("--batch-ratio", default=3, type=int,
                    help='current vs memory ratio in a batch')
    parser.add_argument("--log-bn", default=False, type=bool, required=False,
                        help='Save and log Batch Norm parameter (gamma, beta, running)')
    parser.add_argument("--seperate-batch", default=False, type=bool, required=False,
                        help='seperate current vs previous data in a batch')
    parser.add_argument("--model-freeze", default=False, type=bool, required=False,
                        help="freeze model except BN after first task trained")
    parser.add_argument("--bias-analysis", default=None, type=str, choices=['plain', 'current', 'previous'], required=False,
                        help="for bias analysis")
    parser.add_argument("--noise", default=0, type=float, required=False,
                    help='noise scale for NoisyBatchNorm')
    parser.add_argument("--fix-bn-parameters", default=False, type=bool, required=False,
                        help="whether to fix bn-parameters for analysis")
    
    parser.add_argument('--datasets', default=['cifar100'], type=str,
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr-scheduler', default=['multisteplr'], type=str, choices=['adaptive', 'multisteplr'],
                        help='type of lr scheduler', nargs='*', metavar="LOGGER")
    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--bn-momentum', default=0.1, type=float, required=False,
                        help='BN-Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                        help='Warm-up learning rate factor (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    # gridsearch args
    parser.add_argument('--gridsearch-tasks', default=-1, type=int,
                        help='Number of tasks to apply GridSearch (-1: all tasks) (default=%(default)s)')
    
    parser.add_argument('--bn-splits', default=1, type=int, required=False,
                        help='num splits')
    parser.add_argument('--custom-split-bn', action='store_true',
                        help='custom-split-bn')
    parser.add_argument('--tbbn', action='store_true',
                        help='tbbn')
    parser.add_argument('--cn', action='store_true',
                        help='cn')
    parser.add_argument("--cn-num", default=0, type=int, required=False,
                        help="number of groups for continual normalization. if zero, not use CN")
    parser.add_argument('--InstanceNorm', action='store_true',
                        help='InstanceNorm')
    parser.add_argument('--GroupNorm', action='store_true',
                        help='GroupNorm')
    parser.add_argument("--gn", default=16, type=int, required=False,
                        help="number of groups for group normalization. if zero, not use CN")
    parser.add_argument('--BatchReNorm', action='store_true',
                        help='BathReNorm')
    parser.add_argument('--SwitchNorm', action='store_true',
                        help='SwitchNorm')
    parser.add_argument('--LayerNorm', action='store_true',
                        help='SwitchNorm')

    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    
    if  torch.cuda.device_count() > 1:
        args.batch_size *= torch.cuda.device_count()
        args.num_workers = 8 * torch.cuda.device_count()
    
    
    base_kwargs = dict(nepochs=args.nepochs, lr_scheduler = args.lr_scheduler, lr=args.lr, lr_min=args.lr_min, 
                       lr_factor=args.lr_factor,lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train, batch_size = args.batch_size, amp = args.amp, fix_batch = args.fix_batch, batch_ratio = args.batch_ratio, seperate_batch = args.seperate_batch, bias_analysis = args.bias_analysis, model_freeze = args.model_freeze, noise=args.noise, fix_bn_parameters=args.fix_bn_parameters, cn=args.cn_num)
    
    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False
    

    utils.seed_everything(seed=args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)
    
    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
        
    ####################################################################################################################
    
    if args.custom_split_bn:
        norm_layer = partial(CustomSplitBatchNorm, num_splits=args.bn_splits, momentum = args.bn_momentum, batch_ratio=args.batch_ratio)
        print("norm_layer:",norm_layer)
        
    
    elif args.tbbn:
        
        print ('batch_ratio : ', args.batch_ratio, ' B_c : ', (args.batch_size // (args.batch_ratio+1))*args.batch_ratio, ' B_p : ', (args.batch_size // (args.batch_ratio+1)))
        norm_layer = partial(TaskBalancedBN, momentum = args.bn_momentum, batch_ratio=args.batch_ratio, B_c = (args.batch_size // (args.batch_ratio+1))*args.batch_ratio, B_p = (args.batch_size // (args.batch_ratio+1)))
        print("norm_layer:",norm_layer)
        
    elif args.noise>0:
        norm_layer = partial(NoisyBatchNorm, noise_scale=args.noise) if args.noise > 0 else nn.BatchNorm2d
        
    elif args.cn_num > 0:
        norm_layer = partial(ContinualNorm, num_groups=args.cn_num)
        
    elif args.GroupNorm:
        norm_layer = partial(GroupNorm2d, num_groups=args.gn)
        print("norm_layer:",norm_layer, ' groups : ', args.gn)
        
    elif args.BatchReNorm:
        norm_layer = partial(BatchRenorm2d)
        print("norm_layer:",norm_layer)
        
    elif args.SwitchNorm:
        norm_layer = partial(SwitchNorm2d)
        print("norm_layer:",norm_layer)
        
    elif args.InstanceNorm:
        norm_layer = partial(torch.nn.InstanceNorm2d)
        print("norm_layer:",norm_layer)
        
    elif args.LayerNorm:
        norm_layer = partial(torch.nn.LayerNorm)
        print("norm_layer:",norm_layer)
        
    # Args -- Network
    from networks.network import LLL_Net
    if args.network in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
        if args.network == 'googlenet':
            if args.custom_split_bn or args.tbbn or args.BatchReNorm or args.InstanceNorm or args.GroupNorm or args.InstanceNorm or args.LayerNorm or  args.BatchReNorm:
                init_model = tvnet(pretrained=args.pretrained, aux_logits=False, norm_layer = norm_layer)
            else :
                init_model = tvnet(pretrained=args.pretrained, aux_logits=False)
        else:
            if args.custom_split_bn or args.tbbn or args.BatchReNorm or args.InstanceNorm or args.GroupNorm or args.InstanceNorm or args.LayerNorm or  args.BatchReNorm:
                init_model = tvnet(pretrained=args.pretrained, norm_layer = norm_layer)
            elif args.noise:
                init_model = tvnet(pretrained=args.pretrained, norm_layer = norm_layer)
            elif args.cn_num>0:
                init_model = tvnet(pretrained=args.pretrained, norm_layer = norm_layer)
            else :
                init_model = tvnet(pretrained=args.pretrained)
                
        set_tvmodel_head_var(init_model)
        
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name='networks'), args.network)
        print("C")
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        if args.custom_split_bn or args.tbbn or args.BatchReNorm or args.InstanceNorm or args.GroupNorm or args.InstanceNorm or args.LayerNorm or  args.BatchReNorm:
            init_model = net(pretrained=False, norm_layer = norm_layer)
            print ('init_model with : ', norm_layer)
        elif args.cn_num>0:
            init_model = net(pretrained=False, norm_layer = norm_layer)
        else:
            init_model = net(pretrained=args.pretrained)
    
    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    # Args -- GridSearch
    if args.gridsearch_tasks > 0:
        from gridsearch import GridSearch
        gs_args, extra_args = GridSearch.extra_parser(extra_args)
        Appr_finetuning = getattr(importlib.import_module(name='approach.finetuning'), 'Appr')
        assert issubclass(Appr_finetuning, Inc_Learning_Appr)
        GridSearch_ExemplarsDataset = Appr.exemplars_dataset_class()
        print('GridSearch arguments =')
        for arg in np.sort(list(vars(gs_args).keys())):
            print('\t' + arg + ':', getattr(gs_args, arg))
        print('=' * 108)

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    ####################################################################################################################

    # Log all arguments
#     full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    if len(args.datasets) > 1 :
        full_exp_name = 'dissimilar_tasks'
    
    else:
        full_exp_name = args.datasets[0]
    
    full_exp_name += '_' + args.approach
    
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
        
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))
    
    # Loaders
    utils.seed_everything(seed=args.seed)
    
    
    if args.split and args.split_version == 3:
        drop_last = True
    else:
        drop_last = False
        
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory,
                                                              drop_last = True)
    
    
    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    
    # Network and Approach instances
    utils.seed_everything(seed=args.seed)
    
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
    
    #Multiple gpus
    if torch.cuda.device_count() > 1:
        print ('torch.cuda.device_count() : ', torch.cuda.device_count())
        net = torch.nn.DataParallel(net)
        net.cuda()
   
    if torch.cuda.device_count() > 1:
        print("net.task_cls",net.module.task_cls)
    else:
        print("net.task_cls",net.task_cls)
        
    print("***********************")
    utils.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices,
                                                                 **appr_exemplars_dataset_args.__dict__)
    utils.seed_everything(seed=args.seed)
    
    appr = Appr(net, device, **appr_kwargs)
    
    # GridSearch
    if args.gridsearch_tasks > 0:
        ft_kwargs = {**base_kwargs, **dict(logger=logger,
                                           exemplars_dataset=GridSearch_ExemplarsDataset(transform, class_indices))}
        appr_ft = Appr_finetuning(net, device, **ft_kwargs)
        gridsearch = GridSearch(appr_ft, args.seed, gs_args.gridsearch_config, gs_args.gridsearch_acc_drop_thr,
                                gs_args.gridsearch_hparam_decay, gs_args.gridsearch_max_num_searches)

    # Loop tasks
    print('taskcla : ', taskcla)
    dataset_size_arr = []
    for i in range(len(trn_loader)):
        dataset_size_arr.append(len(trn_loader[i])*args.batch_size)
    print('num tr dataset : ', dataset_size_arr)
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    forg_1 = np.zeros((max_task, max_task))
    forg_2 = np.zeros((max_task, max_task))
    forg_3 = np.zeros((max_task, max_task))
    forg_4 = np.zeros((max_task, max_task))
    
    
    for t, (_, ncla) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue
            
        if args.tbbn:
            for name, layer in appr.model.named_modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.set_number_of_task(t)
                    
        if 'cifar100' in args.datasets  or 'cifar100_icarl' in args.datasets:
            appr.set_schedule([80, 120])
        elif 'svhn' in args.datasets:
            appr.set_schedule([10, 20])
        elif 'imagenet100' in args.datasets or len(args.datasets) > 1:
            appr.set_schedule([40, 80])
        elif 'imagenet_32_reduced' in args.datasets:
            appr.set_hyperparameters(t, num_epochs=args.nepochs, lr=args.lr, schedule = [40, 80], ep_factor = 4)
        elif 'imagenet1000' in args.datasets:
            appr.set_hyperparameters(t, num_epochs=args.nepochs, lr=args.lr, schedule = [40, 80], ep_factor = 4, decay_lr = True)
            
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Add head for current task
        if torch.cuda.device_count() > 1:
            net.module.add_head(taskcla[t][1])
        else:
            net.add_head(taskcla[t][1])
                

        net.to(device)
        # GridSearch
        if t < args.gridsearch_tasks:

            # Search for best finetuning learning rate -- Maximal Plasticity Search
            print('LR GridSearch')
            best_ft_acc, best_ft_lr = gridsearch.search_lr(appr.model, t, trn_loader[t], val_loader[t])
            # Apply to approach
            appr.lr = best_ft_lr
            gen_params = gridsearch.gs_config.get_params('general')
            for k, v in gen_params.items():
                if not isinstance(v, list):
                    setattr(appr, k, v)

            # Search for best forgetting/intransigence tradeoff -- Stability Decay
            print('Trade-off GridSearch')
            best_tradeoff, tradeoff_name = gridsearch.search_tradeoff(args.approach, appr,
                                                                      t, trn_loader[t], val_loader[t], best_ft_acc)
            # Apply to approach
            if tradeoff_name is not None:
                setattr(appr, tradeoff_name, best_tradeoff)

            print('-' * 108)

        
        """
        if args.log_bn and t == 0:
            # Save initial Gamma, beta, running_mean, running_var
            appr.model.save_bn_parameters()
            appr.model.log_bn_parameters(full_exp_name)
        """
        
        if args.bn_momentum == -1:

            max_iteration = len(trn_loader[t])

            for name, layer in appr.model.named_modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.init_num_batches_tracked()
                    layer.set_max_iteration(max_iteration)
        
        appr.train(t, trn_loader[t], val_loader[t])

        if args.log_bn:
            # Save Gamma, beta, running_mean, running_var after each task trained
            appr.model.save_bn_parameters()
            appr.model.log_bn_parameters(full_exp_name)
            


        print("Training Finished")
        print('-' * 108)
            
        # Test
        for u in range(t + 1):
            test_loss, acc_taw[t, u], acc_tag[t, u], forg_1[t, u], forg_2[t, u], forg_3[t, u], forg_4[t, u]  = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))
            logger.log_scalar(task=t, iter=u, name='loss', group='test', value=test_loss)
            logger.log_scalar(task=t, iter=u, name='acc_taw', group='test', value=100 * acc_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag', group='test', value=100 * acc_tag[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_taw', group='test', value=100 * forg_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag', group='test', value=100 * forg_tag[t, u])

        # Save
        print('Save at ' + os.path.join(args.results_path, 'FINAL_' +full_exp_name))
        logger.log_result(acc_taw, name="acc_taw", step=t)
        logger.log_result(acc_tag, name="acc_tag", step=t)
        logger.log_result(forg_taw, name="forg_taw", step=t)
        logger.log_result(forg_tag, name="forg_tag", step=t)
        logger.log_result(forg_1, name="forg_1", step=t)
        logger.log_result(forg_2, name="forg_2", step=t)
        logger.log_result(forg_3, name="forg_3", step=t)
        logger.log_result(forg_4, name="forg_4", step=t)
        if torch.cuda.device_count() > 1:
            logger.save_model(net.module.state_dict(), task=t)
        else:
            logger.save_model(net.state_dict(), task=t)
        logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), name="avg_accs_taw", step=t)
        logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), name="avg_accs_tag", step=t)
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name="wavg_accs_taw", step=t)
        logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name="wavg_accs_tag", step=t)
        
        
        savemat('../results/' + full_exp_name + '.mat', {"acc_taw": acc_taw,"acc_tag": acc_tag,"forg_taw": forg_taw,"forg_tag": forg_tag,"fm": np.mean(forg_tag[-1]),"la": np.sum(acc_tag.diagonal()),})

        # Last layer analysis
        if args.last_layer_analysis:
            if torch.cuda.device_count() > 1:
                weights, biases = last_layer_analysis(net.module.heads, t, taskcla, y_lim=True)
            else:
                weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)

            # Output sorted weights and biases
            if torch.cuda.device_count() > 1:
                weights, biases = last_layer_analysis(net.module.heads, t, taskcla, y_lim=True, sort_weights=True)
            else:
                weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True, sort_weights=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)
        
    # Print Summary
    utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()
