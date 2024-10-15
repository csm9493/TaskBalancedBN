
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MoCoLoss(nn.Module):
    
    def __init__(self):
        super(MoCoLoss, self).__init__()
        
        self.dim = 128
        self.K = 4096
        self.T = 0.1
        
        # create the queue
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0).cuda()
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        
    def forward(self, q, k):

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        
        #print("q.shape",q.shape)
        #print("k.shape",k.shape)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # negative logits: NxK

        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])


        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)


        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)
        
        self._dequeue_and_enqueue(k)

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SwavLoss(nn.Module):
    
    def __init__(self):
        super(SwavLoss, self).__init__()

        
    def forward(self, f1, f2, outputs, queue, use_the_queue, model):
        
        bs = f1.shape[0]
        #print("f1.shape",f1.shape) # [64, 128]
        #print("f2.shape",f2.shape) # [64, 128]
        #print("f1.unsqueeze(1).shape",f1.unsqueeze(1).shape) #[64, 1, 128]
        #print("f2.unsqueeze(1).shape",f2.unsqueeze(1).shape)
        #features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1).detach()
        features = torch.cat([f1, f2], dim=0).detach() # [128, 128]
        #print("features.shape", features.shape)
        loss = 0
        for i, crop_id in enumerate([0]):
            with torch.no_grad():
                out = outputs[bs * crop_id: bs * (crop_id + 1)].detach()
                
                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            model.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    #print("features.shape",features.shape) # [1, 3840, 128]
                    #print("queue.shape",queue.shape) # [128, 128]
                    queue[i, :bs] = features[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = distributed_sinkhorn(out)[-bs:]
                
            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum([2])), crop_id):
                x = outputs[bs * v: bs * (v + 1)] / 0.1 # temperature = 0.1
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum([2]) - 1)
        loss /= 2
                

        return loss

    
@torch.no_grad()
def distributed_sinkhorn(out):
    #Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    Q = torch.exp(out / 0.05).t()
    #B = Q.shape[1] * args.world_size # number of samples to assign
    B = Q.shape[1] * 1
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    #print("Q.shape", Q.shape) # [3000, 64]
    sum_Q = torch.sum(Q)
    #print("sum_Q.shape",sum_Q.shape) # []
    #print("sum_Q", sum_Q) # tensor(893824.4375,device='cuda:0')
    #dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(3):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        #dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()