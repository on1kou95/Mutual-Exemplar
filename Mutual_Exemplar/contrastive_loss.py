# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:37:12 2022

@author: loua2
"""
import torch
from torch import nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

class ConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))
        l_pos = l_pos.view(-1, 1)

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

import torch
import torch.nn.functional as F

class contrastive_loss_sup(torch.nn.Module):
    def __init__(self, temperature=0.7, base_temperature=0.7):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(contrastive_loss_sup, self).__init__()
        # Make temperature a learnable parameter
        self.temperature = torch.nn.Parameter(torch.tensor(temperature))
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))
        l_pos = l_pos.view(-1, 1)
        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))

        return loss


class ContrastiveLossSup2(torch.nn.Module):
    def __init__(self, temperature=0.07, sparse_step=4):
        super(ContrastiveLossSup2, self).__init__()
        self.temperature = temperature
        self.sparse_step = sparse_step
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())

        # 计算稀疏特征的损失
        sparse_q = feat_q[:, :, ::self.sparse_step]
        sparse_k = feat_k[:, :, ::self.sparse_step]
        loss_sparse = self.compute_loss(sparse_q, sparse_k)

        # # 计算密集特征的损失
        # dense_q = feat_q
        # dense_k = feat_k
        # loss_dense = self.compute_loss(dense_q, dense_k)

        # 返回稀疏和密集损失的总和
        return loss_sparse

    def compute_loss(self, feat_q, feat_k):
        batch_size, dim = feat_q.shape[0], feat_q.shape[1]
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1).detach()

        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1)).view(-1, 1)

        feat_q = feat_q.reshape(batch_size, -1, dim)
        feat_k = feat_k.reshape(batch_size, -1, dim)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(l_neg_curbatch.size(1), device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, l_neg_curbatch.size(1))

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss


class ConLoss2(torch.nn.Module):
    def __init__(self, temperature=0.07, sparse_step=4):
        super(ConLoss2, self).__init__()
        self.temperature = temperature
        self.sparse_step = sparse_step
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())

        # 提取稀疏特征
        sparse_q = feat_q[:, :, ::self.sparse_step]
        sparse_k = feat_k[:, :, ::self.sparse_step]

        # 计算稀疏特征的损失
        loss_sparse = self.compute_loss(sparse_q, sparse_k)
        # loss_dense = self.compute_loss(feat_q, feat_k)

        # 返回总损失（稀疏和密集损失的平均）
        return loss_sparse
        # return (loss_sparse + loss_dense) / 2


    def compute_loss(self, feat_q, feat_k):
        batch_size, dim = feat_q.shape[0], feat_q.shape[1]
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1).detach()

        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1)).view(-1, 1)

        feat_q = feat_q.reshape(batch_size, -1, dim)
        feat_k = feat_k.reshape(batch_size, -1, dim)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(l_neg_curbatch.size(1), device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, l_neg_curbatch.size(1))

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss

class ConLoss3(torch.nn.Module):
    def __init__(self, temperature=0.07, sparse_step=4):
        super(ConLoss3, self).__init__()
        self.temperature = temperature
        self.sparse_step = sparse_step
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), "Feature dimensions do not match."

        # Extract sparse features
        sparse_q = feat_q[:, :, ::self.sparse_step]
        sparse_k = feat_k[:, :, ::self.sparse_step]

        # Calculate loss for sparse features
        loss_sparse = self.compute_loss(sparse_q, sparse_k)

        return loss_sparse

    def compute_loss(self, feat_q, feat_k):
        batch_size, dim, _ = feat_q.shape

        # Normalize features
        feat_q = F.normalize(feat_q.view(batch_size, dim, -1), p=2, dim=1)
        feat_k = F.normalize(feat_k.view(batch_size, dim, -1), p=2, dim=1)

        # Positive logit
        l_pos = torch.einsum('ncm,ncm->n', [feat_q, feat_k]).unsqueeze(-1)

        # Negative logits
        neg_mask = ~torch.eye(batch_size, dtype=self.mask_dtype, device=feat_q.device)
        l_neg = torch.einsum('ncm,kcm->nk', [feat_q, feat_k])

        # Mask out self-contrast cases
        l_neg = l_neg.masked_select(neg_mask).view(batch_size, -1)

        # Final logits: positive in front, negatives after
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        # Labels: positives are the 0-th
        labels = torch.zeros(batch_size, dtype=torch.long, device=feat_q.device)

        loss = self.cross_entropy_loss(logits, labels)
        return loss

class ContrastiveLoss_in(nn.Module):
    def __init__(self, temperature=0.7):
        super(ContrastiveLoss_in, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, positive1, positive2, negative1, negative2):
        # 归一化特征向量
        positive1 = F.normalize(positive1, dim=-1, p=2)
        positive2 = F.normalize(positive2, dim=-1, p=2)
        negative1 = F.normalize(negative1, dim=-1, p=2)
        negative2 = F.normalize(negative2, dim=-1, p=2)
        # print("positive1 shape:", positive1.shape)
        # print("negative1 shape:", negative1.shape).

        # 计算正例相似度
        pos_sim = torch.exp(torch.sum(positive1 * positive2, dim=-1) / self.temperature)

        # 计算负例相似度
        # neg_sim1 = torch.exp(torch.sum(positive1 * negative1, dim=-1) / self.temperature)

        # Resize negative1 to match the shape of positive1
        negative1_resized = F.interpolate(negative1, size=positive1.shape[2:], mode='bilinear', align_corners=False)

        # Now you can compute the similarity as before
        neg_sim1 = torch.exp(torch.sum(positive1 * negative1_resized, dim=-1) / self.temperature)
        # neg_sim2 = torch.exp(torch.sum(positive1 * negative2, dim=-1) / self.temperature)
        # Assuming negative2 has the shape [1, 32, 16, 16] or similar
        negative2_resized = F.interpolate(negative2, size=positive1.shape[2:], mode='bilinear', align_corners=False)

        # Then, you can compute neg_sim2 as before
        neg_sim2 = torch.exp(torch.sum(positive1 * negative2_resized, dim=-1) / self.temperature)

        # 损失计算
        loss = -torch.log(pos_sim / (pos_sim + neg_sim1 + neg_sim2))

        return loss.mean()
