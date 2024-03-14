import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from metrics import dice_coef
import scipy.ndimage

BCE = torch.nn.BCELoss()

import torch
import torch.nn.functional as F

def dynamic_scale_factor(mask, base_scale=5, factor=0.1):
    # Adjust scale based on the variance of the mask
    variance = torch.var(mask)
    return base_scale + factor * variance

def select_kernel_size(mask, low_var_thresh, high_var_thresh, small_kernel, large_kernel):
    variance = torch.var(mask)
    if variance < low_var_thresh:
        return small_kernel
    elif variance > high_var_thresh:
        return large_kernel
    else:
        return (small_kernel + large_kernel) // 2  # 或者选择一个中间值

def weighted_loss(pred, mask, low_var_thresh=0.05, high_var_thresh=0.2, small_kernel=3, large_kernel=15, pooling_method='avg', base_scale=5, scale_factor=0.1):
    # Dynamic kernel size selection based on mask variance
    kernel_size = select_kernel_size(mask, low_var_thresh, high_var_thresh, small_kernel, large_kernel)

    # Dynamic scaling
    scale = dynamic_scale_factor(mask, base_scale, scale_factor)

    # Apply different pooling methods
    if pooling_method == 'avg':
        pooled_mask = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    elif pooling_method == 'max':
        pooled_mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    else:
        raise ValueError("Unsupported pooling method. Choose 'avg' or 'max'.")

    weit = 1 + scale * torch.abs(pooled_mask - mask)

    # Rest of the weighted_loss calculation
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def calc_loss(pred, target, bce_weight=0.5):
    bce = weighted_loss(pred, target)
    # dl = 1 - dice_coef(pred, target)
    # loss = bce * bce_weight + dl * bce_weight

    return bce


def loss_sup(*args):
    # 确保传入的参数数量是偶数
    if len(args) % 2 != 0:
        raise ValueError("Arguments should be in pairs of logit_S and labels_S")

    total_loss = 0
    for i in range(0, len(args), 2):
        logit_S = args[i]
        labels_S = args[i + 1]
        total_loss += calc_loss(logit_S, labels_S)

    return total_loss


def dynamic_weighting(pred):
    # Calculate dynamic weights based on the predictions' characteristics
    # Example: use the gradient magnitude as a weight
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(pred, sobel_x, padding=1)
    grad_y = F.conv2d(pred, sobel_y, padding=1)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

    return grad_magnitude

def loss_diff(u_prediction_1, u_prediction_2, batch_size, threshold=5):
    # Calculate the absolute difference between predictions
    diff = torch.abs(u_prediction_1 - u_prediction_2) > 0.5
    diff = diff.float()  # Convert to float for computations

    # Apply dynamic weighting based on the characteristics of the predictions
    weights_1 = dynamic_weighting(u_prediction_1)
    weights_2 = dynamic_weighting(u_prediction_2)
    weighted_diff = diff * (weights_1 + weights_2)

    # Identify continuous regions of difference
    structure = torch.ones((1, 1, threshold, threshold)).to(weighted_diff.device)
    continuous_diff_mask = F.conv2d(weighted_diff, structure, stride=1, padding=threshold // 2) == (threshold * threshold)
    continuous_diff_mask = continuous_diff_mask.float()

    # Apply weighted loss on continuous regions
    a = weighted_loss(u_prediction_1, Variable(u_prediction_2 * continuous_diff_mask, requires_grad=False))
    a = a.item()

    b = weighted_loss(u_prediction_2, Variable(u_prediction_1 * continuous_diff_mask, requires_grad=False))
    b = b.item()

    loss_diff_avg = (a + b)
    return loss_diff_avg / batch_size
