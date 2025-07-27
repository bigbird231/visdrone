#
import torch
from torch import nn
#
from anchor import multibox_prior


scales = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
block_num=len(scales)
ratios = [[1, 2, 0.3]] * block_num
num_anchors = len(scales[0]) + len(ratios[0]) - 1


# predicts the class of each anchor box
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


# predicts the offset for each anchor box
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


# [a,b,c,d] -> [a,c,d,b] -> [a, c*d*b]
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


# concatenate all feature maps
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


# reduces spatial size and increases channel depth
# 2 convolution layers + BatchNorm + ReLU + MaxPool(2)
def down_sample_blk(in_channels, out_channels):
    cnn_layers=2
    blk = []
    for _ in range(cnn_layers):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(kernel_size=2))
    return nn.Sequential(*blk)


# 3 -> 16 -> 32 -> 64
def backbone_basenet():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


# block
def get_blk(i):
    if i == 0:
        blk = backbone_basenet()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


# feature map -> anchor box, class prediction, offset prediction
def blk_forward(X, blk, scale, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = multibox_prior(Y, scales=scale, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        # 5 blocks, block means different scale, like tiny -> huge
        for i in range(block_num):
            # feature map
            setattr(self, f'blk_{i}', get_blk(i))
            # class
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            # offset
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * block_num, [None] * block_num, [None] * block_num
        for i in range(block_num):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X,
                                                                     getattr(self, f'blk_{i}'),
                                                                     scales[i],
                                                                     ratios[i],
                                                                     getattr(self, f'cls_{i}'),
                                                                     getattr(self, f'bbox_{i}')
                                                                     )
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
