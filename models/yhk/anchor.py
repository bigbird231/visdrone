import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def bbox_to_rect(bbox, color):
    # Convert bbox (xmin, ymin, xmax, ymax) to a matplotlib rectangle
    return patches.Rectangle(
        xy=(bbox[0], bbox[1]),
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        fill=False,
        edgecolor=color,
        linewidth=2
    )

def box_corner_to_center(boxes):
    c = (boxes[:, 2:] + boxes[:, :2]) / 2
    wh = boxes[:, 2:] - boxes[:, :2]
    return torch.cat((c, wh), axis=1)

def generate_anchor_labels(scales, ratios):
    labels = []
    for i, ratio in enumerate(ratios):
        if i == 0:
            for scale in scales:
                labels.append(f's={scale}, r={ratio}')
        else:
            labels.append(f's={scales[0]}, r={ratio}')
    return labels

# generate anchor boxes with different shapes centered on each pixel
# @data: [height, width]
# @sizes: [0.75, 0.5, 0.25]. shrink rate from origin image size
# @ratios: [1, 2, 0.5]. width/height
def multibox_prior(data, sizes, ratios):
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    offset_h, offset_w = 0.5, 0.5
    # y
    steps_h = 1.0 / in_height
    # x
    steps_w = 1.0 / in_width

    # all centers of anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # anchor box: (xmin,xmax,ymin,ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 处理矩形输入
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

# show all anchor boxes
def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        # if labels and len(labels) > i:
        #     text_color = 'k' if color == 'w' else 'w'
        #     axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))

# get IOU, Intersection of Union
def box_iou(boxes1, boxes2):
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,shape of areas2:
    # boxes1：(number of boxes1,4),
    # boxes2：(number of boxes2,4),
    # areas1：(number of boxes1,),
    # areas2：(number of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,shape of inters:
    # (number of boxes1, number of boxes2,2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas:(boxes1 number,boxes2 number)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

# assign the closest true bounding box to the anchor box
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # decide weather
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

# transfer anchor box offset
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

# ground truth annotation notate anchor box
def multibox_target(anchors, labels):
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # init to 0
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # set background anchor box to 0
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # offset
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

def load_annotations(annotation_path):
    boxes = []
    with open(annotation_path, 'r') as f:
        for line in f:
            # x, y, width, height, main_class, sub_class, ?, ?
            # in matplotlib, (0,0) is at the top left.
            # x increase from left to right
            # y increases from top to bottom
            parts = line.strip().split(',')
            x, y, w, h = map(int, parts[:4])
            main_class_id, sub_class_id = map(int, (parts[4:6]))
            boxes.append((x, y, w, h, main_class_id, sub_class_id))
    return boxes

# ================ Run the demo ================
path = '../../task1/trainset'
file_name = '0000002_00005_d_0000014'
annotation_name = file_name + '.txt'
annotation_path = os.path.join(path + '/annotations', annotation_name)
image_name = file_name + '.jpg'
image_path = os.path.join(path + '/images', image_name)
image = Image.open(image_path)
width, height = image.size
print(width, height)

X = torch.rand(size=(1, 3, height, width))
scales=[0.1, 0.05]
ratios=[1, 2]
Y = multibox_prior(X, sizes=scales, ratios=ratios)
boxes_num=len(scales)+len(ratios)-1
boxes = Y.reshape(height, width, boxes_num, 4)
# print(boxes)

bbox_scale = torch.tensor((width, height, width, height))
dpi = 150
fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
ax.imshow(image, interpolation='none')
ax.imshow(image)
show_bboxes(ax, boxes[250, 250, :, :] * bbox_scale, generate_anchor_labels(scales, ratios))
show_bboxes(ax, boxes[350, 350, :, :] * bbox_scale, generate_anchor_labels(scales, ratios))
plt.show()