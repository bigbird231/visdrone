import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# transfer absolute pixel length to relative length
# x=x/width, h=h/height
# return: [x1, y1, x2, y2, main_class, sub_class], x-y all normalized
def normalize_annotations(annotations, img_width, img_height):
    norm_annotations = annotations.clone().float()
    # (x1, y1, width, height) to [x1, y1, x2, y2]
    norm_annotations[:, 2] += norm_annotations[:, 0]
    norm_annotations[:, 3] += norm_annotations[:, 1]

    length = annotations.shape[-1]
    template = [img_width, img_height, img_width, img_height]
    while len(template) < length:
        template.append(1)
    norm_factors = torch.tensor(template, dtype=torch.float32)
    norm_annotations /= norm_factors
    # return: [x1, y1, x2, y2, main_class, sub_class], x-y all normalized
    return norm_annotations


# [centerX,centerY, w,h] to [xmin,ymin, xmax,ymax]
def box_center_to_corner(boxes):
    return torch.cat((
        boxes[:, :2] - 0.5 * boxes[:, 2:],
        boxes[:, :2] + 0.5 * boxes[:, 2:]), dim=1)


# [xmin,ymin, xmax,ymax] to [centerX,centerY, w,h]
def box_corner_to_center(boxes):
    c = (boxes[:, 2:] + boxes[:, :2]) / 2
    wh = boxes[:, 2:] - boxes[:, :2]
    return torch.cat((c, wh), axis=1)


# convert offsets back to boxes
def offset_inverse(anchors, offset_preds):
    anc = box_corner_to_center(anchors)
    c_pred = torch.cat((
        offset_preds[:, :2] * anc[:, 2:] / 10 + anc[:, :2],
        torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]), dim=1)
    return box_center_to_corner(c_pred)


# Non-Maximum Suppression
def nms(boxes, scores, iou_threshold):
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = calc_box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


#
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold, non-background
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)


# ['s=0.5, r=1', 's=0.25, r=1', 's=0.5, r=2']
# based on scales and ratios
def get_scales_ratios_desc(scales, ratios):
    labels = []
    for i, ratio in enumerate(ratios):
        if i == 0:
            for scale in scales:
                labels.append(f's={scale}, r={ratio}')
        else:
            labels.append(f's={scales[0]}, r={ratio}')
    return labels


# generate anchor boxes on feature map
# @data: [height, width]
# @scales: [0.75, 0.5, 0.25]. shrink rate from origin image size
# @ratios: [1, 2, 0.5]. width/height
# for w*h=960*540 image, return anchor boxes: [30*16,4]
def multibox_prior(data, scales, ratios):
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(scales), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(scales, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # offset to the center of cells
    offset_h, offset_w = 0.5, 0.5
    # y
    steps_h = 1.0 / in_height
    # x
    steps_w = 1.0 / in_width

    # centers, normalized
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # w*h=size*size, w/h=ratio, all normalized.
    # w = w * in_height / in_width, this is to make sure steps in w and h are the same
    # eg: image is 960*540, scale=0.02, w*h=0.02*0.02*540/960=0.0002
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), scales[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), scales[0] / torch.sqrt(ratio_tensor[1:])))
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    # [xmin, ymin, xmax, ymax], shape=[in_height*in_width, 4]
    output = out_grid + anchor_manipulations
    # for w*h=960*540 image, return anchor boxes: [30*16,4]
    return output.unsqueeze(0)


# width * height
def calc_box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# calculate iou, Intersection of Union
def calc_box_iou(boxes1, boxes2):
    areas1 = calc_box_area(boxes1)
    areas2 = calc_box_area(boxes2)
    # top_left of the intersection area
    top_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # bottom_right of the intersection area
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (bottom_right - top_left).clamp(min=0)
    # inter_areas and union_areas:(boxes1 number,boxes2 number)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


# assign the closest true bounding box to the anchor box
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.1):
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # [480,88]: [anchor_boxes_amount, annotations_amount], only have intersection, the value won't be 0, others are all 0
    box_iou = calc_box_iou(anchors, ground_truth)
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    # decide weather
    max_ious, indices = torch.max(box_iou, dim=1)
    # remain the row. torch.nonzero returns index
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    # which column anc_i is
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        # ensure every label match one anchor box, even it's less than threshold
        max_idx = torch.argmax(box_iou)
        # index in annotations
        box_idx = (max_idx % num_gt_boxes).long()
        # index in anchor box rows
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        box_iou[:, box_idx] = col_discard
        box_iou[anc_idx, :] = row_discard
    # row=index in anchor box rows, column=index in annotations. [480,]
    return anchors_bbox_map


# transfer anchor box offset
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    # [xmin,ymin, xmax,ymax] to [centerX,centerY, w,h]
    c_anc = box_corner_to_center(anchors)
    # if no match, they are [0,0,0,0], still calculates
    c_assigned_bb = box_corner_to_center(assigned_bb)
    # they are not plain distance. x: shift how many units of anchor box width; y: shift how many units of anchor box height
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    # hard to explain, but they are log transferred expand and shrink rates
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    # [shift of centerX, shift of centerY, expand rate of width, expand rate of height], , [480,4]
    return offset


# ground truth annotation notate anchor box
# @anchors:[1, in_height*in_width, 4], 4:[xmin,ymin, xmax,ymax]
# @labels:[[label_amoun, 6]], 6:[xmin,ymin, xmax,ymax, main_class,sub_class]
def multibox_target(anchors, labels):
    batch_size, anchors = len(labels), anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 0:4], anchors, device)
        # >=0: 1; else: 0, [480,4]. only matched rows would be [1,1,1,1]
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        # set background anchor box to 0
        # matched rows in anchor boxes
        indices_true = torch.nonzero(anchors_bbox_map >= 0).squeeze(-1)
        # indices_true = torch.nonzero(anchors_bbox_map >= 0)
        # index in annotations
        bb_idx = anchors_bbox_map[indices_true]
        # main_class + 1
        class_labels[indices_true] = label[bb_idx, 4].long() + 1
        # [xmin,ymin, xmax,ymax] from matched annotations
        assigned_bb[indices_true] = label[bb_idx, 0:4]
        # offset, [shift of centerX, shift of centerY, expand rate of width, expand rate of height], [480,4]
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        # bbox_mask remove non-matched anchor boxes
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    # [1, 1920] = [1, 480*4,], 4: [shift of centerX, shift of centerY, expand rate of width, expand rate of height]
    bbox_offset = torch.stack(batch_offset)
    # [1, 1920] = [1, 480*4,], only matched row would be [1,1,1,1], non-matched would be [0,0,0,0]
    bbox_mask = torch.stack(batch_mask)
    # [1, 480], main_class+1, non-matched would be 0
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


# return pixel value: [x1, y1, x2, y2]
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
            feature7, feature8 = map(int, (parts[6:8]))
            boxes.append([x, y, w, h, main_class_id, sub_class_id, feature7, feature8])
    return torch.tensor(boxes)


# ================ Run the demo ================
def test():
    path = '../../task1/trainset'
    file_name = '0000002_00005_d_0000014'
    image_name = file_name + '.jpg'
    image_path = os.path.join(path + '/images', image_name)
    image = Image.open(image_path)
    width, height = image.size
    print(width, height)

    # generate predicted boxes
    shrink_ratio = 48
    X = torch.rand(size=(1, 3, height // shrink_ratio, width // shrink_ratio))
    # eg:0.02*0.02* 300*300=36 pixels
    scales = [0.06]
    # eg:w/h=1
    ratios = [1, 2, 0.3]
    anchor_boxes = multibox_prior(X, scales=scales, ratios=ratios)
    boxes_per_pixel = len(scales) + len(ratios) - 1

    # get annotated anchor_boxes / get labels
    annotation_name = file_name + '.txt'
    annotation_path = os.path.join(path + '/annotations', annotation_name)
    annotated_boxes = load_annotations(annotation_path)
    normed_annotated_boxes = normalize_annotations(annotated_boxes, width, height)
    # simulate 1 batch
    labels = [normed_annotated_boxes]

    # calculate: offset and expand rate, matched rows, main_class
    (bbox_offset, bbox_mask, class_labels) = multibox_target(anchor_boxes, labels)

    # decode positive predictions
    bbox_offset = bbox_offset[0]
    bbox_mask = bbox_mask[0]
    class_labels = class_labels[0]
    # remove non-matched
    offset_preds = bbox_offset.reshape(-1, 4) * bbox_mask.reshape(-1, 4)
    # index of matched
    positive_indices = class_labels > 0

    # apply offset on anchor boxes
    # pred_boxes = anchor_boxes.squeeze(0)
    # pred_boxes = anchor_boxes.squeeze(0)[positive_indices]
    pred_boxes = offset_inverse(anchor_boxes.squeeze(0), offset_preds)[positive_indices]

    # normalized to real pixels
    pred_boxes_pixel = pred_boxes * torch.tensor([width, height, width, height])

    # plot
    dpi = 150
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.imshow(image, interpolation='none')
    for bbox in pred_boxes_pixel:
        bbox = bbox.detach().numpy()
        rect = patches.Rectangle(
            # [x1, y1, x2, y2] to (x1, y1, width, height)
            xy=(bbox[0], bbox[1]),
            width=bbox[2] - bbox[0],
            height=bbox[3] - bbox[1],
            fill=False,
            edgecolor='cyan',
            linewidth=0.5
        )
        ax.add_patch(rect)
    plt.show()

# test()