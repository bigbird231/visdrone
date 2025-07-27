import torch
from torch.nn import functional as F
#
import torchvision
#
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#
from anchor import multibox_detection


def validate(net, device):
    X = torchvision.io.read_image('../../task1/trainset/images/0000002_00005_d_0000014.jpg').unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()

    def predict(X):
        net.eval()
        anchors, cls_preds, bbox_preds = net(X.to(device))
        cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
        output = multibox_detection(cls_probs, bbox_preds, anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
        return output[0, idx]

    output = predict(X)

    def show_bboxes(ax, bboxes, labels=None, colors=None):
        for i, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = [float(x) for x in bbox]
            width, height = xmax - xmin, ymax - ymin
            color = colors[i] if colors else 'red'
            rect = patches.Rectangle((xmin, ymin), width, height, fill=False, edgecolor=color, linewidth=0.8)
            ax.add_patch(rect)
            # if labels and labels[i]:
            #     ax.text(xmin, ymin, labels[i], bbox=dict(facecolor=color, alpha=0.5), fontsize=10, color='white')

    def display(img, output, threshold=0.5):
        height, width = img.shape[0:2]
        dpi = 150
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax.imshow(img, interpolation='none')

        bboxes = []
        labels = []

        for row in output:
            score = float(row[1])
            if score < threshold:
                continue
            bbox = row[2:6] * torch.tensor([width, height, width, height], device=row.device)
            bboxes.append(bbox)
            labels.append(f'{score:.2f}')

        show_bboxes(ax, bboxes, labels)
        plt.show()

    display(img, output.cpu(), threshold=0.9)
