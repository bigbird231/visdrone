import time
#
import torch
from torch import nn
from torch.utils.data import DataLoader
#
from torchvision import transforms
#
import matplotlib.pyplot as plt
#
from dataset import VisdroneDataset
from anchor import multibox_target
from validate import validate
from model import TinySSD


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    def __init__(self, xlabel=None, xlim=None, legend=None):
        self.fig, self.ax = plt.subplots()
        self.xlabel = xlabel
        self.xlim = xlim
        self.legend = legend
        self.X, self.Y = [], [[] for _ in legend]
        self.lines = [self.ax.plot([], [], label=legend[i])[0] for i in range(len(legend))]
        self.ax.set_xlabel(xlabel)
        self.ax.set_xlim(*xlim)
        self.ax.legend()

    def add(self, x, y):
        self.X.append(x)
        for i, val in enumerate(y):
            self.Y[i].append(val)
            self.lines[i].set_data(self.X, self.Y[i])
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.001)

def train():
    batch_size = 2
    num_epochs = 5


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TinySSD(num_classes=10).to(device)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = VisdroneDataset('../../task1/trainset', transform=transform)
    train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: (
        # batch size
        torch.stack([b[0] for b in batch]),
        # annotation objects amount
        [b[1] for b in batch]
    ))

    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')

    def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
        cls = cls_loss(cls_preds.reshape(-1, num_classes),
                       cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        bbox = bbox_loss(bbox_preds * bbox_masks,
                         bbox_labels * bbox_masks).mean(dim=1)
        return cls + bbox

    def cls_eval(cls_preds, cls_labels):
        return float((cls_preds.argmax(dim=-1).type(
            cls_labels.dtype) == cls_labels).sum())

    def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
        return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
    net = net.to(device)
    start_time = 0
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        net.train()
        for features, target in train_iter:
            start_time = time.time()
            trainer.zero_grad()
            X = features.to(device)
            Y = [label.to(device) for label in target]
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(), bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        animator.add(epoch + 1, (cls_err, bbox_mae))
        time.sleep(6)
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    duration_time = time.time() - start_time
    print(f'{len(train_iter.dataset) / duration_time:.1f} examples/sec on '
          f'{str(device)}')
    return net, device


net, device = train()
validate(net, device)
