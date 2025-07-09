import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

box_color = ['white', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'darkgreen', 'black', 'purple', 'orange', 'crimson', 'chocolate']


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


def show_image_with_boxes(image_path, annotation_path):
    # load image
    # in matplotlib, (0,0) is at the top left.
    # x increase from left to right
    # y increases from top to bottom
    image = Image.open(image_path)
    width, height = image.size
    print(width, height)

    boxes = load_annotations(annotation_path)

    # plot in high resolution
    dpi = 150
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.imshow(image, interpolation='none')
    ax.imshow(image)

    # draw boxes, different color with different sub_class
    for box in boxes:
        x, y, w, h, main_class_id, sub_class_id = box
        rect = patches.Rectangle((x, y), w, h, linewidth=0.5, edgecolor=box_color[sub_class_id], facecolor='none')
        ax.add_patch(rect)
        # ax.text(x, y-5, f'Class {main_class_id}', color='red', fontsize=6)

    plt.axis('off')
    plt.show()


# loop through all images
# for filename in os.listdir('./images'):
#     if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
#         file_name = filename.split('.')[0]
# print(file_name)

path = './trainset'
file_name = '0000002_00005_d_0000014'
# file_name = '0000007_04999_d_0000036'

image_name = file_name + '.jpg'
annotation_name = file_name + '.txt'
image_path = os.path.join(path + '/images', image_name)
annotation_path = os.path.join(path + '/annotations', annotation_name)

show_image_with_boxes(image_path, annotation_path)
