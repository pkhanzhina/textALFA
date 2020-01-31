import matplotlib.pyplot as plt
from PIL import Image
import os


images_root_dir = '../ICDAR2015/ch4_test_images'

colors = {
    'gt': (255, 255, 255),
    'psenet': (0, 255, 0),
    'craft': (255, 0, 0),
    'charnet': (0, 0, 255),
    'nms': (255, 255, 0),
    'text_alfa': (255, 255, 0),
    'dontCare': (200, 200, 200)
    }

def visualize_polygons(img_key, confs, polygons, dontCare, method, img=None, show=False):
    if img is None:
        img = Image.open(os.path.join(images_root_dir, 'img_' + img_key + '.jpg'))
    plt.imshow(img)
    plt.axis('off')
    for i in range(len(confs)):
        conf = confs[i]
        polygon = polygons[i][0]
        color = colors[method]
        if dontCare[i]:
            color = colors['dontCare']
        color = (color[0] / 255., color[1] / 255., color[2] / 255.)
        rect = plt.Polygon(polygon, fill=False,
                             edgecolor=color,
                             linewidth=2.0)
        plt.gca().add_patch(rect)
        # if method != 'gt':
        #     plt.gca().text(polygon[0][0], polygon[0][1],
        #                    '{:.2f} | {:s}'.format(conf, method),
        #                    bbox=dict(facecolor=color, alpha=0.5),
        #                    fontsize=12, color='white')
    if show:
        plt.show()
        plt.clf()
    return img


def visualize_boxes_with_angles(img_key, confs, boxes, angles, dontCare, method, img=None, show=False):
    if img is None:
        img = Image.open(os.path.join(images_root_dir, 'img_' + img_key + '.jpg'))
    plt.imshow(img)
    plt.axis('off')
    for i in range(len(confs)):
        conf = confs[i]
        box = boxes[i]
        color = colors[method]
        if dontCare[i]:
            color = colors['dontCare']
        color = (color[0] / 255., color[1] / 255., color[2] / 255.)
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False,
                             edgecolor=color,
                             linewidth=2.0)
        plt.gca().add_patch(rect)
        if method == 'text_alfa':
            plt.gca().text(box[0], box[1],
                           '{:.2f} | {:s}'.format(conf, method),
                           bbox=dict(facecolor=color, alpha=0.5),
                           fontsize=12, color='white')
    if show:
        plt.show()
        plt.clf()
    return img