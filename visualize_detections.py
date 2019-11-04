import matplotlib.pyplot as plt
from PIL import Image
import os


images_root_dir = '../ch4_test_images'

colors = {
    'gt': (255, 255, 255),
    'psenet': (0, 255, 0),
    'craft': (255, 0, 0),
    'nms': (0, 0, 255)
    }

def visualize(img_key, confs, polygons, method, img=None):
    if img is None:
        img = Image.open(os.path.join(images_root_dir, 'img_' + img_key + '.jpg'))
    plt.imshow(img)
    plt.axis('off')
    color = colors[method]
    color = (color[0] / 255., color[1] / 255., color[2] / 255.)
    for i in range(len(confs)):
        conf = confs[i]
        polygon = polygons[i][0]
        rect = plt.Polygon(polygon, fill=False,
                             edgecolor=color,
                             linewidth=2.0)
        plt.gca().add_patch(rect)
        if method != 'gt':
            plt.gca().text(polygon[0][0], polygon[0][1],
                           '{:.2f} | {:s}'.format(conf, method),
                           bbox=dict(facecolor=color, alpha=0.5),
                           fontsize=12, color='white')
    if method == 'nms':
        plt.show()
    return img