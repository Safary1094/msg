import matplotlib.pyplot as plt
import random
import os
from skimage.draw import line_aa, line_nd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path_to_save = './results/'


def num_results():
    dirs = os.listdir(path_to_save)
    num = len(dirs)
    name_folder = path_to_save + 'result_%d' % num
    while os.path.exists(name_folder):
        num += 1
        name_folder = path_to_save + 'result_%d' % num

    if not os.path.exists(name_folder):
        os.makedirs(name_folder)
    return name_folder


name_folder = num_results()


def image_pred_position(heatmaps, image, coord_3D=None):
    im = np.copy(image)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)

    ax2 = fig.add_subplot(122, projection='3d')

    coord_list = []

    lines_list = [[(0, 1), (1, 2), (2, 3), (3, 4)],
                  [(0, 5), (5, 6), (6, 7), (7, 8), ],
                  [(0, 9), (9, 10), (10, 11), (11, 12)],
                  [(0, 13), (13, 14), (14, 15), (15, 16)],
                  [(0, 17), (17, 18), (18, 19), (19, 20)]]

    color_list = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
    cmap_list = ['red', 'green', 'blue', 'orange', 'purple']

    for i, heatmap in enumerate(heatmaps):
        y = heatmap.argmax() // 64
        x = heatmap.argmax() % 64
        r = 3
        y = y * 4
        x = x * 4
        im[0, y - r:y + r, x - r:x + r] = 1
        im[1, y - r:y + r, x - r:x + r] = 0
        im[2, y - r:y + r, x - r:x + r] = 0
        coord_list.append((x, y))
        # print("%d: (%d; %d)" % (i, x, y))

    for color, lines in zip(color_list, lines_list):
        for line in lines:
            x0, y0 = coord_list[line[0]][0], coord_list[line[0]][1]
            x1, y1 = coord_list[line[1]][0], coord_list[line[1]][1]
            lin = line_nd((y0, x0), (y1, x1))
            im[lin[0], lin[1], 0] = color[0]
            im[lin[0], lin[1], 1] = color[1]
            im[lin[0], lin[1], 2] = color[2]

    ax1.imshow(im)

    if coord_3D is not None:
        for color, lines in zip(cmap_list, lines_list):
            for line in lines:
                x0, y0, z0 = coord_3D[line[0]][0], coord_3D[line[0]][1], coord_3D[line[0]][2]
                x1, y1, z1 = coord_3D[line[1]][0], coord_3D[line[1]][1], coord_3D[line[1]][2]
                z_line = np.linspace(z0, z1, 50)
                x_line = np.linspace(x0, x1, 50)
                y_line = np.linspace(y0, y1, 50)
                ax2.scatter3D(x0, y0, z0, c=color)
                ax2.scatter3D(x1, y1, z1, c=color)
                ax2.plot3D(x_line, y_line, z_line, color)

    dirs = os.listdir(name_folder)
    num_epoch = len(dirs) // 2
    if len(dirs) % 2:
        num_epoch += 1

    save_path = name_folder + '/' + 'predict_' + str(num_epoch) + '.png'
    plt.savefig(save_path)


def image_heatmap(heatmaps, gt_heatmaps, image, coord_3D=None):
    heatmaps = heatmaps.cpu().detach().numpy()
    gt_heatmaps = gt_heatmaps.cpu().detach().numpy()
    image = image.cpu().detach().numpy()
    coord_3D = coord_3D.cpu().detach().numpy() if coord_3D is not None else None

    image_pred_position(heatmaps, image, coord_3D)

    number_feature = [0, 4, 12, 16]
    fig, ax = plt.subplots(len(number_feature), 3, dpi=200)
    ax = ax.reshape(len(number_feature), 3)

    for i, f_idx in enumerate(number_feature):
        ax[i][0].imshow(heatmaps[f_idx])
        ax[i][1].imshow(gt_heatmaps[f_idx])

        im = np.copy(image)
        y = gt_heatmaps[f_idx].argmax() // 64
        x = gt_heatmaps[f_idx].argmax() % 64
        r = 4
        y = y * 4
        x = x * 4
        im[y - r:y + r, x - r:x + r, 0] = 1
        im[y - r:y + r, x - r:x + r, 1] = 0
        im[y - r:y + r, x - r:x + r, 2] = 0
        ax[i][2].imshow(im)

    plt.tight_layout()
    folder_to_save = name_folder + '/'
    dirs = os.listdir(folder_to_save)
    num_epoch = len(dirs) // 2
    if len(dirs) % 2:
        num_epoch += 1
    num_epoch += 1
    plt.savefig(folder_to_save + str(num_epoch) + '.png')
    plt.close(fig)
