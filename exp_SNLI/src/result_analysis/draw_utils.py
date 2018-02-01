import cv2, os
import numpy as np
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

font = {'family' : 'Arial',
        'color'  : 'k',
        'weight' : 'normal',
        'size'   : 12,
        }


def save_heat_image(dir, file_name, mat, reverse=False, box_h=10, box_w=10, color='b', min_max=None):
    path = os.path.join(dir, file_name + '.jpg')
    color_img = draw_heat_mat(mat, reverse, box_h, box_w, color, min_max)
    cv2.imwrite(path, color_img, (cv2.IMWRITE_JPEG_QUALITY, 50))



def draw_heat_mat(mat, reverse=False, box_h=10, box_w=10, color='b', min_max=None):
    """

    :param mat:
    :param reverse:
    :return:
    """
    normalized_img = normalize_img(mat, reverse, min_max)
    box_img = convert_to_boxes(normalized_img, box_h, box_w)
    box_img_color = add_color_to_mat(box_img, color)
    return box_img_color


def normalize_img(mat, reverse=False, min_max=None):
    mat_f = mat.astype(np.float32)

    if min_max is None:
        min_value = np.min(mat)
        max_value = np.max(mat)
    else:
        min_value = min_max[0] or np.min(mat)
        max_value = min_max[1] or np.max(mat)

    # normalization to [0,1]
    mat_normal_f = (mat_f - min_value) / (max_value - min_value)
    mat_normal =(mat_normal_f * 255).astype(np.uint8)

    if not reverse:
        return mat_normal
    else:
        return 255 - mat_normal


def convert_to_boxes(mat, box_h=10, box_w=10):
    assert len(mat.shape) == 2
    new_shape = [mat.shape[0] * box_h, mat.shape[1] * box_w]
    new_mat = np.zeros(new_shape, mat.dtype)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            little_box = np.ones([box_h, box_w],mat.dtype) * mat[r, c]
            new_mat[r*box_h:(r+1)*box_h,c*box_w:(c+1)*box_w] = little_box
    return new_mat


def add_color_to_mat(mat, color='b'):
    other_mat = (np.power(mat.astype(np.float32)/255.0, 2) * 255.0).astype(mat.dtype)

    channels = [other_mat, other_mat, other_mat]
    if color == 'b':
        channels[0] = mat
    elif color == 'g':
        channels[1] = mat
    elif color == 'r':
        channels[2] = mat
    elif color == None:
        channels= [mat,mat,mat]
    else:
        raise RuntimeError('no color named as \'%s\'' % color)

    return np.stack(channels, -1)



# ----------------------
def mat_softmax(mat):
    # [sl,v]
    mat_exp = np.exp(mat)
    return mat_exp/np.sum(mat_exp,0,keepdims=True)


# ------------
def save_imgs_dict(path, data_dict):
    import pickle
    with open(path, 'wb') as file:
        pickle.dump(data_dict, file)


# ------ plot with matplotlib


def save_heatmap_image_svg(dir, file_name, mat, min_max=None, labels=([], []), square=False, axis_reverse=False):
    path = os.path.join(dir, file_name + '.svg')
    min_max = min_max or (None, None)

    # ----- draw -----

    ax = sns.heatmap(mat, cmap=plt.cm.Blues, linewidths=.5, cbar=False, square=square,
                     vmin=min_max[0], vmax=min_max[1],
                     xticklabels=labels[0], yticklabels=labels[1])
    if axis_reverse:
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('right')

    plt.xticks(fontsize=15, rotation=25)
    plt.yticks(fontsize=15, rotation=25)


    # ----------------

    # ---- save -----
    plt.savefig(path)








