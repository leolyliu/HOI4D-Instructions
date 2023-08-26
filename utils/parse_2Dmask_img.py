import numpy as np
from color_map import get_color_map


def parse_2Dmask_img(mask_img, N=10):
    """
    mask_img: RGB image, shape = (H, W, 3)
    N: number of labels (including background)

    return: pixel labels, shape = (H, W)
    """

    color_map = get_color_map(N=N)

    H, W = mask_img.shape[:2]
    labels = np.zeros((H, W)).astype(np.uint8)

    for i in range(N):
        c = color_map[i]
        valid = (mask_img[..., 0] == c[0]) & (mask_img[..., 1] == c[1]) & (mask_img[..., 2] == c[2])
        labels[valid] = i
    
    return labels
