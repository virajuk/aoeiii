import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from util import _visualization_utils as vis_utils

def show_image(img):
    """
    Matplotlib based plotting of PIL Image instance
    Args:
        img:    PIL Image Instance

    Returns:
        None. Shows a matplotlib plot.
    """
    arr = np.array(img)
    plt.imshow(arr)

def compare_image_mask(datum):
    """
    Matplotlib based plotting of PIL Image instance
    Args:
        datum:    datum containing image and mask property

    Returns:
        None. Shows a matplotlib plot.
    """
    image = np.array(datum.image)
    mask = np.array(datum.mask)
    plt.figure(figsize=(10, 7))
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(image)
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(mask)


def image_with_labels(img, boxes, masks, labels=None, color='blue', label_dict=None, show=True):
    """
    Plots image with mask and boxes
    Args:
        img:            image as np.array
        boxes:          np.array of shape [num, 4]
        masks:          np.array of shape [num, h, w]
        labels:         list of labels
        color:          str
        label_dict:     dict mapping class ids (int) to class names (str)
        show:           show or not

    Returns:
       None. Generates matplotlib plot.

    """

    img = Image.fromarray(img)
    img_boxes = np.array(image_with_boxes(img, boxes, labels, color=color, label_dict=label_dict))

    plt.imshow(img_boxes)
    plt.show()

    # plt.figure(figsize=(10, 7))
    # plt.subplot(121)
    # plt.axis('off')
    # plt.imshow(img_boxes)
    # plt.subplot(122)
    # plt.axis('off')
    # plt.imshow(img_boxes)
    # plt.title("masks in colour")
    # plt.show()

    # if len(masks):
    #     masks = np.concatenate([np.zeros((1, *masks.shape[1:])), masks])
    #     masks = np.argmax(np.array(masks), axis=0)
    #     plt.imshow(masks, alpha=0.5)
    # if show:
    #     plt.show()


def image_with_boxes(img, boxes, labels=None, scores=None, color='blue', label_dict=None, score_tresh=0.5):
    """
    Draw on image in place
    Args:
        img:           PIL Image instance
        boxes:         np.array of shape [num, 4]
        labels:        list of labels
        scores:        list of scores
        color:         str
        label_dict:    dict mapping class ids (int) to class names (str)
        score_tresh:   score threshold for plotting

    Returns:
        PIL Image instance
    """
    if labels is None:
        labels = [" "] * len(boxes)
    else:
        labels = [label_dict[i] for i in labels]

    labels = ['explorer']

    if scores is None:
        scores = [1] * len(boxes)

    for box, label, score in zip(boxes, list(labels), scores):
        if score < score_tresh:
            continue

        vis_utils.draw_bounding_box_on_image(img, box[0], box[1], box[2], box[3], display_str_list=[label],
                                             color=color, font_size=40)

    return img


def resize_mask(masks, bboxes, img_h, img_w):
    """

    Args:
        masks:
        bboxes:     np.array of shape [num, 4] containing ymin, xmin, ymax, xmax
        img_h:
        img_w:

    Returns:
        List of different sized masks
    """
    out_masks = []
    for mask, bbox in zip(masks, bboxes):
        b_h, b_w = int((bbox[2] - bbox[0]) * img_h), int((bbox[3] - bbox[1]) * img_w)
        mask_r = Image.fromarray(mask)
        mask_r = mask_r.resize(size=(b_h, b_w), resample=Image.NEAREST)
        mask_r = np.array(mask_r)
        out_masks.append(mask_r)

    return out_masks
