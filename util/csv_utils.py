import json
import os

import numpy as np
import pandas as pd
import skimage.draw
from PIL import Image

from util.dataset_info import label_to_colour, parts_classes_v2_remap, parts_classes_v2


def convert_csv(base_path, csv_path, kind='damages', rank_filter=None, remap=True,
                skip_classes=("check", "no_damage"), close_up=False, binary_mask=True, reverse_dict=None):
    """

    Args:
        base_path:          path to root folder containing all images
        csv_path:           path to relevant csv
        kind:               damage or parts
        rank_filter:        1, 2, or 3 to filter by rank
        remap:              for parts class remap
        skip_classes:       tuple of classes to skip
        close_up:           generate close-up images
        binary_mask:        True / False
        reverse_dict:       dict for mapping class_names to int labels

    Returns:
        Generator. First element is len dataset, after that tf.example protos for each image.

    """

    df = pd.read_csv(csv_path)

    # filter by rank
    if rank_filter is not None:
        try:
            df["ranks"] = df["file_attributes"].apply(lambda x: int(json.loads(x)['rank']) if len(x) > 2 else 3)
            df = df.loc[df['ranks'] == rank_filter]
            df = df.drop(['ranks'], axis=1)
        except KeyError:
            print("rank error in csv file:{}".format(csv_path))

    columns = df.columns
    assert columns[0] in ["filename", "#filename"], "invalid CSV format"
    file_names = np.unique(df[columns[0]].values)

    # init: return length of dataset as the first element yielded
    init = 1
    if init:
        init = 0
        assert isinstance(len(file_names), int), "type error"
        yield len(file_names)

    df = df.drop(['file_size', 'file_attributes', 'region_count', 'region_id'], axis=1)

    for file in file_names:
        idx = df[columns[0]] == file
        df_filt = df[idx]
        image_path = "{}/{}".format(base_path, file)
        width, height = Image.open(image_path).size  # PIL gives size order different to numpy

        # contains masks for each object within image
        mask_data = []
        if close_up:
            x_points = []
            y_points = []
        # iterate over different individual objects within image
        for file_name, mask_points, class_name in df_filt.values:
            assert file_name == file, "wrong file"
            # load mask points, class_name (damage name), and create mask
            p = json.loads(mask_points)
            class_name = json.loads(class_name)
            if len(p) == 0:
                continue
            if kind not in class_name.keys():
                # TODO: quick fix to overcome mis-annotations
                if kind == "damages" and "damage" in class_name.keys():
                    class_name = class_name["damage"]
                else:
                    continue
            else:
                class_name = class_name[kind]

            class_name = class_name.lower()

            # filter unwanted classes
            if class_name in skip_classes:
                continue

            if kind == "parts":
                if remap:
                    class_name = parts_classes_v2[parts_classes_v2_remap[class_name]]
            mask = np.zeros([height, width], dtype=np.uint8)

            # generate masks for each object
            if len(p) < 1:  # ignore images without annotations
                continue

            # decide mask type
            if binary_mask:
                mask_val = 1
            else:
                mask_val = reverse_dict[class_name]
            try:
                if p["name"] in ["polygon", "polyline"]:
                    rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                    try:
                        mask[rr, cc] = mask_val
                    except IndexError:
                        print(file)
                elif p["name"] == "ellipse":
                    rr, cc = skimage.draw.ellipse(p['cy'], p['cx'], p['ry'], p['rx'])
                    try:
                        mask[rr, cc] = mask_val
                    except IndexError:
                        print(file)
                elif p["name"] == "circle":
                    rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'])
                    try:
                        mask[rr, cc] = mask_val
                    except IndexError as e:
                        print(file, e)
                elif p["name"] == "rect":
                    rr, cc = skimage.draw.rectangle(start=(p['y'], p['x']), extent=(p['height'], p['width']))
                    try:
                        mask[rr, cc] = mask_val
                    except IndexError as e:
                        print(file, e)
                else:
                    if p["name"] not in ["polygon", "polyline", "ellipse", "circle", "rect"]:
                        print("unknown annotation class name: {}".format(p["name"]))

                if close_up:
                    x_points.append(p['all_points_x'])
                    y_points.append(p['all_points_y'])

            except KeyError as e:
                print(csv_path, file_name, e)

            mask_data.append({'name': class_name, "mask": mask})

        if close_up:
            if not x_points or not y_points:
                bbox_coordinates = []
            else:
                x_array = np.hstack(x_points)
                y_array = np.hstack(y_points)
                x_max = np.amax(x_array)
                y_max = np.amax(y_array)
                x_min = np.amin(x_array)
                y_min = np.amin(y_array)
                bbox_height = y_max - y_min
                bbox_width = x_max - x_min

                increase_val_h = 200
                increase_val_w = 250

                if bbox_width < 250:
                    increase_val_w = 450
                if bbox_height < 200:
                    increase_val_h = 400

                x_min = max(0, x_min - increase_val_h)
                y_min = max(0, y_min - increase_val_w)
                bbox_height = min(height, bbox_height + increase_val_h)
                bbox_width = min(width, bbox_width + increase_val_w)

                bbox_coordinates = [x_min, y_min, bbox_height, bbox_width]

            out_values = (file, image_path, mask_data, kind, bbox_coordinates)

        else:
            out_values = (file, image_path, mask_data, kind, None)

        yield out_values


def convert_mturk(base_path, kind='mturk', conversion_func=None):
    """
    Function for reading mturk annotation output.
    NOTE: all images in .jpg format & all masks in .png format

    Args:
        base_path:          path to folder container images and png files
        kind:
        conversion_func:    function to create proto example

    Returns:
        Generator. First element is len dataset, after that tf.example protos for each image.

    """

    paths = ["{}/{}".format(base_path, x) for x in os.listdir(base_path) if x.endswith("png") and not x.startswith(".")]

    # init
    init = 1
    if init:
        init = 0
        assert isinstance(len(paths), int), "type error"
        yield len(paths)

    for mask_path in paths:
        img_path = mask_path.replace("png", "jpg")
        img_id = os.path.basename(img_path)

        mask = Image.open(mask_path)
        mask = np.array(mask.convert("RGB"))

        mask_data = []

        for cls in label_to_colour.keys():
            cls_mask = np.all(mask == label_to_colour[cls], axis=-1).astype(np.uint8)
            mask_data.append({'name': cls, "mask": cls_mask})

        example = conversion_func(img_id, img_path, mask_data, kind)

        yield example


def convert_coco(base_path, json_path, *args, **kwargs):
    """

    Args:
        base_path:          root folder containing images
        json_path:              json file containing the annotations

    Returns:

    """
    groundtruth_data = json.load(open(json_path, "r"))
    images = groundtruth_data['images']

    ignore_classes = {2: 'bicycle', 3: 'car', 4: 'motorcycle', 6: 'bus', 7: 'train', 8: 'truck'}

    annotations_index = {}
    for annotation in groundtruth_data['annotations']:
        image_id = annotation['image_id']

        if image_id not in annotations_index:
            annotations_index[image_id] = []
        annotations_index[image_id].append(annotation)

    selected_images = []

    for image in images:
        image_id = image['id']
        if image_id not in annotations_index:
            continue
        else:
            breaker = False
            labels = [a['category_id'] for a in annotations_index[image_id]]
            for annotation in labels:
                if annotation in ignore_classes:
                    breaker = True
                    break
            if breaker:
                continue
            else:
                filename = image['file_name']
                selected_images.append(filename)

    init = True
    if init:
        init = False
        yield len(selected_images)

    for filename in selected_images:
        image_path = os.path.join(base_path, filename)
        yield (filename, image_path)
