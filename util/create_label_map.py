import os
from util.dataset_info import damage_classes_reverse, parts_classes_reverse
# from src.data.dataset_info import damage_classes_reverse, parts_classes_reverse


def create_label_maps(output_path, label_dicts=None):
    if label_dicts is None:
        label_dicts = {
            'damage': damage_classes_reverse,
            'parts': parts_classes_reverse
        }

    for name, labels_dict in label_dicts.items():
        end = '\n'
        s = ' '
        output_name = '{}/{}_label_map.pbtxt'.format(output_path, name)
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        for name, ID in labels_dict.items():
            out = ''
            out += 'item' + s + '{' + end
            out += s * 2 + 'id:' + ' ' + (str(ID)) + end
            out += s * 2 + 'name:' + ' ' + '\'' + name + '\'' + end
            out += '}' + end * 2

            with open(output_name, 'a') as f:
                f.write(out)
