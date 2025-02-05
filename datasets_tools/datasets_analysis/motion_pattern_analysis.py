"""Modified from DanceTrack. please cite the following paper if you use this
code:

@inproceedings{2022sundancetrackmultiobjecttracking,
    title = {DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion},
    booktitle = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    author = {Sun, Peize and Cao, Jinkun and Jiang, Yi and Yuan, Zehuan and Bai, Song and Kitani, Kris and Luo, Ping},
    year = {2022},
    pages = {20961--20970},
    doi = {10.1109/CVPR52688.2022.02032},
    langid = {english},
    keywords = {Behavior analysis,Benchmark testing,Datasets and evaluation,
    Detectors,Location awareness,Motion and tracking,Object detection,Pipelines,Tracking,Visualization}
    }
Script to calculate the average IoU of the same object on consecutive
frames, and the relative switch frequency (Figure3(b) and Figure3(c)).
The original data in paper is calculated on all sets: train+val+test.
On the train-set:
    * Average IoU on consecutive frames = 0.894
    * Relative Position Switch frequency = 0.031
On the val-set:
    * Average IoU on consecutive frames = 0.909
    * Relative Position Switch frequency = 0.030
The splitting of subsets is
"""
import numpy as np
import os
from tqdm import tqdm
import argparse

from filternet.utils import MOTClassesID, logger

from concurrent.futures import ThreadPoolExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir',
                        type=str,
                        default='',
                        help='path to the source directory')
    parser.add_argument('--datasets_name',
                        type=str,
                        default='MOT',
                        help='support MOT, LEO and SNMOT(Soccer Net MOT)')
    parser.add_argument('--classes',
                        nargs='+',
                        default=[],
                        help='input multiple items')
    return parser.parse_args()


def box_area(arr) -> np.array:
    # arr: np.array([[x1, y1, x2, y2]])
    width = arr[:, 2] - arr[:, 0]
    height = arr[:, 3] - arr[:, 1]
    return width * height


def _box_inter_union(arr1, arr2) -> tuple[np.array, np.array]:
    # arr1 of [N, 4]
    # arr2 of [N, 4]
    area1 = box_area(arr1)
    area2 = box_area(arr2)

    # Intersection
    top_left = np.maximum(arr1[:, :2], arr2[:, :2])  # [[x, y]]
    bottom_right = np.minimum(arr1[:, 2:], arr2[:, 2:])  # [[x, y]]
    wh = bottom_right - top_left
    # clip: if boxes not overlap then make it zero
    intersection = wh[:, 0].clip(0) * wh[:, 1].clip(0)

    # union
    union = area1 + area2 - intersection
    return intersection, union


def box_iou(arr1, arr2) -> np.array:
    # arr1[N, 4]
    # arr2[N, 4]
    # N = number of bounding boxes
    if not (arr1[:, 2:] > arr1[:, :2]).all():
        return np.zeros((1, ))
    if not (arr2[:, 2:] > arr2[:, :2]).all():
        return np.zeros((1, ))
    inter, union = _box_inter_union(arr1, arr2)
    iou = inter / union
    return iou


def consecutive_iou(annos, ID2CLASSES) -> tuple[int, int]:
    """calculate the IoU over bboxes on the consecutive frames."""
    max_frame = int(annos[:, 0].max())
    min_frame = int(annos[:, 0].min())
    total_iou = 0
    total_frequency = 0
    for find in range(min_frame, max_frame):
        anno_cur = annos[np.where(annos[:, 0] == find)]
        anno_next = annos[np.where(annos[:, 0] == find + 1)]
        ids_cur = np.unique(anno_cur[:, 1])
        ids_next = np.unique(anno_next[:, 1])
        common_ids = np.intersect1d(ids_cur, ids_next)
        for tid in common_ids:
            if anno_cur[np.where(
                    anno_cur[:, 1] == tid)][0][7] not in ID2CLASSES.keys():
                continue
            cur_box = anno_cur[np.where(anno_cur[:, 1] == tid)][:, 2:6]
            next_box = anno_next[np.where(anno_next[:, 1] == tid)][:, 2:6]

            cur_box[:, 2:] += cur_box[:, :2]
            next_box[:, 2:] += next_box[:, :2]
            iou = box_iou(cur_box, next_box).item()
            total_iou += iou
            total_frequency += 1
    return total_iou, total_frequency


def process_frame(find, annos, ID2CLASSES) -> tuple[int, int]:
    anno_cur = annos[np.where(annos[:, 0] == find)]
    anno_next = annos[np.where(annos[:, 0] == find + 1)]
    ids_cur = np.unique(anno_cur[:, 1])
    ids_next = np.unique(anno_next[:, 1])
    common_ids = np.intersect1d(ids_cur, ids_next)
    total_iou = 0
    total_frequency = 0
    for tid in common_ids:
        if anno_cur[np.where(
                anno_cur[:, 1] == tid)][0][7] not in ID2CLASSES.keys():
            continue
        cur_box = anno_cur[np.where(anno_cur[:, 1] == tid)][:, 2:6]
        next_box = anno_next[np.where(anno_next[:, 1] == tid)][:, 2:6]

        cur_box[:, 2:] += cur_box[:, :2]
        next_box[:, 2:] += next_box[:, :2]
        iou = box_iou(cur_box, next_box).item()
        total_iou += iou
        total_frequency += 1
    return total_iou, total_frequency


def consecutive_iou_threaded(annos,
                             ID2CLASSES,
                             num_threads=4) -> tuple[int, int]:
    """Calculate the IoU over bboxes on the consecutive frames in parallel."""
    max_frame = int(annos[:, 0].max())
    min_frame = int(annos[:, 0].min())
    total_iou = 0
    total_frequency = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for find in range(min_frame, max_frame):
            futures.append(
                executor.submit(process_frame, find, annos, ID2CLASSES))

        for future in futures:
            iou, frequency = future.result()
            total_iou += iou
            total_frequency += frequency

    return total_iou, total_frequency


def center(box) -> np.ndarray:
    return (box[0] + 0.5 * box[2], box[1] + 0.5 * box[3])


def relative_switch(annos, ID2CLASSES) -> tuple[int, int]:
    """calculate the frequency of relative position switch regarding center
    location."""
    max_frame = int(annos[:, 0].max())
    min_frame = int(annos[:, 0].min())
    switch = 0
    sw_freq = 0
    for find in range(min_frame, max_frame):
        anno_cur = annos[np.where(annos[:, 0] == find)]
        anno_next = annos[np.where(annos[:, 0] == find + 1)]
        ids_cur = np.unique(anno_cur[:, 1])
        ids_next = np.unique(anno_next[:, 1])
        common_ids = np.intersect1d(ids_cur, ids_next)
        for id1 in common_ids:
            if anno_cur[np.where(
                    anno_cur[:, 1] == id1)][0][7] not in ID2CLASSES.keys():
                # only compute the specific class in ID2CLASSES in the current frame
                continue
            for id2 in common_ids:
                sw_freq += 1
                if id1 == id2:
                    continue
                box_cur_1 = anno_cur[np.where(anno_cur[:, 1] == id1)][0][2:6]
                box_cur_2 = anno_cur[np.where(anno_cur[:, 1] == id2)][0][2:6]
                box_next_1 = anno_next[np.where(anno_next[:,
                                                          1] == id1)][0][2:6]
                box_next_2 = anno_next[np.where(anno_next[:,
                                                          1] == id2)][0][2:6]
                left_right_cur = center(box_cur_1)[0] >= center(box_cur_2)[0]
                left_right_next = center(box_next_1)[0] >= center(
                    box_next_2)[0]
                top_down_cur = center(box_cur_1)[1] >= center(box_cur_2)[1]
                top_down_next = center(box_next_1)[1] >= center(box_next_2)[1]
                if (left_right_cur != left_right_next) or (top_down_cur !=
                                                           top_down_next):
                    switch += 1
    return switch, sw_freq


def process_relative_switch_frame(find, annos, ID2CLASSES) -> tuple[int, int]:
    anno_cur = annos[np.where(annos[:, 0] == find)]
    anno_next = annos[np.where(annos[:, 0] == find + 1)]
    ids_cur = np.unique(anno_cur[:, 1])
    ids_next = np.unique(anno_next[:, 1])
    common_ids = np.intersect1d(ids_cur, ids_next)
    switch = 0
    sw_freq = 0
    for id1 in common_ids:
        if anno_cur[np.where(
                anno_cur[:, 1] == id1)][0][7] not in ID2CLASSES.keys():
            continue
        for id2 in common_ids:
            sw_freq += 1
            if id1 == id2:
                continue
            box_cur_1 = anno_cur[np.where(anno_cur[:, 1] == id1)][0][2:6]
            box_cur_2 = anno_cur[np.where(anno_cur[:, 1] == id2)][0][2:6]
            box_next_1 = anno_next[np.where(anno_next[:, 1] == id1)][0][2:6]
            box_next_2 = anno_next[np.where(anno_next[:, 1] == id2)][0][2:6]
            left_right_cur = center(box_cur_1)[0] >= center(box_cur_2)[0]
            left_right_next = center(box_next_1)[0] >= center(box_next_2)[0]
            top_down_cur = center(box_cur_1)[1] >= center(box_cur_2)[1]
            top_down_next = center(box_next_1)[1] >= center(box_next_2)[1]
            if (left_right_cur != left_right_next) or (top_down_cur !=
                                                       top_down_next):
                switch += 1
    return switch, sw_freq


def relative_switch_threaded(annos,
                             ID2CLASSES,
                             num_threads=4) -> tuple[int, int]:
    """Calculate the frequency of relative position switch regarding center
    location in parallel."""
    max_frame = int(annos[:, 0].max())
    min_frame = int(annos[:, 0].min())
    switch = 0
    sw_freq = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for find in range(min_frame, max_frame):
            futures.append(
                executor.submit(process_relative_switch_frame, find, annos,
                                ID2CLASSES))

        for future in futures:
            s, f = future.result()
            switch += s
            sw_freq += f

    return switch, sw_freq


if __name__ == '__main__':
    args = parse_args()

    ID2CLASSES = MOTClassesID.id2classes(args.datasets_name, args.classes)
    logger.info(f'{args.datasets_name}: \n{ID2CLASSES}')
    seqs = os.listdir(args.source_dir)
    all_iou, all_freq = 0, 0
    all_switch, all_sw_freq = 0, 0
    tbar = tqdm(seqs)
    for seq in tbar:
        tbar.set_description('Processing {}'.format(seq))
        if seq == '.DS_Store':
            continue
        anno_file = os.path.join(args.source_dir, seq, 'gt/gt.txt')
        annos = np.loadtxt(anno_file, delimiter=',')
        seq_iou, seq_freq = consecutive_iou_threaded(annos, ID2CLASSES)
        seq_switch, seq_sw_freq = relative_switch_threaded(annos, ID2CLASSES)
        all_iou += seq_iou
        all_freq += seq_freq
        all_switch += seq_switch
        all_sw_freq += seq_sw_freq
    print('Average IoU on consecutive frames = {}'.format(all_iou / all_freq))
    print('Relative Position Switch frequency = {}'.format(all_switch /
                                                           all_sw_freq))
