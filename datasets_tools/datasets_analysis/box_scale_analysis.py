import numpy as np
import os
import argparse

from filternet.utils import MOTClassesID, logger
import mmengine
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


def process_frame(annos, ID2CLASSES, imgsz) -> tuple[np.array]:
    all_ids = np.unique(annos[:, 1])

    area_ratios = []
    wh_ratios = []
    for tid in all_ids:
        anno_cur = annos[np.where(annos[:, 1] == tid)]
        if anno_cur[0][7] not in ID2CLASSES.keys():
            continue
        ws, hs = anno_cur[:, [4]], anno_cur[:, [5]]
        # cur_box = anno_cur[np.where(anno_cur[:, 1] == tid)][:, 2:6]
        area_ratio = np.multiply(ws, hs) / imgsz
        # print(area_ratio)
        area_ratios.append(area_ratio.mean())
        wh_ratios.append((ws / hs).mean())
    if area_ratios == []:
        return 0, 0
    return np.array(area_ratios).mean(), np.array(wh_ratios).mean()


if __name__ == '__main__':
    args = parse_args()

    ID2CLASSES = MOTClassesID.id2classes(args.datasets_name, args.classes)
    logger.info(f'{args.datasets_name}: \n{ID2CLASSES}')
    seqs = os.listdir(args.source_dir)
    # tbar = tqdm(seqs)

    all_area_ratios = []
    all_wh_ratios = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for seq in seqs:
            if seq == '.DS_Store':
                continue
            anno_file = os.path.join(args.source_dir, seq, 'gt', 'gt.txt')
            infos = mmengine.list_from_file(
                os.path.join(args.source_dir, seq, 'seqinfo.ini'))
            width = int(infos[5].strip().split('=')[1])
            height = int(infos[6].strip().split('=')[1])
            imgsz = width * height
            annos = np.loadtxt(anno_file, delimiter=',')
            futures.append(
                executor.submit(process_frame, annos, ID2CLASSES, imgsz))
        for future in futures:
            area_ratios, wh_ratios = future.result()
            all_area_ratios.append(area_ratios)
            all_wh_ratios.append(wh_ratios)

    print(
        f'Average area ratio on {ID2CLASSES.values()} = {np.array(all_area_ratios).mean()},\n',
        f'Average wh ratio on {ID2CLASSES.values()} = {np.array(all_wh_ratios).mean()}',
    )
