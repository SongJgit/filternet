import argparse
import os
import os.path as osp
from collections import defaultdict

import mmengine
import numpy as np
import pandas as pd
from tqdm import tqdm
from filternet.utils import MOTClassesID


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MOT label and detections to track format.')
    parser.add_argument('-i', '--input', help='path of MOT data')
    parser.add_argument('-o',
                        '--output',
                        help='path to save coco formatted label file')
    parser.add_argument(
        '--max_truncate',
        type=int,
        default=50,
        help='Maximum number of interrupt frames,,fps25,50 is 2 sec')
    parser.add_argument('--data_mode',
                        nargs='+',
                        default=['train'],
                        help='input multiple items')
    parser.add_argument(
        '--split-train',
        action='store_true',
        help='split the train set into half-train and half-validate.')
    parser.add_argument('--datasets_name',
                        type=str,
                        default='MOT',
                        help='support MOT, LEO and SNMOT(Soccer Net MOT)')

    return parser.parse_args()


def main():
    args = parse_args()
    if not osp.isdir(args.output):
        os.makedirs(args.output)

    sets = args.data_mode
    if args.split_train:
        sets += ['half-train', 'half-val']
    ID2CLASSES = MOTClassesID.id2classes(args.datasets_name, args.classes)
    for subset in sets:
        if 'half' in subset:
            in_folder = osp.join(args.input, 'train')
        else:
            in_folder = osp.join(args.input, subset)
        outputs = defaultdict(list)
        outputs['data_info'] = dict(categories=ID2CLASSES)

        out_file = osp.join(args.output, f'{subset}_track.json')
        video_names = os.listdir(in_folder)
        vid_id = 1
        global_track_id = 1
        for video_name in tqdm(video_names):
            video_folder = osp.join(in_folder, video_name)
            infos = mmengine.list_from_file(f'{video_folder}/seqinfo.ini')
            # img_folder = infos[2].strip().split('=')[1]
            # img_names = os.listdir(f'{video_folder}/{img_folder}')
            # img_names = sorted(img_names)
            fps = int(infos[3].strip().split('=')[1])

            # num_imgs = int(infos[4].strip().split('=')[1])
            # assert num_imgs == len(img_names)
            width = int(infos[5].strip().split('=')[1])
            height = int(infos[6].strip().split('=')[1])
            video = dict(id=vid_id,
                         name=video_name,
                         fps=fps,
                         width=width,
                         height=height)
            gts = mmengine.list_from_file(f'{video_folder}/gt/gt.txt')
            gts = [gt.strip().split(',') for gt in gts]
            df = pd.DataFrame(gts)
            df = df.astype('float')
            df.rename(columns={
                0: 'frame_id',
                1: 'id',
                2: 'x',
                3: 'y',
                4: 'w',
                5: 'h',
                6: 'conf',
                7: 'category_id',
                8: 'visibility'
            },
                      inplace=True)  # noqa: E126
            track_ids = df['id'].unique()

            for track_id in track_ids:
                track = df[df['id'] == track_id]

                if len(track) <= 100:
                    continue

                if 'half' in subset:
                    split_frame = len(track) // 2 + 1
                    if 'train' in subset:
                        track = track.iloc[:split_frame]
                    elif 'val' in subset:
                        track = track.iloc[split_frame:]
                    else:
                        raise ValueError(
                            'subset must be named with `train` or `val`')
                if len(track) == 0:
                    continue

                if len(track['category_id'].unique()) > 1:
                    # There are some annotation errors in VisDrone.
                    # The first half of a trajectory is people, and the second half is car.
                    # After our inspection, it was found to be a clear labeling error,
                    # so we will skip this type of error again.
                    # Ex: In trainset, uav0000329_04715_v, target id=45, frame id=46~47, people -> car, actual 'car'
                    continue
                cat_id = track['category_id'].unique().item()

                if ID2CLASSES[cat_id] in outputs:
                    pass
                else:
                    outputs[ID2CLASSES[cat_id]] = []
                new_tracks = []

                for idx in range(len(track)):
                    frame_id = track.iloc[idx]['frame_id']
                    visibility = track.iloc[idx]['visibility']
                    # log the track id
                    last_frame_id = new_tracks[-1][0] if len(
                        new_tracks) > 0 else frame_id - 1

                    if last_frame_id == frame_id:
                        print(
                            'Please check the data,',
                            f'there is a object with the same ID in frame {int(frame_id)}  of video {video_name}.'
                        )
                    if abs(frame_id - last_frame_id) >= args.max_truncate:
                        #
                        break
                    while last_frame_id + 1 != frame_id:  # make track continuous.
                        new_tracks.append([
                            last_frame_id + 1, np.nan, np.nan, np.nan, np.nan,
                            0
                        ])
                        last_frame_id += 1

                    x = track.iloc[idx]['x'] if track.iloc[idx]['x'] > 0 else 0
                    y = track.iloc[idx]['y'] if track.iloc[idx]['y'] > 0 else 0

                    w = track.iloc[idx]['w']
                    h = track.iloc[idx]['h']
                    new_track = [frame_id, x, y, w, h, visibility]
                    new_tracks.append(new_track)
                outputs[ID2CLASSES[cat_id]].append(
                    dict(video=video,
                         sequence=new_tracks,
                         global_track_id=global_track_id))
                global_track_id += 1
            vid_id += 1
        outputs['data_info']['num_track'] = {}
        for _, cat_name in tqdm(ID2CLASSES.items()):
            if cat_name in outputs.keys():
                outputs['data_info']['num_track'][cat_name] = len(
                    outputs[cat_name])
        print(outputs['data_info'])
        mmengine.dump(outputs, out_file)
        print(f'Done! Saved as {out_file}')


if __name__ == '__main__':

    main()
