from src.config import cfg
from src.models.model_builder import ModelBuilder
from src.utils.bbox import get_axis_aligned_bbox, bbox_to_polygon
from src.utils.data_collector import DataCollector
from src.trackers.tracker_builder import build_tracker
from toolkit.utils.region import vot_overlap, vot_float2str
from toolkit.datasets import DatasetFactory
from pathlib import Path
import numpy as np
import traceback
import argparse
import torch
import cv2


def main():
    parser = argparse.ArgumentParser(description='Tracking')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--dataset-root', type=str, help='dataset root')
    parser.add_argument('--config', default='', type=str, help='config file')
    parser.add_argument('--snapshot', default='', type=str, help='snapshot of models to eval')
    parser.add_argument('--video', default='', type=str, help='eval one special video')
    parser.add_argument('--video-subset', default='', type=str, help='eval a subset of videos specified in file')
    parser.add_argument('--vis', action='store_true', help='whether to visualize result')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    dataset_name = args.dataset
    dataset = DatasetFactory.create_dataset(
        name=dataset_name,
        dataset_root=str(Path(args.dataset_root) / dataset_name),
        load_img=False
    )

    net = ModelBuilder()
    net.load_state_dict(torch.load(args.snapshot))
    net.eval().cuda()

    tracker = build_tracker(net)

    videos_subset = []
    if args.video:
        videos_subset.append(args.video)
    if args.video_subset:
        with open(Path(args.video_subset), 'r') as file:
            videos_subset.extend([line.strip() for line in file])

    total_frames = 0
    for v_idx, video in enumerate(dataset):
        if videos_subset:
            if video.name not in videos_subset:
                continue
        total_frames += len(video)

    save_path = Path(args.config).parent / 'results' / dataset_name / cfg.META_ARC

    if cfg.REFINE_TEMPLATE.COLLECT_DATASET:
        # Define dataset shape to save it to hdf5 format
        shape = (total_frames, *cfg.REFINE_TEMPLATE.FEATURE_SIZE[1:])
        data_collector = DataCollector(save_path, shape)
    else:
        data_collector = None

    total_lost = 0
    total_fps = 0
    for v_idx, video in enumerate(dataset):
        try:
            if videos_subset:
                if video.name not in videos_subset:
                    continue
            print(f"Started: {video.name}")

            res = track(video, tracker, args.vis, data_collector)

            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f} fps Lost: {:2d}'.format(
                v_idx+1, video.name, res['total_time'], res['fps'], res['lost_times']))

            total_lost += res['lost_times']
            total_fps += res['fps']

            Path(save_path).mkdir(exist_ok=True, parents=True)
            result_path = Path(save_path) / f'{video.name}.txt'
            with open(result_path, 'w') as f:
                for x in res['pred_bboxes']:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([str(i) for i in x]) + '\n')

        except Exception as e:
            print(f"Failed to process {video.name}: {e}, {v_idx}")
            traceback.print_exc()

    print(f"Total lost: {total_lost}")

    if data_collector:
        data_collector.save()

    mean_fps = total_fps / v_idx
    fps_result_path = Path(save_path) / f'mean_fps'
    with open(fps_result_path, 'w') as f:
        f.write(str(mean_fps))


def track(video, tracker, visualize=False, data_collector=None):
    num_frames = len(video)
    frame_counter = 0
    frame_reset = 0  # used to indicate how many times the tracker was re-initialized
    lost_times = 0
    pred_bboxes = []  # Filled according to VOT protocol & used for metric calculation
    total_time = 0

    zero_tensor = torch.zeros(cfg.REFINE_TEMPLATE.FEATURE_SIZE, dtype=torch.float32).cpu().data

    for f, (im, gt) in enumerate(video):
        if len(gt) == 4:
            gt = bbox_to_polygon(gt)

        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt))

        start_time = cv2.getTickCount()
        if f == frame_counter:  # Init or reset after lost frame
            gt_bbox = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
            tracker.init(im, gt_bbox)
            pred_bbox = gt_bbox
            pred_bboxes.append(1)
            total_time += cv2.getTickCount() - start_time
            frame_reset = 0
            if data_collector:
                data_collector.add_init(f, num_frames, init_feat=tracker.zf_init)

        elif f > frame_counter:  # Tracking
            frame_reset += 1
            outputs = tracker.track(im)
            pred_bbox = outputs['bbox']
            if data_collector:
                # Extract ground-truth template features
                gt_rect = np.array([cx, cy, w, h])
                gt_zf = tracker.extract_template(im, gt_rect) if w * h != 0 else zero_tensor
                data_collector.add_tracking(
                    f, num_frames, frame_reset, cur_feat=outputs['zf'],
                    pre_feat=tracker.zf, gt_feat=gt_zf
                )

            overlap = vot_overlap(pred_bbox, gt, (im.shape[1], im.shape[0]))
            if overlap > 0:
                pred_bboxes.append(pred_bbox)
            else:
                pred_bboxes.append(2)
                # skip 5 frames after object lost (as suggested by VOT)
                frame_counter = f + 5
                lost_times += 1
            total_time += cv2.getTickCount() - start_time

        elif f < frame_counter or w * h == 0:  # Skipping
            pred_bboxes.append(0)
            total_time += cv2.getTickCount() - start_time
            frame_reset = 0
            if data_collector:
                data_collector.add_init(f, num_frames, zero_tensor)

        if visualize:
            cv2.polylines(im, [np.array(gt, np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)

            bbox = list(map(int, pred_bbox))
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (0, 255, 255), 3)

            cv2.putText(im, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(im, str(lost_times), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            window_name = 'test'
            cv2.imshow(window_name, im)
            cv2.moveWindow(window_name, 100, 10)
            cv2.waitKey(1)

    total_time /= cv2.getTickFrequency()
    cv2.destroyAllWindows()
    return {
        'pred_bboxes': pred_bboxes,
        'lost_times': lost_times,
        'total_time': total_time,
        'fps': f / total_time,
    }


if __name__ == '__main__':
    main()
