import argparse

from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from toolkit.datasets import OTBDataset, LaSOTDataset, VOTDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark

parser = argparse.ArgumentParser(description='Tracking evaluation')
parser.add_argument('--tracker_path', '-p', type=str, help='tracker result path')
parser.add_argument('--dataset', '-d', type=str, help='dataset name')
parser.add_argument('--dataset-root', '-dr', type=str, help='datasets root dir')
parser.add_argument('--num', '-n', default=1, type=int, help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='', type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level', action='store_true')
parser.set_defaults(show_video_level=False)
args = parser.parse_args()


def main():
    tracker_dir = Path(args.tracker_path)
    tracker_path = tracker_dir / args.dataset
    trackers = tracker_path.glob("*")
    trackers = [Path(x).stem for x in trackers]

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    root = str(Path(args.dataset_root) / args.dataset)
    if 'OTB' in args.dataset:
        dataset = OTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
    elif 'LaSOT' == args.dataset:
        dataset = LaSOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                show_video_level=args.show_video_level)
    elif args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019', 'debug']:
        dataset = VOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_path, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                trackers), desc='eval ar', total=len(trackers), ncols=100):
                ar_result.update(ret)

        benchmark = EAOBenchmark(dataset)
        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval eao', total=len(trackers), ncols=100):
                eao_result.update(ret)
        ar_benchmark.show_result(ar_result, eao_result,
                show_video_level=args.show_video_level)


if __name__ == '__main__':
    main()
