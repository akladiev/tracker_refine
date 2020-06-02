from src.models.refine import get_refine_net
from src.utils.log_helper import Dummy
from src.utils.average_meter import AverageMeter
from src.utils.data_collector import RefineDataset

from pathlib import Path
from tensorboardX import SummaryWriter
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Training network for template refining')
    parser.add_argument('--data-path', required=True, type=str, help='Path to dataset')
    parser.add_argument('--num-epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', default=96, type=int, help='Batch size')
    parser.add_argument('--save-every', default=10, type=int, help='Checkpoint save frequency')
    parser.add_argument('--save-dir', default='checkpoints', type=str, help='Folder path to save checkpoints')
    parser.add_argument('--log-dir', default='logs', type=str, help='Folder path to write tensorboard logs')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate')
    parser.add_argument('--net', default='Refine3L', type=str, help='Name of the network to train')

    args = parser.parse_args()

    data_path = Path(args.data_path)
    save_path = Path(args.save_dir)

    save_path.mkdir(exist_ok=True, parents=True)
    tb_writer = SummaryWriter(args.log_dir) if args.log_dir else Dummy()

    net = get_refine_net(args.net)
    net.cuda()

    loss_func = nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.SGD(net.parameters(), args.lr, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=args.num_epochs // 10, gamma=0.5)

    dataset = RefineDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(0, args.num_epochs):
        losses = AverageMeter()

        for t, batch in enumerate(dataloader):
            input = Variable(batch['input']).cuda()
            target = Variable(batch['target']).cuda()

            output = net(input)
            loss = loss_func(output, target) / target.size(0)
            losses.update(loss=loss.cpu().data.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)

            print(f'Epoch {str(epoch).zfill(3)} iter {str(t).zfill(5)}/{len(dataset)}\t'
                  f'Loss {str(round(losses.val["loss"], 4)).zfill(10)} '
                  f'(avg: {round(losses.avg("loss"), 4):.4f})\t')
            tb_writer.add_scalar(f'loss', losses.val["loss"], t)

        if (epoch + 1) % args.save_every == 0:
            name = save_path / f'refine{epoch + 1}.pth'
            torch.save({'state_dict': net.state_dict()}, str(name))


if __name__ == '__main__':
    main()
