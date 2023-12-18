import argparse
import os

import torch

# Distribute Dependence
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from cv_lib.utils import make_deterministic
from torchinfo import summary
from torchvision.datasets import ImageFolder

from torchvision.models.resnet import resnet50

from VQCLR.dist_utils.util import TensorboardLogger, MetricLogger, reduce_mean
from VQCLR.model.simclr import SimCLR
from VQCLR.model.simclr.modules import NT_Xent
from VQCLR.model.simclr.modules.transformations import TransformsSimCLR
from VQCLR.utils.functional import get_encoder
from VQCLR.utils.logger import get_logger
from VQCLR.utils.optimizer import load_optimizer


def make_args():
    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument("--save_dir", type=str, default='results')
    parser.add_argument("--weight", type=str)

    # Data Config
    parser.add_argument("--data_dir", type=str, default='/home/dataset_ssd/ILSVRC2012/train')

    # Train Config
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)

    # Model Config
    parser.add_argument("--encoder", type=str, default='resnet50', choices=['resnet50', 'resnet18', 'vit'])
    parser.add_argument("--projection_dim", type=int, default=64)

    # Loss Config
    parser.add_argument("--optimizer", type=str, default='Adam')
    parser.add_argument("--weight_decay", type=float, default=1.0e-6)
    parser.add_argument("--temperature", type=float, default=0.5)

    # Distribute Config
    parser.add_argument("-distributed", action="store_true")
    parser.add_argument('--ip', default='127.0.0.1', type=str)
    parser.add_argument('--port', default='23456', type=str)
    parser.add_argument('--beta', default=0.25, type=float)

    # Run Config
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--save_freq', default=1, type=int)

    args = parser.parse_args()
    return args


def main():
    args = make_args()
    args.nprocs = torch.cuda.device_count()
    print(args)
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    logger = get_logger('VQ-SimCLR')
    args.local_rank = local_rank
    make_deterministic(args.seed)
    init_method = 'tcp://' + args.ip + ':' + args.port

    # Init Distribute
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=args.nprocs, rank=local_rank)
    # Init Model
    encoder = get_encoder(args.encoder, pretrained=False)
    n_features = encoder.fc.in_features
    model = SimCLR(encoder, args.projection_dim, n_features)
    # Config Model
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer, scheduler = load_optimizer(args, model)
    logger.info('Model load successfully...')

    batch_size = args.batch_size

    train_dataset = ImageFolder(
        root=args.data_dir,
        transform=TransformsSimCLR(224),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )
    logger.info('Data load successfully...')
    if args.local_rank == 0:
        summary(model, input_data=(torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224)))
        log_writer = TensorboardLogger(log_dir=args.save_dir)
    else:
        log_writer = None

    num_training_steps_per_epoch = len(train_loader)
    logger.info(f'num_training_steps_per_epoch: {num_training_steps_per_epoch}')
    criterion = NT_Xent(args.batch_size, args.temperature, nprocs)

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        metric_logger = MetricLogger(delimiter="  ", is_master=args.local_rank == 0, logger=logger)
        header = 'Epoch: [{}]'.format(epoch)
        lr = optimizer.param_groups[0]["lr"]
        total_loss = 0.0
        for i, ((x_i, x_j), _) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):
            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)
            h_i, h_j, z_i, z_j = model(x_i, x_j)
            loss = criterion(z_i, z_j)
            torch.distributed.barrier()

            loss = reduce_mean(loss, args.nprocs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(loss=loss)

            total_loss += loss.item()

        if args.local_rank == 0:
            log_writer.update('Loss/train', loss=total_loss / len(train_loader))
            log_writer.update('Misc/learning_rate', lr=lr / len(train_loader))
            log_writer.set_step()

        metric_logger.synchronize_between_processes()
        if args.local_rank == 0:
            save_model_name(args, model, optimizer, epoch, filename=f'epoch_{epoch}')
            log_writer.flush()


def save_model_name(args, model, optimizer, epoch, filename='last', **kwargs):
    assert os.path.exists(args.save_dir)
    save_path = args.save_dir
    results_to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    results_to_save.update(**kwargs)
    torch.save(results_to_save, os.path.join(save_path, f'{filename}.pth'))


if __name__ == '__main__':
    main()
