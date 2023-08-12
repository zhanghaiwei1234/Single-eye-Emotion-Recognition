from pathlib import Path
import json
import random
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.backends import cudnn
from opts import parse_opts
from model_snn_cnn import generate_model_snn, make_data_parallel
from mean import get_mean_std
from spatial_transforms import Compose, Normalize, Resize, ToTensor, ScaleValue
from temporal_transforms import TemporalRandomCrop, TemporalSubsampling, TemporalCompose
from datasets import get_inference_data
from utils import worker_init_fn
import inference

def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.event_video_path = opt.root_path / opt.event_video_path
        opt.frame_video_path = opt.root_path / opt.frame_video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.begin_epoch = 1
    opt.event_mean, opt.event_std = get_mean_std(opt.value_scale, data_type='event')
    opt.frame_mean, opt.frame_std = get_mean_std(opt.value_scale, data_type='frame')

    if opt.distributed:
        opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)
    return opt

def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)

def get_inference_utils(opt):
    event_normalize = get_normalize_method(opt.event_mean, opt.event_std, opt.no_mean_norm,opt.no_std_norm)
    frame_normalize = get_normalize_method(opt.frame_mean, opt.frame_std, opt.no_mean_norm,opt.no_std_norm)
    event_spatial_transform = [Resize(opt.sample_size)]
    frame_spatial_transform = [Resize(opt.sample_size)]
    event_spatial_transform.append(ToTensor())
    frame_spatial_transform.append(ToTensor())
    event_spatial_transform.append(ScaleValue(opt.value_scale))
    event_spatial_transform.append(event_normalize)
    frame_spatial_transform.append(ScaleValue(opt.value_scale))
    frame_spatial_transform.append(frame_normalize)

    event_spatial_transform = Compose(event_spatial_transform)
    frame_spatial_transform = Compose(frame_spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(TemporalRandomCrop(opt.inference_sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)
    inference_data, collate_fn = get_inference_data(opt.event_video_path, opt.frame_video_path, opt.annotation_path, opt.inference_subset, event_spatial_transform, frame_spatial_transform, temporal_transform)

    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=opt.inference_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)

    return inference_loader, inference_data.class_names

def main_worker(index, opt):
    
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)
    os.environ['PYTHONHASHSEED'] = str(opt.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    if opt.distributed:
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
        dist.init_process_group(backend='nccl',
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.n_threads = int(
            (opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    model = generate_model_snn()
    model.load_state_dict(torch.load(opt.test_epoch)['state_dict'])
    epoch_idx = int(opt.test_epoch.split('_')[-1].split('.')[0])
    #model.load_state_dict(torch.load('/data/sometest/test-modify/save_100.pth')['state_dict'])

    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports Distributed DataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = make_data_parallel(model, opt.distributed, opt.device)
 
    if opt.is_master_node:
        print(model)
        
    if opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    if opt.inference:
        for test_num in range(20):
            inference_loader, inference_class_names = get_inference_utils(opt)
            inference_result_path = opt.result_path / '{}.json'.format(opt.inference_subset +  '_' + str(test_num+1))
            inference.inference(inference_loader, model, inference_result_path, inference_class_names, opt.inference_no_average, opt.output_topk, epoch_idx, tb_writer, str(test_num+1))

if __name__ == '__main__':
    opt = get_opt()
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True

    opt.ngpus_per_node = torch.cuda.device_count()
    print('opt.ngpus_per_node',opt.ngpus_per_node)

    main_worker(-1, opt)