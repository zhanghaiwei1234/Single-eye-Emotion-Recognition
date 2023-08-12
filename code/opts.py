import argparse
from pathlib import Path

def parse_opts():
    parser = argparse.ArgumentParser()

    ## Pars of training and inference setting
    parser.add_argument('--manual_seed', default=2022, type=int, help='Manually set random seed')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--inference_batch_size', default=120, type=int, help='Batch Size for inference. 0 means this is the same as batch_size.')
    parser.add_argument('--batchnorm_sync', action='store_true', help='If true, SyncBatchNorm is used instead of BatchNorm.')
    parser.add_argument('--n_epochs', default=180, type=int, help='Number of total epochs to run')
    parser.add_argument('--learning_rate', default=0.015, type=float, help='Initial learning rate')
    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--test_epoch', default='/data/sometest/test_163/save_100.pth', type=str, help='path of test_checkpoints')

    ## Pars of optimizer SGD
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum of SGD optimizer')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay of SGD optimizer')

    ## Pars of input
    parser.add_argument('--root_path', default='/data', type=Path, help='Root directory path')
    parser.add_argument('--event_video_path', default='event_30', type=Path, help='path of events')
    parser.add_argument('--frame_video_path', default='frame', type=Path, help='path of frame')
    parser.add_argument('--annotation_path', default='emotion_new_adjust2.json', type=Path, help='path of .json (division of training and testing dataset)')
    parser.add_argument('--result_path', default=None, type=Path, help='path of results')
    parser.add_argument('--resume_path', default=None, type=Path, help='Save data (.pth) of previous training')
    parser.add_argument('--n_classes', default=7, type=int, help='Number of classes')
    parser.add_argument('--n_pretrain_classes', default=0, type=int, help=('Number of classes of pretraining task'))
    parser.add_argument('--pretrain_path', default=None, type=Path, help='Pretrained model path (.pth).')

    ## Pars of SNN neuron
    parser.add_argument('--thresh', default=0.3, type=float, help='threshold of snn neuron')
    parser.add_argument('--lens', default=0.5, type=float, help='hyper-parameters of approximate function')
    parser.add_argument('--decay', default=0.2, type=float, help='decay constants of snn neuron')

    ## output and show
    parser.add_argument('--output_topk', default=7, type=int, help='Top-k scores are saved in json file.')
    parser.add_argument('--checkpoint', default=100, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--tensorboard',  action='store_true', help='If true, output tensorboard log file.')

    ## train/val/inference
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
    parser.add_argument('--no_std_norm', action='store_true', help='If true, inputs are not normalized by standard deviation.')
    parser.add_argument('--value_scale', default=1, type=int, help='If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')
    parser.add_argument('--sample_size', default=90, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=4, type=int, help='Temporal duration of inputs')
    parser.add_argument('--sample_t_stride', default=4, type=int, help='If larger than 1, input frames are subsampled with the stride.')

    parser.add_argument('--inference', action='store_true', help='If true, inference is performed.')
    parser.add_argument('--inference_subset', default='test', type=str, help='Used subset in inference (train | val | test)')
    parser.add_argument('--inference_stride', default=0, type=int, help='Stride of sliding window in inference.')
    parser.add_argument('--inference_no_average', action='store_true', help='If true, outputs for segments in a video are not averaged.')
    parser.add_argument('--inference_sample_duration', default=4, type=int, help='Temporal duration of inputs')

    ## Pars of transforming distribution
    parser.add_argument( '--distributed', action='store_true', help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str, help='url used to set up distributed training')
    parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')

    args = parser.parse_args()

    return args
