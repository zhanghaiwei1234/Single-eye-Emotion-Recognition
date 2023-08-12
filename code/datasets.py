from Videodataset import VideoDataset, VideoDatasetMultiClips, collate_fn
from dataloader import VideoLoader

def image_name_formatter(x):
    return f'{x:05d}.jpg'

def get_training_data(event_video_path,
                      frame_video_path,
                      annotation_path,
                      event_spatial_transform=None,
                      frame_spatial_tramsform=None,
                      temporal_transform=None,
                      target_transform=None):
    loader = VideoLoader(image_name_formatter)
    video_path_formatter = (
        lambda root_path, label, video_id: root_path / label / video_id)
    training_data = VideoDataset(
        event_video_path,
        frame_video_path,
        annotation_path,
        'training',
        event_spatial_transform,
        frame_spatial_tramsform,
        temporal_transform,
        target_transform,
        loader,
        video_path_formatter,
    )
    return training_data

def get_inference_data(event_video_path,
                       frame_video_path,
                       annotation_path,
                       inference_subset,
                       event_spatial_transform=None,
                       frame_spatial_transform=None,
                       temporal_transform=None,
                       target_transform=None):
    loader = VideoLoader(image_name_formatter)
    video_path_formatter = (
        lambda root_path, label, video_id: root_path / label / video_id)
    subset = ''
    if inference_subset == 'train':
        subset = 'training'
    elif inference_subset == 'test':
        subset = 'testing'
    inference_data = VideoDatasetMultiClips(
        event_video_path,
        frame_video_path,
        annotation_path,
        subset,
        event_spatial_transform=event_spatial_transform,
        frame_spatial_transform=frame_spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        video_loader=loader,
        video_path_formatter=video_path_formatter,
        target_type=['video_id', 'segment'])
    return inference_data, collate_fn