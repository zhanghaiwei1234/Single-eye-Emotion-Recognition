import json
import copy
import torch
import torch.utils.data as data
from dataloader import VideoLoader
from torch.utils.data.dataloader import default_collate

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map

def get_database(data, subset, event_video_path, frame_video_path, video_path_formatter):
    video_ids = []
    event_video_paths = []
    frame_video_paths = []
    annotations = []
    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            label = value['annotations']['label']
            event_video_paths.append(video_path_formatter(event_video_path, label, key))
            frame_video_paths.append(video_path_formatter(frame_video_path, label, key))
    return video_ids, event_video_paths, frame_video_paths, annotations

class VideoDataset(data.Dataset):
    def __init__(self,
                 event_video_path,
                 frame_video_path,
                 annotation_path,
                 subset,
                 event_spatial_transform=None,
                 frame_spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:root_path / label / video_id),
                 image_name_formatter=lambda x: f'{x:05d}.jpg',
                 target_type='label'):

        self.data, self.class_names = self.__make_dataset(event_video_path, frame_video_path, annotation_path, subset, video_path_formatter)
        self.event_spatial_transform = event_spatial_transform
        self.frame_spatial_transform = frame_spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader
        self.target_type = target_type

    def __make_dataset(self, event_video_path, frame_video_path, annotation_path, subset,
                       video_path_formatter):
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, event_video_path, frame_video_path, annotations = get_database(
            data, subset, event_video_path, frame_video_path, video_path_formatter)
        class_to_idx = get_class_labels(data)
        
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name
        n_videos = len(video_ids)
        dataset = []

        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            event_path = event_video_path[i]
            if not event_path.exists():
                continue

            frame_path = frame_video_path[i]
            if not frame_path.exists():
                continue
                
            segment = annotations[i]['segment']
            if segment[1] == 0:
                continue

            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                'event_video': event_path,
                'frame_video': frame_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id}
            dataset.append(sample)
        return dataset, idx_to_class

    def __loading(self, event_path, frame_path, frame_indices):
        clip1 = self.loader(event_path, frame_indices)
        clip2 = self.loader(frame_path, frame_indices)
        self.event_spatial_transform.randomize_parameters()
        self.frame_spatial_transform.randomize_parameters()
        clip1 = [self.event_spatial_transform(img) for img in clip1]
        clip2 = [self.frame_spatial_transform(img) for img in clip2]
        clip1 = torch.stack(clip1, 0).permute(1, 0, 2, 3)
        clip2 = torch.stack(clip2, 0).permute(1, 0, 2, 3)
        return clip1, clip2

    def __getitem__(self, index):
        event_path = self.data[index]['event_video']
        frame_path = self.data[index]['frame_video']

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        event_clip, frame_clip = self.__loading(event_path, frame_path, frame_indices)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return event_clip, frame_clip, target

    def __len__(self):
        return len(self.data)

class VideoDatasetMultiClips(VideoDataset):
    def __loading(self, event_path, frame_path, video_frame_indices):
        clip1 = []
        clip2 = []
        segments = []        
        clip1 = self.loader(event_path, video_frame_indices)
        clip2 = self.loader(frame_path, video_frame_indices)
        self.event_spatial_transform.randomize_parameters()
        self.frame_spatial_transform.randomize_parameters()
        clip1 = [self.event_spatial_transform(img) for img in clip1]
        clip2 = [self.frame_spatial_transform(img) for img in clip2]
        clip1 = torch.stack(clip1, 0).permute(1, 0, 2, 3)
        clip2 = torch.stack(clip2, 0).permute(1, 0, 2, 3)
        segments.append([min(video_frame_indices),
                 max(video_frame_indices) + 1])

        return clip1, clip2, segments

    def __getitem__(self, index):
        event_path = self.data[index]['event_video']
        frame_path = self.data[index]['frame_video']

        video_frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            video_frame_indices = self.temporal_transform(video_frame_indices)
        event_clip, frame_clip, segments = self.__loading(event_path, frame_path, video_frame_indices)

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        if 'segment' in self.target_type:
            if isinstance(self.target_type, list):
                segment_index = self.target_type.index('segment')
                targets = []
                for s in segments:
                    targets.append(copy.deepcopy(target))
                    targets[-1][segment_index] = s
            else:
                targets = segments
        else:
            targets = [target for _ in range(len(segments))]

        return event_clip, frame_clip, targets

def collate_fn(batch):
    batch_clips1, batch_clips2, batch_targets = zip(*batch)
    batch_clips1 = [clip for clip in batch_clips1]
    batch_clips2 = [clip for clip in batch_clips2]
    batch_targets = [target for multi_targets in batch_targets for target in multi_targets]
    target_element = batch_targets[0]
    if isinstance(target_element, int) or isinstance(target_element, str):
        return default_collate(batch_clips1), default_collate(batch_clips2), default_collate(batch_targets)
    else:
        return default_collate(batch_clips1), default_collate(batch_clips2), batch_targets