<<<<<<< HEAD
from torchvision import get_image_backend
import json
from pathlib import Path

import torch
import torch.utils.data as data
from PIL import Image

class ImageLoaderPIL(object):

    def __call__(self, path):
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            if image_path.exists():
                video.append(self.image_loader(image_path))

        return video
=======
from torchvision import get_image_backend
import json
from pathlib import Path

import torch
import torch.utils.data as data
from PIL import Image

class ImageLoaderPIL(object):

    def __call__(self, path):
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            if image_path.exists():
                video.append(self.image_loader(image_path))

        return video
>>>>>>> 8a8b3087985886893c8b6aa3e111af742ccd699b
