import random

class TemporalCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame_indices):
        for i, t in enumerate(self.transforms):
            if isinstance(frame_indices[0], list):
                next_transforms = TemporalCompose(self.transforms[i:])
                dst_frame_indices = [
                    next_transforms(clip_frame_indices)
                    for clip_frame_indices in frame_indices
                ]
                return dst_frame_indices
            else:
                frame_indices = t(frame_indices)
        return frame_indices

class LoopPadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices
        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)
        return out

class TemporalRandomCrop(object):
    def __init__(self, size):
        self.size = size
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))
        out = frame_indices[begin_index:end_index]
        if len(out) < self.size:
            out = self.loop(out)
        return out


class TemporalSubsampling(object):
    def __init__(self, stride):
        self.stride = stride

    def __call__(self, frame_indices):
        num = random.randint(0, 4)
        return frame_indices[num::self.stride]
