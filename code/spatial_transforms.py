from torchvision.transforms import transforms

class Compose(transforms.Compose):
    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()

class ToTensor(transforms.ToTensor):
    def randomize_parameters(self):
        pass

class Normalize(transforms.Normalize):
    def randomize_parameters(self):
        pass

class ScaleValue(object):
    def __init__(self, s):
        self.s = s

    def __call__(self, tensor):
        tensor *= self.s
        return tensor

    def randomize_parameters(self):
        pass

class Resize(transforms.Resize):
    def randomize_parameters(self):
        pass



