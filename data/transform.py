import torch
from torchvision import transforms

class AMCTransform(object):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __call__(self, signal):
        return self.transform(signal)

# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80