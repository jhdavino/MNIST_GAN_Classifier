import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt


dim = 28

class GeneratorNet(torch.nn.Module):

    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = dim*dim

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n


generator = GeneratorNet()

if torch.cuda.is_available():
    generator.cuda()


generator.load_state_dict(torch.load('model/gen147.model', map_location='cpu'))
generator.eval()


n_imgs = 1000
count = 1

imgs = generator(noise(n_imgs)).cpu()
path = 'dataset/'

for f in imgs:
    #f = fake.detach().numpy()
    filename = path + 'img' + str(count) + ".png"

    img_ = f.reshape(28,28).detach().numpy()
    img = plt.imshow(img_, cmap='gray')
    plt.imsave(filename, img_, cmap='gray')
    #plt.show()

    count += 1
