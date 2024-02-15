
#%%
from torch_snippets import *
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.datasets import MNIST
from torchsummary import summary


#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=0.5)])

data_loader = torch.utils.data.DataLoader(MNIST('~/data', train=True, download=True, transform=transform),
                                          batch_size=128, shuffle=True, drop_last=True
                                          )


#%% define discriminator model class
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)
        
        
#%% summary of discriminator
discriminator = Discriminator().to(device=device)
summary(discriminator, torch.zeros(1,784))
    

# %% define generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.model(x)
    
generator = Generator().to(device=device)
summary(generator, torch.zeros(1, 100))

#%% define func to generate noise
def noise(size):
    n = torch.randn(size, 100)
    return n.to(device)
