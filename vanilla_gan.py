
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

# %% define func to train discriminator
def discriminator_train_step(real_data, fake_data, 
                             d_optimizer, loss):
    d_optimizer.zero_grad()
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, torch.ones(len(real_data),1).to(device))
    error_real.backward()
    
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, torch.zeros(len(fake_data), 1).to(device))
    error_fake.backward()
    
    d_optimizer.step()
    return error_real + error_fake


def generator_train_step(real_data, fake_data, g_optimizer, loss):
    g_optimizer.zero_grad()
    
    prediction = discriminator(fake_data)
    error = loss(prediction, torch.ones(len(real_data), 1).to(device))
    error.backward()
    g_optimizer.step()
    return error
    
#%% define discriminator and loss
discriminator = Discriminator().to(device)
generator = Generator().to(device)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
loss = nn.BCELoss()
num_epochs = 200
log = Report(num_epochs)


#%% run model overepochs
for epoch in range(num_epochs):
    N = len(data_loader)
    for i, (images, _) in enumerate(data_loader):
        real_data = images.view(len(images), -1).to(device)
        fake_data = generator(noise(len(real_data))).to(device)
        fake_data = fake_data.detach()
        d_loss = discriminator_train_step(real_data, fake_data, d_optimizer, loss)
        
        fake_data = generator(noise(len(real_data))).to(device)
        g_loss = generator_train_step(real_data, fake_data, g_optimizer, loss)
        log.record(epoch + (1+i)/N, d_loss=d_loss.item(), g_loss=g_loss.item(), end="\r")
    log.report_avgs(epoch+1)
log.plot_epochs(["d_loss", "g_loss"])
    
    
#%% visualize fake data
z = torch.randn(64, 100).to(device)
sample_images = generator(z).data.cpu().view(64, 1, 28, 28)
grid = make_grid(sample_images, nrow=8, normalize=True)
show(grid.cpu().detach().permute(1,2,0), sz=5)
    
# %%
### print(12)
print("12")
# %%
