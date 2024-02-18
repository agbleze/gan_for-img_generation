
#%%
!wget https://www.dropbox.com/s/rbajpdlh7efkdo1/male_female_face_images.zip


# %%
import torchvision
from torch_snippets import *
from torchvision import transforms
import torchvision.utils as vutils
import cv2, numpy as np, pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"


#%% import model for face detection to crop them to focus on only relevant information we
## want to generate
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#%% crop faces (target to generate) and export them
img_folder = "C:/Users/agbji/Documents/codebase/gan_for_img_generation/male_female_face_images/"
images = Glob(img_folder + "females/*.jpg") + Glob(img_folder + "males/*.jpg")

#%%
for i in range(len(images)):
    img = read(images[i], 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img2 = img[y:(y+h), x:(x+w), :]
    cv2.imwrite("cropped_faces/" + str(i) + ".jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))


#%% image transformation
transform = transforms.Compose([transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]
                               )

#%% define face dataset class
class Faces(Dataset):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.images = sorted(Glob(folder))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        image = transform(image)
        return image

#%%  ### create dataset object  #####
ds = Faces(folder="cropped_faces/")

#%% define dataloader
dataloader = DataLoader(ds, batch_size=64, shuffle=True)#, num_workers=8)

#%% define weight initialization so weights have a smaller spread
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(tensor=m.weight.data, mean=0.0, std=0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(tensor=m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(tensor=m.bias.data, val=0)
        
        
#%% define discriminator model class
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(in_channels=64, out_channels=64*2,
                                             kernel_size=4, stride=2, padding=1, bias=False
                                             ),
                                   nn.BatchNorm2d(num_features=64*2),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(in_channels=64*2, out_channels=64*4,
                                             kernel_size=4, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(64*4),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(in_channels=64*4, out_channels=64*8, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(64*8),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(in_channels=64*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
                                   nn.Sigmoid()
                                   )
        self.apply(weight_init)
        
    def forward(self, input):
        return self.model(input)
        


#%%

from torchsummary import summary

discriminator = Discriminator().to(device)
summary(model=discriminator, input_data=torch.zeros(1, 3, 64, 64))
# %% define generator model
class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=64*8, kernel_size=4,
                               stride=2, padding=0, bias=False
                               ),
            nn.BatchNorm2d(num_features=64*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64*8, out_channels=64*4, kernel_size=4,
                               stride=2, padding=1, bias=False
                               ),
            nn.BatchNorm2d(num_features=64*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64*4, out_channels=64*2, kernel_size=4, stride=2, padding=1,
                               bias=False
                               ),
            nn.BatchNorm2d(num_features=64*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64*2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(weight_init)
        
    def forward(self, input):
        return self.model(input)


#%% summary of define model
generator = Generator().to(device)
summary(model=generator, input_data=torch.zeros(1, 100, 1, 1))


#%% define func to train discriminator

def discriminator_train_step(real_data, fake_data, d_optimizer, loss):
    d_optimizer.zero_grad()
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real.squeeze(), torch.ones(len(real_data)).to(device))
    error_real.backward()
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake.squeeze(), torch.zeros(len(fake_data)).to(device))
    error_fake.backward()
    d_optimizer.step()
    return error_real + error_fake

def generator_train_step(real_data, fake_data, g_optimizer, loss):
    g_optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = loss(prediction.squeeze(), torch.ones(len(real_data)).to(device))
    error.backward()
    g_optimizer.step()
    return error
 
 
 #%% generator and discriminator objs
discriminator = Discriminator().to(device)
generator = Generator().to(device)
d_optimizer = optim.Adam(params=discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
                         )
g_optimizer = optim.Adam(params=generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss = nn.BCELoss()

#%% run models during epochs
# loop through N epochs over the data
log = Report(55)
for epoch in range(25):
    N = len(dataloader)
    # load real dtata
    for i, images in enumerate(dataloader):
        real_data = images.to(device)
        # generate fake data
        fake_data = generator(torch.randn(len(real_data), 100, 1, 1).to(device)).to(device)
        fake_data = fake_data.detach()
        
        # train discriminator network
        d_loss = discriminator_train_step(real_data, fake_data, d_optimizer, loss)
       
       # generate new set of fake-data and train generator
        fake_data = generator(torch.randn(len(real_data), 100, 1, 1).to(device)).to(device)
        fake_data = fake_data#.detach()
        g_loss = generator_train_step(real_data, fake_data, g_optimizer, loss)
        
        # record losses
        log.record(epoch+(1+i)/N, d_loss=d_loss.item(), g_loss=g_loss.item(), end="\r")
    log.report_avgs(epoch+1)
log.plot_epochs(["d_loss", "g_loss"])

# %%
generator.eval()
noise = torch.randn(64, 100, 1, 1, device=device)
sample_images = generator(noise).detach().cpu()
grid = vutils.make_grid(sample_images, nrow=8, normalize=True)
show(grid.cpu().detach().permute(1,2,0), sz=10, title="Generated images")
# %% define dataset


