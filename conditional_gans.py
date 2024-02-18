
#%%
from torch_snippets import *
from torchvision.utils import make_grid
from PIL import Image
import torchvision
from torchvision import transforms
import torchvision.utils as vutils

#%%

img_folder = "C:/Users/agbji/Documents/codebase/gan_for_img_generation/male_female_face_images/"
#female_images = Glob(img_folder + "females/*.jpg") 
#male_images = Glob(img_folder + "males/*.jpg")

cropped_female_faces_output_folder = "C:/Users/agbji/Documents/codebase/gan_for_img_generation/cropped_faces_females/"
cropped_male_faces_output_folder = "C:/Users/agbji/Documents/codebase/gan_for_img_generation/cropped_faces_males/"

#%%

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_images(folder, cropped_output_folder):
    images = Glob(folder+"*.jpg")
    for i in range(len(images)):
        img = read(images[i], 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img2 = img[y:(y+h), x:(x+w),:]
            cv2.imwrite(cropped_output_folder + str(i)+".jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

#%%
crop_images(folder=img_folder + "females/", cropped_output_folder=cropped_female_faces_output_folder)
crop_images(folder=img_folder + "males/", cropped_output_folder=cropped_male_faces_output_folder)


# %%

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

#%%
class Faces(Dataset):
    def __init__(self, folders):
        super().__init__()
        self.folderfemale = folders[0]
        self.foldermale = folders[1]
        self.images = sorted(Glob(self.folderfemale)) + sorted(Glob(self.foldermale))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        image = transform(image)
        gender = np.where("female" in [image_path], 1, 0)
        return image, torch.tensor(gender).long()
    
    
#%% define ds dataset and data loader
ds = Faces(folders=[cropped_female_faces_output_folder, cropped_male_faces_output_folder])

dataloader = DataLoader(dataset=ds, batch_size=64, shuffle=True)

#%%  denine weight
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

#%% define discriminator model architecture
class Discriminator(nn.Module):
    def __init__(self, emb_size=32):
        super(Discriminator, self).__init__()
        self.emb_size = 32
        self.label_embeddings = nn.Embedding(num_embeddings=2, embedding_dim=self.emb_size)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=64*2, out_channels=64*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=64*4, out_channels=64*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=64*8, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Flatten()            
        )
        self.model2 = nn.Sequential(
                                    nn.Linear(in_features=288, out_features=100),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Linear(in_features=100, out_features=1),
                                    nn.Sigmoid()
                                )
        self.apply(weights_init)
        
    def forward(self, input, labels):
        x = self.model(input)
        y = self.label_embeddings(labels)
        input = torch.cat(tensors=[x,y], dim=1)
        final_out = self.model2(input)
        return final_out
        
        
#%% summary of defined model
from torchsummary import summary

#%%
discriminator = Discriminator().to(device)
summary(discriminator, torch.zeros(32, 3, 64, 64).to(device), torch.zeros(32).long().to(device))


#%% define generator model arch

class Generator(nn.Module):
    def __init__(self, emb_size=32):
        super(Generator, self).__init__()
        self.emb_size = emb_size
        self.label_embedding = nn.Embedding(num_embeddings=2, embedding_dim=self.emb_size)
        
        self.model = nn.Sequential(nn.ConvTranspose2d(in_channels=100+self.emb_size,
                                                      out_channels=64*8, kernel_size=4, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(num_features=64*8),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(in_channels=64*8, out_channels=64*4, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=64*4),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(in_channels=64*4, out_channels=64*2, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=64*2),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(in_channels=64*2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.Tanh()
                                   )
        self.apply(weights_init)
        
    def forward(self, input_noise, labels):
        label = self.label_embedding(labels).view(len(labels), self.emb_size, 1, 1)
        input = torch.cat([input_noise, label], 1)
        return self.model(input)
    
#%%  ## summary
generator = Generator().to(device)
summary(model=generator, input_data=(torch.zeros(32, 100, 1, 1).to(device), torch.zeros(32).long().to(device)))

def noise(size):
    n = torch.randn(size, 100, 1, 1, device=device)
    return n.to(device)


#%% func to train discriminator

def discriminator_train_step(real_data, real_labels, fake_data, fake_labels):
    d_optimizer.zero_grad()
    prediction_real = discriminator(real_data, real_labels)
    error_real = loss(prediction_real, torch.ones(len(real_data), 1).to(device))
    error_real.backward()
    
    prediction_fake = discriminator(fake_data, fake_labels)
    error_fake = loss(prediction_fake, torch.zeros(len(fake_data), 1).to(device))
    error_fake.backward()
    d_optimizer.step()
    return error_real + error_fake


#%% func to train generator

def generator_train_step(fake_data, fake_labels):
    g_optimizer.zero_grad()
    prediction = discriminator(fake_data, fake_labels)
    error = loss(prediction, torch.ones(len(fake_data), 1).to(device))
    error.backward()
    g_optimizer.step()
    return error


#%% define the generator and discriminator models

discriminator = Discriminator().to(device)
generator = Generator().to(device)
loss = nn.BCELoss()
d_optimizer = optim.Adam(params = discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(params=generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
fixed_noise = torch.randn(64, 100, 1, 1, device=device)
fixed_fake_labels = torch.LongTensor([0]*(len(fixed_noise)//2) + [1]*(len(fixed_noise)//2)).to(device)
loss = nn.BCELoss()
n_epochs = 25
img_list = []

#%% train model
log = Report(n_epochs)
for epoch in range(n_epochs):
    for bx, (images, labels) in enumerate(dataloader):
        N = len(dataloader)
        real_data, real_labels = images.to(device), labels.to(device)
        fake_labels = torch.LongTensor(np.random.randint(0, 2, len(real_data))).to(device)
        fake_data = generator(noise(len(real_data)), fake_labels)
        fake_data = fake_data.detach()
        
        # train discriminator
        d_loss = discriminator_train_step(real_data, real_labels, fake_data, fake_labels)
        
        # regenerate fake images and fake labels and train generator
        fake_labels = torch.LongTensor(np.random.randint(low=0, high=1, size=len(real_data))).to(device)
        fake_data = generator(noise(len(real_data)), fake_labels).to(device)
        g_loss = generator_train_step(fake_data, fake_labels)
        
        # log metrics
        pos = epoch + (1+bx) / N
        log.record(pos, d_loss=d_loss.detach(), g_loss=g_loss.detach(), end="\r")
    log.report_avgs(epoch+1)
log.plot_epochs(["d_loss", "g_loss"])    
    
#%% generta male and female images
with torch.no_grad():
    fake = generator(fixed_noise, fixed_fake_labels).detach().cpu()
    imgs = vutils.make_grid(fake, padding=2, normalize=True).permute(1,2,0)
    img_list.append(imgs)
    show(imgs, sz=10)



# %%
