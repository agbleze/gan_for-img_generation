
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
        gender = np.where("female" in image_path, 1, 0)
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