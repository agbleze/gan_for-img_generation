
#%%
from torch_snippets import *
from torchvision.utils import make_grid
from PIL import Image
import torchvision
from torchvision import transforms
import torchvision.utils as vutils

#%%
female_images = Glob()

img_folder = "C:/Users/agbji/Documents/codebase/gan_for_img_generation/male_female_face_images/"
female_images = Glob(img_folder + "females/*.jpg") 
male_images = Glob(img_folder + "males/*.jpg")


#%%

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_images(folder):
    pass

