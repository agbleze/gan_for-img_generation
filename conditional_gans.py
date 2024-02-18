
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

