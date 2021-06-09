import os
from PIL import Image
import numpy as np
import cv2 as cv
# Multiple (min 10) shots clicked continously using a tripod (stable images), works excellently well in low light/ night setting. 
#(Make sure the noise reduction is off on the camera : Helps in sharp final_image)
image_folder= "Input"
if not os.path.exists(image_folder):
    print("ERROR {} not found!".format(image_folder))
    exit()

file_list = sorted(os.listdir(image_folder))
file_list = [os.path.join(image_folder, x)
              for x in file_list if x.endswith(('.JPG', '.png','.bmp','.jpeg'))]
image=Image.open(file_list[0])
im=np.array(image,dtype=np.float32)
for i in range(1,len(file_list)):
    currentimage=Image.open(file_list[i])
    im += np.array(currentimage, dtype=np.float32)
im /= len(file_list)*0.5 # lows brightness by 50% on stacked values for each pixel
final_image = Image.fromarray(np.uint8(im.clip(0,255)))
final_image.save('semi_final_image.jpg', 'JPEG')
img= cv.imread('all_averaged.jpg')
dst = cv.fastNlMeansDenoisingColored(img,None,3,3,7,21)
cv.imwrite('final_image.jpg', dst)