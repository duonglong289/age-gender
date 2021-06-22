import glob 
import os 
img_path = glob.glob("./dataset/last_face_age_gender/*")
image_name = os.path.basename(img_path[0])
age = image_name.strip().split("_")[1].split("A")[1]
gender = image_name.strip().split("_")[2][1]
print(image_name)
print(age, gender)