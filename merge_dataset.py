import glob
import cv2
import numpy as np
import os

class AllAgeFace:
    '''
    Format name: %05dA%02d.jpg
        00000 - 07380: female
        07381 - 13321: male
        {id}A{age}.jpg
    '''
    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, "original_images")
        self.abs_image_paths = []
        self.image_names = []
        self.image_ages = []
        self.image_genders = []
    
    def _process_data(self):
        list_images = glob.glob(os.path.join(self.image_dir, "*.jpg"))
        for path in list_images:
            abs_image_path = os.path.abspath(path)
            image_name = os.path.basename(path)
            image_age = image_name.split(".")[0][-2:]
            image_gender = image_name.split(".")[0][:5]
        
            self.abs_image_paths.append(abs_image_path)
            self.image_names.append(image_name)
            self.image_ages.append(image_age)
            self.image_genders.append(image_gender)
        

    def get_dataset(self):
        self._process_data()
        return {
            'path': self.abs_image_paths,
            'name': self.image_names,
            'age': self.image_ages,
            'gender': self.image_genders
        }
    
class MegaAgeAsian:
    '''
    MegaAgeAsian
        |----list
        |----|---test_age.txt
        |----|---test_name.txt
        |----|---train-age.txt  : train image label
        |----|---train-name.txt : train image name
        |----test               : test image 
        |----train              : train image
    '''
    def __init__(self, data_dir):
        self.abs_image_paths = []
        self.image_names = []
        self.image_ages = []
        self.image_genders []

    
        
    def get_label(self, image_file, label_file):
        with open(image_file, "r") as f:
            image_names = f.readlines()

        with open(label_file, "r") as f:
            labels = f.readlines()
        
        return list(zip(image_names, labels))

    def _process_data(self, data_dir, type_data):
        path_images = glob.glob(os.path.join(data_dir, type_data, "*"))
        name_path = os.path.join(data_dir, 'list', '{}_age.txt')
        age_path = os.path.join(data_dir, 'list', '{}_name.txt')
        ages = open(age_path, "r").readlines()
        names = open(name_path, "r").readlines()


    def merge_train_test(self):
        pass

