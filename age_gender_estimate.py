import numpy as np 
import cv2 
import onnx
import onnxruntime
from copy import deepcopy

class AgeGender:
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)
    
    def _preprocess_image(self, images):
        batch_size = len(images)
        batch = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
        for ind in range(batch_size):
            image = deepcopy(images[ind])
            image = cv2.resize(image, (224, 224), cv2.INTER_CUBIC)
            image = (image-127.5)/127.5
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            batch[ind] = image
        return batch

    def _post_process(self, results):
        ages, genders = [], []
        for ind in range(len(results[0])):
            age_prob = results[1][ind]
            gender_prob = results[2][ind]
            # Predict age
            prob_levels = age_prob > 0.5
            age = np.sum(prob_levels)
            # Predict gender
            gender_prob = np.exp(gender_prob)
            gender = np.argmax(gender_prob) # 0: Female, 1: Male
            
            ages.append(age)
            genders.append(gender)
        
        return ages, genders


    def predict_batch(self, images):
        ''' Predict age-gender of an image
        Params
            :image: List of images - np.ndarray
        Returns
            :age: List(age)
            :gender: List(gender) 
        '''
        images = self._preprocess_image(images)
        input_images = {"input": images}
        result = self.model.run(None, input_images)
        ages, genders = self._post_process(result)
        return ages, genders


    def predict(self, image):
        ''' Predict age-gender of an image
        Params
            :image: (np.ndarray) RGB image - aligned face
        Returns
            :age: List(age)
            :gender: List(gender) 
        '''
        image = self._preprocess_image([image])
        input_image = {"input": image}
        result = self.model.run(None, input_image)
        age, gender = self._post_process(result)

        return age, gender
        
if __name__ == "__main__":
    model = AgeGender("weights/age_gender_mb0.25_08112020.onnx")
    img = cv2.imread("/media/geneous/01D62877FB2A4900/Techainer/face/label_megaage_dataset/mega_age_gender/32550_A22_G0.jpg")
    age, gender = model.predict(img)
    print(age, gender)