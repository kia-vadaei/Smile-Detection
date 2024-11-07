import os
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from mtcnn import MTCNN

class Preprocessing:
    def __init__(self, output_smile, output_non_smile, input_main_path):
        
        if not os.path.exists(output_smile):
            os.makedirs(output_smile)

        if not os.path.exists(output_non_smile):
            os.makedirs(output_non_smile)


        self.output_smile = output_smile
        self.output_non_smile = output_non_smile
        self.input_main_path = input_main_path
        self.detector = MTCNN()
    
    def run_on_dataset(self, constant_size = (128, 128)):
        input_folders = [os.path.join(self.input_main_path, 'smile'),
                            os.path.join(self.input_main_path, 'non_smile')]
        for indx, input_folder in enumerate(input_folders):

            if indx == 0:
                output_folder = self.output_smile
                desc = 'smile class is being preprocessed'

            else:
                output_folder = self.output_non_smile
                desc = 'non-smile class is being preprocessed'

        
            for filename in tqdm(os.listdir(input_folder), desc=desc):
                if filename.endswith(('.jpg',)):
                    image_path = os.path.join(input_folder, filename)
                    image = cv2.imread(image_path)

                    results = self.detector.detect_faces(image)

                    for i, result in enumerate(results):
                        x, y, width, height = result['box']
                        side_length = max(width, height)

                        x = x + max((width - side_length) // 2, 0)
                        y = y + max((height - side_length) // 2, 0)

                        cropped = image[y:min(y + side_length, image.shape[0]), x:min(x + side_length, image.shape[1])]
                        
                        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

                        resized_gray_face = cv2.resize(cropped_gray, constant_size)

                        cropped_face_filename = f"{filename.split('.')[0]}_{i+1}.jpg"

                        cv2.imwrite(os.path.join(output_folder, cropped_face_filename),
                                    resized_gray_face)
                        
    def run_on_image(self, image_path, constant_size = (128, 128)):

        image = cv2.imread(image_path)
        results = self.detector.detect_faces(image)

        for i, result in enumerate(results):
            x, y, width, height = result['box']
            side_length = max(width, height)

            x = x + max((width - side_length) // 2, 0)
            y = y + max((height - side_length) // 2, 0)

            cropped = image[y:min(y + side_length, image.shape[0]), x:min(x + side_length, image.shape[1])]
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            resized_gray_face = cv2.resize(cropped_gray, constant_size)
            return resized_gray_face
