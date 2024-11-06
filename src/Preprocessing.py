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
    
    def run(self, constant_size = (128, 128)):
        detector = MTCNN()
        input_folders = [os.path.join(self.input_main_path, 'smile'),
                            os.path.join(self.input_main_path, 'non_smile')]
        for indx, input_folder in enumerate(input_folders):
            if indx == 0:
                output_folder = self.output_smile
            else:
                output_folder = self.output_non_smile
                
            for filename in tqdm(os.listdir(input_folder)):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(input_folder, filename)
                    image = cv2.imread(image_path)

                    results = detector.detect_faces(image)

                    for i, result in enumerate(results):
                        x, y, width, height = result['box']
                        side_length = max(width, height)

                        x_new = x + (width - side_length) // 2
                        y_new = y + (height - side_length) // 2

                        x_new = max(0, x_new)
                        y_new = max(0, y_new)
                        x_end = min(x_new + side_length, image.shape[1])
                        y_end = min(y_new + side_length, image.shape[0])

                        cropped_face = image[y_new:y_end, x_new:x_end]

                        gray_cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

                        resized_gray_face = cv2.resize(gray_cropped_face, constant_size)

                        cropped_face_filename = f"{filename.split('.')[0]}_face_{i+1}.jpg"
                        cropped_face_path = os.path.join(output_folder, cropped_face_filename)
                        cv2.imwrite(cropped_face_path, resized_gray_face)
                        break