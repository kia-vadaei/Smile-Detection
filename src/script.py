
import sys
import torch
from smileNonSmile import SmileDetectionCNN
from preprocessing import Preprocessing
from mtcnn import MTCNN 
import torchvision.transforms as transforms
from PIL import Image
import cv2


def load_model():
    model = SmileDetectionCNN()
    model.load_state_dict(torch.load('./model/model.pth',
                                     map_location=torch.device('cuda' if torch.cuda.is_available()
                                                               else 'cpu')))
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')
    

def preprocess(detector, image):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
        ])
    
    result = detector.detect_faces(image)[0]

    x, y, width, height = result['box']
    side_length = max(width, height)
    x = x + max((width - side_length) // 2, 0)
    y = y + max((height - side_length) // 2, 0)
    cropped_face = image[y:min(y + side_length, image.shape[0]), x:min(x + side_length, image.shape[1])]

    return transform(Image.fromarray(cropped_face)).unsqueeze(0)


def run_cam():

    model = load_model()
    detector = MTCNN()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera Feed", frame)
        try:
            image = preprocess(detector, frame)

            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)

            label = 'Smile' if predicted.item() == 0 else 'Non-Smile'

        except:
            label = 'None'


        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Camera Feed", frame)

        print(f"Predicted label: {label}")
    

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    module_path = './src'
    if 'module_path' not in sys.path:
        sys.path.append(module_path)
        run_cam()


