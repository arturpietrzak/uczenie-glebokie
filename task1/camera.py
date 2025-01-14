import cv2
import torch
import torchvision.transforms as transforms
from skimage.feature import Cascade
from PIL import Image
from task1.GenderClassifierCNN import GenderClassifierCNN
from task1.SmilingClassifierResnet import SmilingClassifierResnet


def detect(frame, detector):
    detections = detector.detect_multi_scale(img=frame, scale_factor=1.2, step_ratio=1,
                                             min_size=(100, 100), max_size=(200, 200))
    boxes = []
    for detection in detections:
        x = detection['c']
        y = detection['r']
        w = detection['width']
        h = detection['height']

        padding = int(h * 0.6)

        # Adjust x, y, w, h for padding
        x_padded = max(0, x - padding)  # Ensure x doesn't go out of bounds
        y_padded = max(0, y - padding)
        w_padded = min(frame.shape[1] - x_padded, w + 2 * padding)
        h_padded = min(frame.shape[0] - y_padded, h + 2 * padding)

        # Adjust to 1:1.22 aspect ratio (178px/218px)
        new_h = int(w_padded * 1.22)
        y_adjustment = (h_padded - new_h) // 2

        # Ensure we don't go out of frame bounds
        y = max(0, y_padded + y_adjustment)
        h = min(new_h, frame.shape[0] - y)

        boxes.append((x_padded, y, w_padded, h))
    return boxes


def preprocess_face(frame, box, resnet=False):
    x, y, w, h = box
    face = frame[y:y + h, x:x + w]
    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    if resnet:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    return transform(face_pil).unsqueeze(0)


def draw(frame, boxes, gender_preds, smile_preds):
    for (x, y, w, h), gender_pred, smile_pred in zip(boxes, gender_preds, smile_preds):
        # Draw rectangle for face
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                              color=(255, 0, 0), thickness=2)

        # Prepare predictions text
        gender = "Male" if gender_pred > 0.5 else "Female"
        gender_conf = gender_pred if gender_pred > 0.5 else 1 - gender_pred
        gender_text = f"{gender} ({gender_conf:.2f})"

        smile = "Smiling" if smile_pred > 0.5 else "Not Smiling"
        smile_conf = smile_pred if smile_pred > 0.5 else 1 - smile_pred
        smile_text = f"{smile} ({smile_conf:.2f})"

        # Position and draw text
        gender_pos = (x, y - 25)  # Gender text position
        smile_pos = (x, y - 5)  # Smile text position

        # Add background rectangles for better text visibility
        (gender_w, gender_h), _ = cv2.getTextSize(gender_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        (smile_w, smile_h), _ = cv2.getTextSize(smile_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        cv2.rectangle(frame,
                      (gender_pos[0] - 2, gender_pos[1] - gender_h - 2),
                      (gender_pos[0] + gender_w + 2, gender_pos[1] + 2),
                      (255, 255, 255), -1)
        cv2.rectangle(frame,
                      (smile_pos[0] - 2, smile_pos[1] - smile_h - 2),
                      (smile_pos[0] + smile_w + 2, smile_pos[1] + 2),
                      (255, 255, 255), -1)

        # Draw text
        cv2.putText(frame, gender_text, gender_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, smile_text, smile_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def main():
    # Load both models
    gender_model = GenderClassifierCNN()
    smile_model = SmilingClassifierResnet()

    gender_model.load_model("./models/gender_classifier1.pth")
    smile_model.load_model("./models/smiling_classifier1.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gender_model = gender_model.to(device)
    smile_model = smile_model.to(device)

    gender_model.eval()
    smile_model.eval()

    # Load face detector
    file = "./face.xml"
    detector = Cascade(file)

    cap = cv2.VideoCapture(0)
    skip = 5
    i = 0
    boxes = []
    gender_predictions = []
    smile_predictions = []

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if i % skip == 0:
                boxes = detect(frame, detector)
                gender_predictions = []
                smile_predictions = []

                # Process each detected face
                for box in boxes:
                    gender_tensor = preprocess_face(frame, box).to(device)
                    smile_tensor = preprocess_face(frame, box).to(device)

                    # Get predictions from both models
                    gender_pred = gender_model(gender_tensor).item()
                    smile_pred = smile_model(smile_tensor).item()

                    gender_predictions.append(gender_pred)
                    smile_predictions.append(smile_pred)

            draw(frame, boxes, gender_predictions, smile_predictions)
            cv2.imshow('Face Analysis', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()