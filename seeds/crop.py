from keras.models import load_model
from PIL import Image, ImageOps
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def predict_webcam():
    # CAMERA can be 0 or 1 based on the default camera of your computer
    camera = cv2.VideoCapture(0)

    while True:
        # Grab the webcam image
        ret, frame = camera.read()

        # Define the region of interest (ROI) where gesture is detected
        roi = frame[100:400, 100:400]  # Increased size of the ROI

        # Resize the ROI to (224, 224) pixels
        roi_resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the model's input shape
        image_array = np.asarray(roi_resized, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image_array = (image_array / 127.5) - 1

        # Predict the model
        prediction = model.predict(image_array)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Remove leading/trailing whitespace

        # Display predicted class label inside the webcam frame
        cv2.putText(frame, class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw a rectangle around the ROI (blue color)
        cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)  # Increased size of the rectangle

        # Show the image in a window
        cv2.imshow("Webcam Image", frame)

        # Listen to the keyboard for presses
        keyboard_input = cv2.waitKey(1)

        # 27 is the ASCII for the esc key on your keyboard
        if keyboard_input == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

# For webcam prediction:
predict_webcam()
