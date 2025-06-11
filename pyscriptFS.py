import cv2
import os

# Base directory for saving face images
BASE_SAVE_PATH = "face_database"

def capture_face(name):
    # Create a subfolder for the person
    person_folder = os.path.join(BASE_SAVE_PATH, name)
    os.makedirs(person_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = 1  # To save multiple images if needed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        cv2.imshow("Capture Face - Press 's' to save, 'q' to quit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to save the image
            image_path = os.path.join(person_folder, f"{name}_{count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved: {image_path}")
            count += 1  # Increment count for multiple images

        elif key == ord('q'):  # Press 'q' to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example Usage
name = input("Enter person's name: ")
capture_face(name)
