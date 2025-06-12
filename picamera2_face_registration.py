#!/usr/bin/env python3
"""
Face Registration using Picamera2 - Optimized for Pi Camera Rev 1.3
This should work reliably on your setup
"""

import cv2
import face_recognition
import pickle
import os
import time
import numpy as np
from picamera2 import Picamera2

def register_faces_picamera2():
    print("=== Face Registration with Picamera2 ===")
    
    # Load existing database
    known_encodings = []
    known_names = []
    
    if os.path.exists('face_database.pkl'):
        with open('face_database.pkl', 'rb') as f:
            data = pickle.load(f)
            known_encodings = data['encodings']
            known_names = data['names']
        print(f"Loaded {len(known_names)} existing faces")
    
    # Initialize Picamera2
    print("Initializing Picamera2...")
    try:
        picam2 = Picamera2()
        
        # Configure camera for optimal performance
        config = picam2.create_still_configuration(
            main={"size": (640, 480)},
            lores={"size": (320, 240)},  # Lower resolution for preview
            display="lores"
        )
        picam2.configure(config)
        
        # Start camera
        picam2.start()
        time.sleep(2)  # Allow camera to warm up
        print("✓ Camera initialized successfully")
        
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        print("Make sure camera is properly connected and enabled")
        return
    
    print("\nInstructions:")
    print("- Position face clearly in frame")
    print("- Press SPACE to capture and register face")
    print("- Press 'q' to quit and save database")
    print("- Press 'p' to preview current frame")
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Convert from RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add overlay information
            cv2.putText(frame, f"Registered Faces: {len(known_names)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Register | Q: Quit | P: Preview", (10, 460), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Detect faces and draw rectangles
            try:
                # Use smaller image for faster face detection
                small_frame = cv2.resize(frame, (320, 240))
                face_locations = face_recognition.face_locations(small_frame, model="hog")
                
                # Scale back up face locations
                for (top, right, bottom, left) in face_locations:
                    top *= 2; right *= 2; bottom *= 2; left *= 2
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected", (left, top-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
            except Exception as e:
                print(f"Face detection error: {e}")
            
            # Show frame
            cv2.imshow('Face Registration - Picamera2', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE key - Register face
                print("\n--- Registering Face ---")
                name = input("Enter person's name: ").strip()
                
                if name:
                    try:
                        # Capture a fresh frame for encoding
                        capture_frame = picam2.capture_array()
                        if len(capture_frame.shape) == 3 and capture_frame.shape[2] == 3:
                            # Convert RGB to BGR for face_recognition (it expects RGB)
                            rgb_frame = cv2.cvtColor(capture_frame, cv2.COLOR_RGB2BGR)
                            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                        else:
                            rgb_frame = capture_frame
                        
                        # Find face encodings
                        face_encodings = face_recognition.face_encodings(rgb_frame, model="small")
                        
                        if face_encodings:
                            known_encodings.append(face_encodings[0])
                            known_names.append(name)
                            print(f"✓ Face registered successfully for '{name}'")
                            print(f"Total faces in database: {len(known_names)}")
                        else:
                            print("✗ No face detected. Please position face clearly and try again.")
                            
                    except Exception as e:
                        print(f"Error processing face: {e}")
                else:
                    print("Name cannot be empty. Please try again.")
            
            elif key == ord('p') or key == ord('P'):  # Preview mode
                print("Preview mode - press any key to continue registration")
                cv2.waitKey(0)
                
            elif key == ord('q') or key == ord('Q'):  # Quit
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        picam2.stop()
        cv2.destroyAllWindows()
    
    # Save database
    if known_encodings and known_names:
        try:
            data = {'encodings': known_encodings, 'names': known_names}
            with open('face_database.pkl', 'wb') as f:
                pickle.dump(data, f)
            print(f"\n✓ Database saved successfully!")
            print(f"Registered faces ({len(known_names)}):")
            for i, name in enumerate(known_names, 1):
                print(f"  {i}. {name}")
        except Exception as e:
            print(f"Error saving database: {e}")
    else:
        print("No faces were registered.")
    
    print("Face registration completed!")

if __name__ == "__main__":
    # Check if required libraries are available
    try:
        import face_recognition
        import cv2
        import pickle
        from picamera2 import Picamera2
        print("✓ All required libraries loaded successfully")
    except ImportError as e:
        print(f"✗ Missing library: {e}")
        print("Install missing libraries:")
        print("sudo apt install python3-picamera2")
        print("pip3 install face_recognition opencv-python")
        exit(1)
    
    register_faces_picamera2()