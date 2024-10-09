import face_recognition
import cv2
import numpy as np

#Get a reference to the defualt webcam
video_capture = cv2.VideoCapture(0)

#Load a sample picture and learn how to recognize it
your_image = face_recognition.load_image_file("/home/jarvis/Downloads/Jyupter/face_detection-master/Tony.png")

if your_image is None:
    print("Failed to load the image.")  
else:
    print("Image loaded successfully with shape:", your_image.shape)
    
your_face_encoding = face_recognition.face_encodings(your_image)[0]

print(your_face_encoding)   

#Create an   array of known face encodings
known_face_encodings = [
    your_face_encoding
]

known_face_names = [
    "Brijesh"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    #Grab a single frame of video
    ret, frame = video_capture.read()
    
    if process_this_frame:
        #Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        #Convert the image from BGR(which OpenCV uses) to RGB(face_recgnition)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        #Final all the faces and facial encodings in the current frame of the video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)