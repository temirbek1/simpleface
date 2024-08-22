import cv2 as cv
import numpy as np


detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

i = 0

offset = 50
name = input('номер пользователя: ')


video = cv.VideoCapture(1)
#'rtsp://test:Realmonitor@192.168.10.61:554/cam/realmonitor?channel=1&subtype=0'

while True:
    ret, im = video.read()

    gray = cv.cvtColor (im, cv.COLOR_BGR2GRAY)


    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    for (x, y, width, height) in faces:
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(gray.shape[1], x + width + offset)
        y2 = min(gray.shape[0], y + height + offset)

    
        i += 1
    
    
    
    
        cv.imwrite(f"dataset/face-{name}.{i}.jpg", gray[y1:y2, x1:x2])
        cv.rectangle(im, (x1, y1), (x2, y2), (225, 0, 0), 2)
        cv.imshow('im', im[y1:y2, x1:x2])
        cv.waitKey(100)
    if i > 30:
        video.release()
        cv.destroyAllWindows()
        break

    from PIL import Image
import os
import cv2 as cv

path = os.getcwd()


dataPath = os.path.join(path, 'dataSet')
if not os.path.exists(dataPath):
    os.makedirs(dataPath)


recognizer = cv.face.LBPHFaceRecognizer_create()
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')



def get_img(datapath):
    image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath) if os.path.isfile(os.path.join(datapath, f))]

    images = []
    labels = []


    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')

        image = np.array(image_pil, 'uint8')

        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", " "))

        faces = faceCascade.detectMultiScale(image)


        for (x, y, width, height) in faces:
            images.append(image[y: y + height, x: x + width])

            labels.append(nbr)
            cv.imshow("Adding faces to training set...", image[y: y + height, x: x + width])
            cv.waitKey(100)
    return images, labels




images, labels = get_img(dataPath)


if len(images) > 1:
    recognizer.train(images, np.array(labels))
else:
    print("Недостаточно данных")


trainer_path = os.path.join(path, 'trainer')
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)
recognizer.save(os.path.join(trainer_path, 'trainer.yml'))

cv.destroyAllWindows()
path = os.getcwd()


face_count = 0
face_count1 = 0

recognizer = cv.face.LBPHFaceRecognizer_create()



recognizer.read(os.path.join(path, 'trainer', 'trainer.yml'))


faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")


last_detected_face = None

cam = cv.VideoCapture('rtsp://test:Realmonitor@192.168.10.61:554/cam/realmonitor?channel=1&subtype=0')

font = cv.FONT_HERSHEY_SIMPLEX


while True:

    ret, im = cam.read()

    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv.CASCADE_SCALE_IMAGE)
    

    for (x, y, w, h) in faces:

        nbr_predicted, coord = recognizer.predict(gray[y:y+h, x:x+w])

        
        if nbr_predicted == 1:
 
            nbr_predicted = 'Temirbek'


            
            if last_detected_face is None or (abs(x - last_detected_face[0]) > 50 or abs(y - last_detected_face[1]) > 50):
                face_count += 1
                last_detected_face = (x, y)
                face_found = True
                

                
        if nbr_predicted == 2:
 
            nbr_predicted = 'Bek'
            
            if last_detected_face is None or (abs(x - last_detected_face[0]) > 500 or abs(y - last_detected_face[1]) > 500):
                face_count1 += 1
                last_detected_face = (x, y)
                face_found = True

 



        
        cv.putText(im, str(nbr_predicted), (x, y+h), font, 1.1, (0, 255, 0))
        cv.putText(im, str(face_count), (x, y), font, 1.1, (0, 255, 0))
        

        cv.imshow('Face recognition', im)
        
    



    
    if cv.waitKey(10) & 0xFF == ord('q'):
        break



cam.release()
cv.destroyAllWindows()
print(nbr_predicted, face_count)