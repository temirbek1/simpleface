import cv2 as cv
import os

# Задаем путь
path = os.getcwd()

face_count = 0
face_count1 = 0

# Создаем объект распознавателя лиц LBPH
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(path, 'trainer', 'trainer.yml'))

# Загружаем каскад Хаара для распознавания лиц
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

last_detected_face = None

# Подключаемся к RTSP-потоку
cam = cv.VideoCapture('rtsp://test:Realmonitor@192.168.10.61:554/cam/realmonitor?channel=1&subtype=0')

if not cam.isOpened():
    print("Не удалось подключиться к RTSP потоку")
    exit()

font = cv.FONT_HERSHEY_SIMPLEX

while True:
    ret, im = cam.read()

    if not ret:
        print("Не удалось получить кадр")
        break

    # Преобразуем изображение в оттенки серого
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    
    # Обнаруживаем лица на изображении
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        nbr_predicted, coord = recognizer.predict(gray[y:y+h, x:x+w])
        
        if nbr_predicted == 1:
            nbr_predicted = 'Temirbek'
            if last_detected_face is None or (abs(x - last_detected_face[0]) > 50 or abs(y - last_detected_face[1]) > 50):
                face_count += 1
                last_detected_face = (x, y)
                
        elif nbr_predicted == 2:
            nbr_predicted = 'Bek'
            if last_detected_face is None or (abs(x - last_detected_face[0]) > 50 or abs(y - last_detected_face[1]) > 50):
                face_count1 += 1
                last_detected_face = (x, y)

        # Отображаем имя и количество обнаружений на изображении
        cv.putText(im, str(nbr_predicted), (x, y+h), font, 1.1, (0, 255, 0))
        cv.putText(im, str(face_count), (x, y), font, 1.1, (0, 255, 0))
        
    # Отображаем кадр
    cv.imshow('Face recognition', im)
    
    # Завершаем работу программы при нажатии клавиши 'q'
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
