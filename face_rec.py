import cv2
import os
from deepface import DeepFace

imagesPath = "images"
outputPath = "Rostros encontrados"

if not os.path.exists(outputPath):
    print('Carpeta creada:', outputPath)
    os.makedirs(outputPath)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
for imageName in os.listdir(imagesPath):
    image = cv2.imread(os.path.join(imagesPath, imageName))
    imageAux = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceClassif.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 255), 2)
    
    cv2.rectangle(image, (10, 5), (450, 25), (255, 255, 255), -1)
    cv2.putText(image, 'Presione s, para almacenar los rostros encontrados', (10, 20), 2, 0.5, (128, 0, 255), 1,
                cv2.LINE_AA)
    
    cv2.imshow('image', image)
    k = cv2.waitKey(0)

    if k == ord('s'):
        for (x, y, w, h) in faces:
            rostro = imageAux[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(outputPath, 'rostro_{}.jpg'.format(count)), rostro)
            count += 1
    elif k == 27:
        break

cv2.destroyAllWindows()

# Analizar rostros encontrados
for image in os.listdir(outputPath):
    im = cv2.imread(os.path.join(outputPath, image))
    results = DeepFace.analyze(im, actions=("gender", "age", "emotion"), enforce_detection=False)
    
    print("Resultados para", image)

    if results:
        print("Genero:", results[0]["gender"])
        print("Edad:", results[0]["age"])
        print("Emocion:", results[0]["dominant_emotion"])
    else:
        print("No se detectaron resultados.")
    
    print("-----------------------")
