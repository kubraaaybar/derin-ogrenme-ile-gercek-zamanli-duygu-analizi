#Modelin Yüklenmesi ve Yüz Algılama İçin Gerekli Dosyanın Tanımlanması
import cv2
from keras.models import model_from_json
import numpy as np

# Modelin yüklenmesi
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Yüz algılama için Haar kaskadının yüklenmesi
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Görüntü Özelliklerinin Çıkarılması Fonksiyonu
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Kameranın açılması
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Kamera açılamadı!")
    exit()

#Etiketlerin Tanımlanması
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

#Yüz Algılama, Duygu Tahmini ve Görüntüleme
while True:
    ret, im = webcam.read() #Bu döngü, kameradan sürekli görüntü alır.
    if not ret:
        print("Kamera açılmadı veya görüntü alınamadı.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #yüzleri tespit eder ve her yüzün (x, y, genişlik, yükseklik) değerlerini döndürür.
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))#yüz görüntüsü 48x48 boyutuna yeniden boyutlandırılır
            img = extract_features(image)#normalleştirilir (extract_features) ve modele verilerek tahmin yapılır.
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()] #en yüksek olasılık değerine sahip sınıfın indeksini döndürür ve bu indeks labels sözlüğünde etikete çevrilir.
            cv2.putText(im, '%s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            #Algılanan yüzün üzerine tahmin edilen duygu etiketi yazdırılır. (p-10, q-10) konumunda duygu ismi belirtilir.

        # OpenCV ile görüntüyü gösterme
        cv2.imshow("Output", im)

        # 'q' tuşuna basıldığında döngüden çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print("Hata:", e)

# Kaynakların serbest bırakılması
webcam.release()
cv2.destroyAllWindows()