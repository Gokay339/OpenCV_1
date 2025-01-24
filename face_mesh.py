import cv2
import time
import mediapipe as mp

# Kamera akışını başlatma
cap = cv2.VideoCapture(0)

# Mediapipe yüz mesh modülü
mpYuzMesh = mp.solutions.face_mesh  # Yüz mesh işlemi için modülü çağırıyoruz
yuzMesh = mpYuzMesh.FaceMesh(max_num_faces=2)  # Aynı anda maksimum 1 yüz algılaması için ayar
mpCizim = mp.solutions.drawing_utils  # Çizim araçları
cizimStili = mpCizim.DrawingSpec(thickness=1, circle_radius=1)  # Çizim stili: Çizgi kalınlığı ve nokta boyutu

# FPS hesaplaması için zaman değişkenleri
oncekiZaman = 0

# Sonsuz döngü: Kamera karelerini sürekli işlemek için
while True:
    basarili, img = cap.read()  # Kameradan bir kare oku
    if not basarili:  # Eğer görüntü alınamazsa döngüden çık
        break

    # RGB formatına dönüştürme (Mediapipe, RGB formatında çalışır)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Yüz mesh işlemi
    sonuclar = yuzMesh.process(imgRGB)
    print(sonuclar.multi_face_landmarks)  # Algılanan yüz noktalarını konsola yazdır
    
    # Eğer yüz algılandıysa, her bir yüz için işlem yap
    if sonuclar.multi_face_landmarks:
        for yuzNoktalar in sonuclar.multi_face_landmarks:
            # Yüz mesh çizimini görüntüye ekle
            mpCizim.draw_landmarks(
                img, yuzNoktalar, mpYuzMesh.FACEMESH_TESSELATION, cizimStili, cizimStili
            )
        
        # Her bir yüz noktasını işleme
        for id, nokta in enumerate(yuzNoktalar.landmark):
            h, w, _ = img.shape  # Görüntünün boyutlarını al
            cx, cy = int(nokta.x * w), int(nokta.y * h)  # Oranlı koordinatları piksel değerlerine çevir
            print([id, cx, cy])  # Nokta id'si ve koordinatlarını konsola yazdır

    # FPS hesaplama
    suankiZaman = time.time()  # Şimdiki zaman
    fps = 1 / (suankiZaman - oncekiZaman)  # FPS formülü
    oncekiZaman = suankiZaman  # Şimdiki zamanı önceki zaman olarak sakla
    cv2.putText(img, f"FPS: {int(fps)}", (10, 65), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  

    # Görüntüyü ekranda göster
    cv2.imshow("img", img)
    
    # Görüntü akışını durdurmak için kısa bekleme
    cv2.waitKey(1)
