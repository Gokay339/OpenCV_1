import cv2
import mediapipe as mp
import time
pTime = 0  # Önceki zaman (FPS hesaplaması için başlangıç değeri)

# MediaPipe Pose modülünü başlat
mpPose = mp.solutions.pose  # Poz tahmini için MediaPipe Pose sınıfını kullanıyoruz
pose = mpPose.Pose()  # Pose nesnesini oluştur (varsayılan parametreler ile)

# MediaPipe çizim yardımcı aracı
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("video3.mp4")

while True:
    success, img = cap.read()  # Videodan bir kare oku
    imgRGB = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    # MediaPipe ile poz tahmini işlemi yap
    results = pose.process(imgRGB)
    print(results.pose_landmarks)  # Tespit edilen poz işaretlerini ekrana yazdır
    
    # Eğer poz işaretleri tespit edilmişse
    if results.pose_landmarks:
        # Poz işaretlerini görüntü üzerine çiz
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # Tüm poz işaretlerinin koordinatlarını al
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, _ = img.shape  # Görüntünün yüksekliği, genişliği ve kanal sayısını al
            cx, cy = int(lm.x * w), int(lm.y * h)  # Poz işaretinin x ve y koordinatlarını hesapla

            # ID 14 olan nokta (örneğin, sağ dirsek için) üzerine mavi bir daire çiz
            if id == 14:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # FPS hesaplama (şu anki zaman - önceki zaman farkı ile hesaplanır)
    cTime = time.time()  # Şu anki zamanı al
    fps = 1 / (cTime - pTime)  # FPS hesapla (1 saniye / zaman farkı)
    pTime = cTime  # Önceki zamanı güncelle

    # FPS bilgisini görüntü üzerine yaz
    cv2.putText(img, "FPS : " + str(int(fps)), (5, 45), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    # Görüntüyü ekranda göster
    cv2.imshow("img", img)

# Video işleme tamamlandığında pencereyi kapat
cap.release()  # Video kaynağını serbest bırak
cv2.destroyAllWindows()  # Tüm OpenCV pencerelerini kapat




































