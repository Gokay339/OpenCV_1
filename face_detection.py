import cv2
import mediapipe as mp

# Video dosyasını açma
video = cv2.VideoCapture(0)

# Mediapipe yüz algılama modülü
mpYuzAlgilama = mp.solutions.face_detection  # Yüz algılama için modülü çağırıyoruz
yuzAlgilama = mpYuzAlgilama.FaceDetection(0.20)  # Algılama hassasiyeti belirleniyor (0.20 = %20 hassasiyet)
mpCizim = mp.solutions.drawing_utils  # Çizim araçları

# Sonsuz döngü: Video karelerini sürekli işlemek için
while True:
    basarili, goruntu = video.read()  # Videodan bir kare oku

    # RGB formatına dönüştürme (Mediapipe, RGB formatında çalışır)
    goruntuRGB = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
    
    # Yüz algılama işlemi
    sonuclar = yuzAlgilama.process(goruntuRGB)
    
    # Algılanan yüzleri konsola yazdır
    print(sonuclar.detections)
    
    # Eğer yüz algılandıysa, her bir yüz için işlem yap
    if sonuclar.detections:
        for id, algilama in enumerate(sonuclar.detections):
            # Yüzün çerçeve koordinatlarını al (oranlı değerler)
            cerceveC = algilama.location_data.relative_bounding_box
            yukseklik, genislik, _ = goruntu.shape  # Görüntünün boyutlarını al
            
            # Oranlı değerleri gerçek piksel koordinatlarına çevir
            cerceve = (int(cerceveC.xmin * genislik), int(cerceveC.ymin * yukseklik),
                       int(cerceveC.width * genislik), int(cerceveC.height * yukseklik))
            
            # Yüzün çevresine dikdörtgen çiz (sarı renkli)
            cv2.rectangle(goruntu, cerceve, (0, 255, 255), 2)
    
    # Görüntüyü ekranda göster
    cv2.imshow("Goruntu", goruntu)
    
    # Görüntü akışını durdurmak için kısa bekleme
    cv2.waitKey(10)


























































































































































































































































