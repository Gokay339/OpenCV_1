import cv2
import mediapipe as mp

# Kamera kaynağını aç
kamera = cv2.VideoCapture(0)

# MediaPipe el tespiti modelini başlat
elModulu = mp.solutions.hands  # MediaPipe'ın el tespiti modülünü başlatır
eller = elModulu.Hands()  # El tespiti modelini başlatır
cizimAraci = mp.solutions.drawing_utils  # Tespit edilen el eklemlerini çizmek için kullanılır
parmakUclari = [4, 8, 12, 16, 20]  # Parmak uçlarının (baş parmak ve 4 parmak) ID'leri

# Parmak uçları ve eklem noktalarına göre elin açık ya da kapalı olduğunu algılamak için
# her parmağın uç noktasının pozisyonu üzerinden karar verilecek.
# Bas parmak için eklem fark

# https://forum.yazbel.com/t/opencv-ve-mediapipe-el-kombinasyolarinin-kosul-ifadeleri/19836ını hesaplamak için ekstra bir şey yapmaya gerek yok.

while True:
    basarili, img = kamera.read()  # Kamera görüntüsünü al
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Görüntüyü RGB formatına dönüştür
    
    sonuclar = eller.process(imgRGB)  # El tespiti modelini çalıştır
    # Görüntüde elleri tespit eder
    lmList = []
    if sonuclar.multi_hand_landmarks:
        # Eğer el tespit edilmişse, elin koordinatlarını döner.
        for elLms in sonuclar.multi_hand_landmarks:
            # El işaretçilerini çiz
            cizimAraci.draw_landmarks(img, elLms, elModulu.HAND_CONNECTIONS)
            # draw_landmarks: Elin eklem noktalarını ve bağlantı çizgilerini görüntü üzerine çizer.
            # mpEl.HAND_CONNECTIONS: Eklemler arasındaki bağlantıların çizilmesini sağlar.
            # elLms Eklem yerlerine koordinatlarına nokta koyar multi_hand_landmarks
            
            for id, lm in enumerate(elLms.landmark):
                # enumerate() fonksiyonu, elin tüm eklem noktalarını dönerken hem indeksini (id) hem de o noktaya ait koordinatları (lm) verir.
                # handLms.landmark elin 21 noktasının normalleştirilmiş koordinatlarını verir.
                h, w, _ = img.shape  # Görüntünün boyutlarını al
                cx, cy = int(lm.x * w), int(lm.y * h)  # Normalleştirilmiş koordinatları piksel cinsine çevir
                lmList.append([id, cx, cy])  # Koordinatları listeye ekle

    if len(lmList) != 0:
        parmaklar = []
        # Baş parmak için parmak ucu ve alt eklem kontrolü
        if lmList[parmakUclari[0]][1] < lmList[parmakUclari[0] - 1][1]:
            parmaklar.append(1)  # Baş parmak yukarıda
        else:
            parmaklar.append(0)  # Baş parmak aşağıda
            
        # Diğer dört parmak için aynı işlemi yap
        for id in range(1, 5):
            if lmList[parmakUclari[id]][2] < lmList[parmakUclari[id] - 2][2]:
                parmaklar.append(1)  # Parmak açık
            else:
                parmaklar.append(0)  # Parmak kapalı
        
        # Parmakların sayısını yazdır
        toplamParmak = parmaklar.count(1)  # Açık parmakları say
        cv2.putText(img, str(toplamParmak), (30, 125), (cv2.FONT_HERSHEY_PLAIN), 10, (255, 0, 0), 8)  # Ekrana yaz

    cv2.imshow("img", img)  # Görüntüyü ekranda göster
    cv2.waitKey(1)  # 1 ms bekle ve sonraki çerçeveye geç
