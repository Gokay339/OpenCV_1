import cv2
import mediapipe as mp
import time

öncekiZaman = 0
suankiZaman = 0

cap = cv2.VideoCapture(0)  

# MediaPipe El Algılama Modülünü Başlatma
mpEl = mp.solutions.hands  # MediaPipe kütüphanesindeki el algılama modülünü başlatır.

# Hands modülünü başlatıyoruz ve maksimum 2 el algılanacak şekilde ayarlıyoruz
eller = mpEl.Hands(max_num_hands=2)

mpCizim = mp.solutions.drawing_utils
# Tespit edilen el eklemlerini ve bağlantılarını çizmek için kullanılır.

while True:
    basari, img = cap.read()
    # `basari`: Görüntünün başarıyla okunup okunmadığını kontrol eder (True/False).
    # `img`: Kameradan okunan görüntüyü tutar.

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # MediaPipe, BGR yerine RGB formatında görüntü ister.
    # Bu nedenle, OpenCV'den gelen BGR formatını RGB formatına çeviriyoruz.

    sonuclar = eller.process(imgRGB)
    # MediaPipe elleri algılamak için RGB formatındaki görüntüyü işler.
    # Sonuç olarak tespit edilen ellerin eklem noktalarını döndürür.
    print(sonuclar) # burada classları görüyoruz ama biz içini görmek istiyoruz bu yüzden alttaki kodu deneyelim
    print(sonuclar.multi_hand_landmarks)
    # Eğer bir veya daha fazla el tespit edilirse, `multi_hand_landmarks` değişkeni,
    # her el için eklem noktalarının (landmark) koordinatlarını içerir.
    # Eğer el tespit edilmezse, `None` döner.

    if sonuclar.multi_hand_landmarks:
        for elLms in sonuclar.multi_hand_landmarks:
            # `multi_hand_landmarks`: Görüntüde tespit edilen her bir elin eklem noktalarını içerir.
            # Bu döngü, tespit edilen her el için çalışır.

            mpCizim.draw_landmarks(img, elLms, mpEl.HAND_CONNECTIONS)
            # Görüntü üzerine tespit edilen elin eklem noktalarını (landmark) ve bağlantılarını (HAND_CONNECTIONS) çizer.

            for id, lm in enumerate(elLms.landmark):
                #print(id,lm) # sırayla bütün noktaların koordinatlarını yazar
                # `enumerate()`: El eklemlerini sırasıyla döner.
                # `id`: Her eklem noktasının indeksini belirtir (0'dan 20'ye kadar).
                # `lm`: Her bir eklem noktasının normalleştirilmiş (0 ile 1 arasında) x ve y koordinatlarını döner.

                h, w, c = img.shape
                # `h`: Görüntünün yüksekliği (height).
                # `w`: Görüntünün genişliği (width).
                # `c`: Görüntünün renk kanallarının sayısı (3: BGR formatı).

                cx, cy = int(lm.x * w), int(lm.y * h)
                # Normalleştirilmiş x ve y koordinatlarını piksel değerine dönüştürüyoruz.
                # Bu işlem, eklem noktalarının görüntü üzerindeki kesin konumlarını bulmamızı sağlar.

                if id == 4:
                    cv2.circle(img, (cx, cy), 9, (255,0,0),cv2.FILLED)
                    # `id == 4`: Baş parmağın ucuna karşılık gelir.
                    # Bu durumda, baş parmağın ucu belirginleştirilerek görüntüye bir dolu daire çizilir.

    # fps
    suankiZaman = time.time()
    fps = 1/(suankiZaman-öncekiZaman)
    öncekiZaman=suankiZaman
    
    cv2.putText(img, "FPS : "+str(int(fps)), (5,45), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),2)
    cv2.imshow("img",img)
    cv2.waitKey(1)


    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()



