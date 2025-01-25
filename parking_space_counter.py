import cv2
import pickle
import numpy as np

# Park alanı kontrol fonksiyonu
def ParkBoslukKontrol(imgg):
    bosAlan = 0  # Boş park alanı sayısını başlatıyoruz.
    
    # Park yerlerinin pozisyonları üzerinde iterasyon
    for pos in posList:
        x, y = pos
        img_crop = imgg[y:y+height, x:x+width]  # Görüntüden belirtilen alanı kesiyoruz.
        count = cv2.countNonZero(img_crop)  # Görüntüdeki beyaz piksellerin sayısını sayıyoruz.
        
        # Eğer sayım 150'den küçükse bu alan boş demektir
        if count < 150:
            color = (0, 255, 0)  # Boş alan, yeşil renk ile işaretleniyor
            bosAlan += 1  # Boş alan sayısını artırıyoruz.
        else:
            color = (0, 0, 255)  # Dolu alan, kırmızı renk ile işaretleniyor.
        
        # Belirtilen alana dikdörtgen çiziyoruz.
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, 2) 
        # Bu alandaki piksellerin sayısını ekliyoruz.
        cv2.putText(img, str(count), (x, y + height - 2), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
    
    # Toplam boş alan sayısını ekranda gösteriyoruz.
    cv2.putText(img, f"Bos Alan : {bosAlan}/{len(posList)}", (11, 16), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

# Park alanının genişliği ve yüksekliği
width = 27
height = 15

# Videoyu okuma
cap = cv2.VideoCapture("video.mp4")

# Daha önce kaydedilmiş park alanı pozisyonlarını yüklüyoruz.
with open("CarParkPos", "rb") as f:
    posList = pickle.load(f)

# Sonsuz döngü ile video kareleri üzerinde işlem yapıyoruz.
while True:
    basari, img = cap.read()  # Video karesi okunuyor.
    
    # Görüntüyü gri tonlamaya çeviriyoruz.
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gri görüntüyü bulanıklaştırıyoruz.
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    
    # Görüntüyü adaptif eşikleme yöntemi ile eşikliyoruz.
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    # Medyan bulanıklaştırma işlemi yapıyoruz.
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    # Görüntüyü dilate ediyoruz.
    imgDilate = cv2.dilate(imgMedian, np.ones((3, 3)), iterations=1)
    
    # Park yerlerini kontrol ediyoruz.
    ParkBoslukKontrol(imgDilate)
    
    # İşlenmiş görüntüyü ekranda gösteriyoruz.
    cv2.imshow("img", img)
    # cv2.imshow("img_gray", imgGray)
    # cv2.imshow("imgBlur", imgBlur)
    # cv2.imshow("imgThreshold", imgThreshold)
    # cv2.imshow("imgMedian", imgMedian)
    # cv2.imshow("imgDilate", imgDilate)
    cv2.waitKey(200)  # 200 ms bekleyerek bir sonraki kareyi işliyoruz.
