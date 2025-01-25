import cv2
import pickle

# Park alanı genişliği ve yüksekliği
width = 27
height = 15

# Daha önce kaydedilmiş park alanı pozisyonları yükleniyor.
try:
    with open("CarParkPos", "rb") as f:
        posList = pickle.load(f)
except:
    posList = []  # Eğer dosya bulunamazsa, boş bir liste başlatıyoruz.

# Mouse tıklamaları ile park yerlerini seçme işlemi
def mouseClick(events, x, y, flags, params):
    
    if events == cv2.EVENT_LBUTTONDOWN:  # Sol tıklama
        posList.append((x, y))  # Yeni bir park alanı pozisyonu ekliyoruz.
    
    if events == cv2.EVENT_RBUTTONDOWN:  # Sağ tıklama
        for i, pos in enumerate(posList):
            x1, y1 = pos
            # Eğer tıklanan alan, mevcut park yerlerinden birinin içindeyse, o park yerini siliyoruz.
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)
    
    # Güncellenmiş park alanı pozisyonlarını kaydediyoruz.
    with open("CarParkPos", "wb") as f:
        pickle.dump(posList, f)

# Sonsuz döngü ile kullanıcıdan tıklama alıyoruz.
while True:
    # İlk görüntüyü yüklüyoruz (bu örnekte "first_frame.png").
    img = cv2.imread("first_frame.png")
    
    # Seçilen park alanlarını ekranda çiziyoruz.
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 0), 2)
    
    # Görüntüyü ekranda gösteriyoruz.
    cv2.imshow("img", img)
    # Mouse tıklamaları için geri çağırma fonksiyonu ayarlıyoruz.
    cv2.setMouseCallback("img", mouseClick)
    cv2.waitKey(1)  # 1 ms bekleyerek bir sonraki kareyi işliyoruz.

    
    
    
    
    
    
    
    