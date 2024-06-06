import cv2
import numpy as np

# Web kamerayı başlat
cap = cv2.VideoCapture(0)

# Gerçek dünya birimlerini pixel birimlerine dönüştürmek için kalibrasyon faktörü
# Burada "kalibrasyon_factor" olarak kullanılan değeri, kendi kameranız ve ortama göre kalibre etmelisiniz
kalibrasyon_factor = 37.79  # Örnek değer (1 cm = 37.79 pixel)

def nothing(x):
    pass

# Pencere ve trackbar'ların isimleri
window_name = 'Object Measurement'
trackbar1_name = 'Min Threshold'
trackbar2_name = 'Max Threshold'

# Pencere oluştur ve trackbar'ları ekle
cv2.namedWindow(window_name)
cv2.createTrackbar(trackbar1_name, window_name, 100, 255, nothing)
cv2.createTrackbar(trackbar2_name, window_name, 200, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Görüntüyü bulanıklaştır
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kenarları tespit et
    threshold1 = cv2.getTrackbarPos(trackbar1_name, window_name)
    threshold2 = cv2.getTrackbarPos(trackbar2_name, window_name)
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    # Konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Konturun çevresini bul
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        # Konturun bounding box'ını bul
        x, y, w, h = cv2.boundingRect(approx)
        
        # Gerçek boyutları hesapla
        width = w / kalibrasyon_factor
        height = h / kalibrasyon_factor
        
        # Eğer boyutlar 2cm'den büyükse
        if width >= 2 and height >= 2:
            # Konturu ve bounding box'ı çiz
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{width:.2f}cm x {height:.2f}cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Sonuçları göster
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizle
cap.release()
cv2.destroyAllWindows()
