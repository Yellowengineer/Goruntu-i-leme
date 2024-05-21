import cv2

# Gerçek dünya boyutları (örneğin, telefonun gerçek boyutları)
object_width_cm =  5# cm
object_height_cm = 15# cm

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Görüntüyü gri tonlamaya dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Görüntüdeki kenarları tespit et
    edges = cv2.Canny(gray, 50, 150)

    # Kenarlar arasındaki konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # En büyük konturu seç
    largest_contour = max(contours, key=cv2.contourArea)

    # Konturu çiz
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Ölçek faktörünü hesapla (piksel cinsinden gerçek dünya boyutlarına dönüştürmek için)
    scale_factor_width = frame.shape[1] / object_width_cm
    scale_factor_height = frame.shape[0] / object_height_cm

    # Sınırlayıcı kutunun gerçek dünya boyutlarını hesapla (cm cinsinden)
    object_width_real = w / scale_factor_width
    object_height_real = h / scale_factor_height

    # Çıktıya boyutları ekle
    cv2.putText(frame, "Width: {:.2f} cm".format(object_width_real), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Height: {:.2f} cm".format(object_height_real), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Sonuçları göster
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
