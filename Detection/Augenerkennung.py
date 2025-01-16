import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #  Medianfilter
    gray_blurred = cv2.medianBlur(gray, 15)

    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=30)

    #  Kreise erkennen und zeichnen
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)


    cv2.imshow('Eye Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Befreie die Videoquelle und schließe alle Fenster
cap.release()
cv2.destroyAllWindows()
