import cv2
import numpy as np

img = cv2.imread("path_to_your_image.jpg")

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked coordinates: ({x}, {y})")
        # Optionally, display the coordinates on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f'({x}, {y})', (x, y), font, 0.5, (255, 0, 0), 1)
        cv2.imshow('Image', img)

cv2.imshow('Image', img)
cv2.setMouseCallback('Image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()