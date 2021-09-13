import cv2
import os
import numpy as np

img_path = os.path.join(os.getcwd(), "_out", "episode_0000", "CameraRGBSemanticSegmentation", "000056.png")
print(img_path)
img = cv2.imread(img_path)

lower_black = np.array([0,0,4], dtype = "uint16")
upper_black = np.array([0,0,4], dtype = "uint16")
black_mask = cv2.inRange(img, lower_black, upper_black)

# contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0,255,0), 3)

# for cnt in contours:
#     x,y,w,h = cv2.boundingRect(cnt)
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)

cv2.imshow("New", black_mask)
cv2.waitKey(0)
