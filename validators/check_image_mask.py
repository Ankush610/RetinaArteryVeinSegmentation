import cv2, numpy as np
mask = cv2.imread("./dataset/avseg/masks/001_G.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
print(np.unique(mask.reshape(-1,3), axis=0))