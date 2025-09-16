import cv2
import numpy as np
import sys

if len(sys.argv) != 3:
    print('Usage: python q2.py contrast brightness')
    exit(1)

contrast = int(sys.argv[1])
brightness = int(sys.argv[2])

# contrast = 100
# brightness = 40

# print(contrast, brightness)

img = cv2.imread('q1.jpg')
result = np.array(img, dtype = np.int32)
n, m, _ = img.shape
for i in range(n):
    for j in range(m):
        b = result[i][j][0]
        g = result[i][j][1]
        r = result[i][j][2]
        # if (b + g) * 0.3 > r:
        #     result[i][j] = (result[i][j] - 127) * (contrast / 127 + 1) + 127 + brightness
            
result = np.clip(result, 0, 255)


result = np.array(result, dtype=np.uint8)

cv2.imshow('q1_1.jpg',result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('q1_2_ans.jpg', result)
