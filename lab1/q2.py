import cv2
import numpy as np
import sys
import math
from tqdm import tqdm
if len(sys.argv) != 2:
    print('Usage: q2.py ratio')

r = float(sys.argv[1])

img = cv2.imread('q2.jpg')
img = np.array(img, np.int32)
old_n, old_m, _ = img.shape
n = int(old_n * r)
m = int(old_m * r)

result = np.zeros([n, m, 3], dtype = np.int32)

for x in tqdm(range(n)):
    for y in range(m):
        old_x = x / r
        old_y = y / r
        left_x = math.floor(x / r)
        left_y = math.floor(y / r)
        # print(x, y, old_x, old_y)
        if left_y + 1 >= old_m and left_x + 1 >= old_n:
            result[x][y] = img[left_x][left_y].copy()
        elif left_y + 1 >= old_m:
            result[x][y] = (old_x - left_x) * img[left_x + 1][left_y] + (left_x + 1 - old_x) * img[left_x][left_y]
        elif left_x + 1 >= old_n:
            result[x][y] = (old_y - left_y) * img[left_x][left_y + 1] + (left_y + 1 - old_y) * img[left_x][left_y]
        else:
            left_val = (old_x - left_x) * img[left_x + 1][left_y] + (left_x + 1 - old_x) * img[left_x][left_y]
            right_val = (old_x - left_x) * img[left_x + 1][left_y + 1] + (left_x + 1 - old_x) * img[left_x][left_y + 1]
            result[x][y] = (old_y - left_y) * right_val + (left_y + 1 - old_y) * left_val

result = np.clip(result, 0, 255)
result = np.array(result, np.uint8)

cv2.imshow('q2.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('q2_ans.jpg', result)
