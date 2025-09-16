import cv2
import numpy as np
import math
img = cv2.imread('q3.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0)

n, m = img.shape

gx = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])
gy = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

x_grad = cv2.filter2D(img, -1, gx)
y_grad = cv2.filter2D(img, -1, gy)


result = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        x_g = int(x_grad[i][j])
        y_g = int(y_grad[i][j])
        result[i][j] = math.sqrt(x_g * x_g + y_g * y_g)

result = np.clip(result, 0, 255)
result = np.array(result, np.uint8)

# cv2.imshow('result.jpg', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('q3_ans.jpg', result)