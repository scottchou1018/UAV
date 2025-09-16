import cv2
img = cv2.imread('q1.jpg')
# cv2.imshow('q1.jpg',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

result_img = img.copy()
result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)

n, m, _ = img.shape
for i in range(n):
    for j in range(m):
        b = img[i][j][0]
        g = img[i][j][1]
        r = img[i][j][2]
        if b > 100 and b * 0.6 > g and b * 0.6 > r:
            result_img[i][j] = (b, g, r)

cv2.imshow('q1.jpg', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('q1_1_ans.jpg', result_img)