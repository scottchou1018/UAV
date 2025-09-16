import cv2
import numpy as np
import math

def histo_equalize(arr):
    histo = np.zeros((256), dtype=np.int32)
    n, m = arr.shape
    for i in range(n):
        for j in range(m):
            histo[arr[i,j]] += 1
    cum_arr = np.zeros((256), dtype=np.int32)
    cum_arr[0] = histo[0]
    for i in range(1, 256):
        cum_arr[i] = cum_arr[i - 1] + histo[i]
    
    MAX_VAL = 255
    result = np.zeros(arr.shape, dtype=np.int32)

    for i in range(n):
        for j in range(m):
            result[i, j] = round(cum_arr[arr[i, j]] / cum_arr[MAX_VAL] * MAX_VAL)
    result = np.clip(result, 0, 255)
    result = np.array(result, dtype=np.uint8)
    return result

def BGR_histo(img, file_name: str):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    b_his_eq = histo_equalize(b)
    g_his_eq = histo_equalize(g)
    r_his_eq = histo_equalize(r)
    
    result = cv2.merge([b_his_eq, g_his_eq, r_his_eq])
    cv2.imshow('result.jpg', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(file_name, result)

def HSV_histo(img, file_name: str):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = img[:,:,0]
    s = img[:,:,1]
    v = img[:,:,2]
    v_his_eq = histo_equalize(v)
    result = cv2.merge([h, s, v_his_eq])
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    cv2.imshow('result.jpg', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(file_name, result)


if __name__ == '__main__':
    img = cv2.imread('histogram.jpg')
    BGR_histo(img, 'q1_bgr_ans.jpg')
    HSV_histo(img, 'q1_hsv_ans.jpg')
    