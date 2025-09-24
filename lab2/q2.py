import cv2
import numpy as np

VAL_MAX = 255

def get_histo(img: np.ndarray):
    n, m = img.shape
    histo = np.zeros((256), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            histo[img[i,j]] += 1
    
    return histo

def otsu_threshold(img: np.ndarray) -> int:
    histo = get_histo(img)
    n_b = 0
    n_o = np.sum(histo)
    b_sum = 0
    o_sum = 0
    for i in range(VAL_MAX + 1):
        o_sum += histo[i] * i

    max_val = -1
    max_thres = 0
    for i in range(0, VAL_MAX + 1):
        
        
        if n_b != 0 and n_o != 0:
            bavg = b_sum / n_b
            oavg = o_sum / n_o
            if n_b * n_o * ((bavg - oavg) ** 2) > max_val:
                max_thres = i
                max_val = n_b * n_o * ((bavg - oavg) ** 2)
        
        b_sum = (b_sum + i * histo[i])
        o_sum = (o_sum - i * histo[i])
        n_b = n_b + histo[i]
        n_o = n_o - histo[i]
    
    return max_thres

def threshold_filter(img: np.ndarray, threshold: int) -> np.ndarray:
    result = img.copy()
    n, m = result.shape
    for i in range(n):
        for j in range(m):
            if(result[i, j] > threshold):
                result[i, j] = 255
            else:
                result[i, j] = 0
    return result
    


if __name__ == '__main__':
    img = cv2.imread('otsu.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thres = otsu_threshold(img)
    print(thres)
    result = threshold_filter(img, thres)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.imshow('result.jpg', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('q2_otsu_ans.jpg', result)
    
    