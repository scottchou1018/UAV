import cv2
import numpy as np
import random
from collections import defaultdict
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
            if(result[i, j] >= threshold):
                result[i, j] = 255
            else:
                result[i, j] = 0
    return result


class DSU:
    def __init__(self, n):
        self.n = n
        self.dsu: list = [-1 for _ in range(self.n + 1)]
        self.min_id: list = [i for i in range(self.n + 1)]
    
    def find(self, x: int) -> int:
        if self.dsu[x] < 0:
            return x
        self.dsu[x] = self.find(self.dsu[x])
        return self.dsu[x]

    def find_min_id(self, x: int) -> int:
        return self.min_id[self.find(x)]

    def Union(self, x: int, y: int) -> None:
        x = self.find(x)
        y = self.find(y)
        if(x == y):
            return
        if self.dsu[x] > self.dsu[y]:
            x, y = (y, x)
        self.dsu[x] += self.dsu[y]
        self.dsu[y] = x
        self.min_id[x] = min(self.min_id[x], self.min_id[y])
    
    def increase_size(self):
        self.n += 1
        self.dsu.append(-1)
        self.min_id.append(self.n)
    


def connected_component(img: np.ndarray):
    n, m = img.shape
    result = np.zeros((n, m), dtype=np.int32)
    dsu = DSU(0)
    component_cnt = 0
    for i in range(n):
        for j in range(m):
            if img[i, j] == 0:
                continue
            if i != 0 and result[i - 1, j] != 0:
                result[i, j] = result[i - 1, j]
            if j != 0 and result[i, j - 1] != 0:
                if result[i, j] != 0:
                    dsu.Union(result[i, j], result[i, j - 1])
            if result[i, j] == 0:
                component_cnt += 1
                dsu.increase_size()
                result[i, j] = component_cnt
    
    for i in range(n):
        for j in range(m):
            if result[i, j] != 0:
                result[i, j] = dsu.find_min_id(result[i, j])
    return result

def rand_color():
    return np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

def draw_by_component(img: np.ndarray):
    compo = connected_component(img)
    n, m = compo.shape
    colored = np.zeros((n, m, 3), dtype=np.uint8)
    color_map = defaultdict(rand_color)
    for i in range(n):
        for j in range(m):
            if compo[i, j] != 0:
                colored[i, j] = color_map[compo[i, j]]
    
    return colored
            

if __name__ == "__main__":
    img = cv2.imread('connected_component.jpg')
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thres = otsu_threshold(gray_img)
    print(f'threshold: {thres}')
    gray_img = threshold_filter(gray_img, thres)
    result = draw_by_component(gray_img)
    
    
    cv2.imshow('result.jpg', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('q1_ans.jpg', result)