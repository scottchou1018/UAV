import cv2
import numpy as np
from typing import List, Tuple
from collections import defaultdict
import random


class DSU:
    def __init__(self, n):
        self.n = n
        self.dsu: List[int] = [-1 for _ in range(self.n + 1)]
        self.min_id: List[int] = [i for i in range(self.n + 1)]
    
    def get_size(self, x: int) -> int:
        return -self.dsu[self.find(x)]
    

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
    


def connected_component(img: np.ndarray) -> Tuple[np.ndarray, DSU]:
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
    return result, dsu

if __name__ == "__main__":

    T = 30

    cap = cv2.VideoCapture('car.mp4')
    if not cap.isOpened():
        raise RuntimeError("video not open successfully")
    backSub = cv2.createBackgroundSubtractorMOG2()
    while True:
        ret, frame = cap.read()
        work_frame = backSub.apply(frame)
        
        if not ret:
            break
        shadow_thres = backSub.getShadowThreshold()
        ret, work_frame = cv2.threshold(work_frame, shadow_thres, 255, cv2.THRESH_BINARY)
        cc, dsu = connected_component(work_frame)
        n, m = work_frame.shape
        work_frame = cv2.cvtColor(work_frame, cv2.COLOR_GRAY2BGR)
        gx_max = defaultdict(lambda:0)
        gx_min = defaultdict(lambda:n)
        gy_max = defaultdict(lambda:0)
        gy_min = defaultdict(lambda:m)

        big_group = set()

        print(cc)
        for i in range(n):
            for j in range(m):
                if(cc[i, j] == 0):
                    continue
                if dsu.get_size(cc[i, j]) >= T:
                    g_id = dsu.find_min_id(cc[i, j])
                    big_group.add(g_id)
                    gx_max[g_id] = max(gx_max[g_id], i)
                    gx_min[g_id] = min(gx_min[g_id], i)
                    gy_max[g_id] = max(gy_max[g_id], j)
                    gy_min[g_id] = min(gy_min[g_id], j)
        
        for g_id in big_group:
            frame = cv2.rectangle(frame, (gy_min[g_id], gx_min[g_id]), (gy_max[g_id], gx_max[g_id]), (0,0,255))


        cv2.imshow('video', frame)
        cv2.waitKey(33)
    
