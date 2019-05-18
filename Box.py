class Box:
    def __init__(self, L, T, R, B):
        self.left = L
        self.top = T
        self.right = R
        self.bottom = B
        self.area = abs((self.right - self.left) * (self.top - self.bottom))

    def IOU(self, box):
        if (box.left > self.right) or (box.right < self.left):
            return 0

        if (box.top > self.bottom) or (box.bottom < self.top):
            return 0

        W = [box.left, box.right, self.left, self.right]
        W.sort()
        w = abs(W[1] - W[2])

        H = [box.top, box.bottom, self.top, self.bottom]
        H.sort()
        h = abs(H[1] - H[2])
        overlap_box_area = w * h


        return overlap_box_area / float(self.area + box.area - overlap_box_area)
        