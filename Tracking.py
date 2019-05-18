import numpy as np

class Tracking:
    def __init__(self):
        self.boxes = list()
        self.numbers = list()
        self.face_num = 0
    
    def update(self, boxes: list):
        previous_boxes = self.boxes
        previous_numbers = self.numbers

        self.boxes = boxes
        
        if len(boxes) == 0:
            return []

        if len(previous_boxes) == 0:
            self.numbers = np.zeros(len(boxes))
            self.boxes = boxes
            for i, box in enumerate(boxes):
                self.face_num += 1
                
                self.numbers[i] = self.face_num

            return self.numbers

        self.numbers = 2* np.ones(len(boxes))
        IOUs = np.zeros((len(previous_boxes), len(boxes)))
        for i, previous_box in enumerate(previous_boxes):
            IOUs[i] = [box.IOU(previous_box) for box in boxes]
        
        if len(boxes) >= len(previous_boxes):
            for i, previous_box in enumerate(previous_boxes):
                self.numbers[np.argmax(IOUs[i])] = previous_numbers[i]
            
            for i, number in enumerate(self.numbers):
                if number == 2:
                    self.face_num += 1
                    self.numbers[i] = self.face_num

        else:
            for i, box in enumerate(boxes):
                self.numbers[i] = previous_numbers[np.argmax(IOUs[:, i])]
            
            for i, number in enumerate(self.numbers):
                if number == 2:
                    self.face_num += 1
                    self.numbers[i] = self.face_num
                    
        return self.numbers
        

    
    