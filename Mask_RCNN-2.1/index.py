import cv2
import json
import os
import numpy as np 
from PIL import Image


class Line():
    def __init__(self, index, start, end, points = []):
        self.index = index
        self.start = start
        self.end = end
        self.points = points
    
    def update(self, point):
        self.points.append(point)
    
    def get_line(self):
        return self.index, self.start, self.end

    def get_point(self):
        return self.points

#Tính khoảng cách từ point đến đường thằng đi qua 2 điểm start và end
def distance(start, end, point):
    start = np.asarray(start)
    end = np.asarray(end)
    point = np.asarray(point)
    return abs(np.cross(end-start, point - start)/np.linalg.norm(end-start))

def index(img, bounding_boxs, rect):
    #Create mask:
    w, h = img.shape[0:2]
    print("weight", w)
    print("height", h)
    mask = np.zeros((w, h))
    bounding_boxs_center = []
    for bb in bounding_boxs:
        #Center_point
        center_point = (int((bb[0] + bb[2])/2), int((bb[1] + bb[3])/2))
        bounding_boxs_center.append(list(reversed(center_point)))
        mask[center_point] = 255
    
    cv2.imshow("af", mask)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite("ajfkaf.jpg", mask)
    # mask = Image.fromarray(mask, 'RGB')
    # mas = mask.copy()
    mask = cv2.imread("ajfkaf.jpg", cv2.IMREAD_COLOR)
    os.remove("ajfkaf.jpg")
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray, 50, 200)
    # Detect points that form a line
    max_slider = 5 #Số lượng điểm giao nhau tối thiểu để tạo thành đường thẳng
    lines = cv2.HoughLinesP(edges, 10, np.pi/180, max_slider, minLineLength=700, maxLineGap = 1024)
    # Draw lines on the image
    print(len(lines))
    lines = [line[0] for line in lines]
    line_array = []
    i = 0
    #Sắp xếp các line
    lines.sort(key = lambda lines: lines[1])
    for line in lines:
        print(line)
        x1, y1, x2, y2 = line
        line_object = Line(i, [x1, y1], [x2, y2])
        line_array.append(line_object)
        cv2.putText(img, str(i), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (25,112,3), 1)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        i+=1
    # Show result
    # for bb in bbs:
    #     cv2.rectangle(img, (bb[1], bb[0]), (bb[3], bb[2]), (55, 255, 0), 1)

    for box in rect:
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0,0,255), 1)

    cv2.imshow("Result Image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    threshold = 8
    print(len(bounding_boxs_center))
    bounding_boxs_center.sort(key = lambda bounding_boxs_center: bounding_boxs_center[0])
    for line in line_array:
        index, start, end = line.get_line()
        i = 0
        for center_point in bounding_boxs_center:
            if (distance(start, end, center_point) < threshold):
                line.update(center_point)
                cv2.putText(img, str(index) + "-" + str(i), tuple(center_point), cv2.FONT_HERSHEY_PLAIN, 1, (25,112,3), 1)
                i += 1
                print(str(i))

    cv2.imshow("Result Image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

res = json.load(open("/home/manhas/Desktop/result_image/save_folder/84.json"))
bbs = res["bounding_box"]
rect = res["boxes"]
print(rect)

img = cv2.imread("/home/manhas/Desktop/result_image/save_folder/84.jpg")
cv2.imshow("sss", img)
cv2.waitKey()
cv2.destroyAllWindows()
index(img, bbs, rect)