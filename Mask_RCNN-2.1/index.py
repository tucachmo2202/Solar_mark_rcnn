import cv2
import json
import os
import numpy as np
from PIL import Image


#Tham so cho ham HoughLinesP
RHO = 10 #The resolution of the parameter r in pixels.
THRESHOLD = 5 #The minimum number of intersections to “detect” a line
MINLINLENGTH = 450 #The minimum number of points that can form a line.
MAXLINEGAP = 280 #The maximum gap between two points to be considered in the same line.

class Line():

    def __init__(self, index = 0, start = [0, 0], end = [0, 0], points = []):
        self.index = index
        self.start = start
        self.end = end
        self.points = list(points)
    
    def set_line(self, index, start, end):
        self.index = index
        self.start = start
        self.end = end

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

#Hàm này loại bỏ những đường thẳng song song trùng nhau
def reject_line(list_line):
    for j in range(len(list_line) - 1, -1, -1):
        for i in range(len(list_line) - 1, -1, -1):
            x1,y1 = list_line[j][:2]
            if (i!=j and distance((list_line[i][0], list_line[i][1]), (list_line[i][2], list_line[i][3]), (x1, y1)) < 5):
                del list_line[i]
                print("1 line removed")
                break
    return list_line

def index(img, bounding_boxs, rect):
    #Create mask:
    w, h = img.shape[0:2]
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
    #88 image is 5, 10, 700, 1024
    # Detect points that form a line
    max_slider = 5 #Số lượng điểm giao nhau tối thiểu để tạo thành đường thẳng
    lines = cv2.HoughLinesP(edges, RHO, np.pi/180, THRESHOLD, minLineLength = MINLINLENGTH, maxLineGap = MAXLINEGAP)
    # Draw lines on the image
    print(len(lines))
    lines = [line[0] for line in lines]
    line_array = []
    i = 0
    #Sắp xếp các line
    lines.sort(key = lambda lines: lines[1])
    lines = reject_line(lines)
    for line in lines:
        x1, y1, x2, y2 = line
        start = [x1, y1]
        end = [x2, y2]
        line_object = Line(i, start, end)
        line_array.append(line_object)
        i+=1


    for box in rect:
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (255,255,255), 1)

    cv2.imshow("Result Image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    threshold = 8
    bounding_boxs_center.sort(key = lambda bounding_boxs_center: bounding_boxs_center[0])
    
    for line in line_array:
        index, start, end = line.get_line()
        i = 0
        for center_point in bounding_boxs_center:
            cv2.circle(img, tuple(center_point), 3, (0,0,255), -1)
            # print(distance(start, end, tuple(center_point)))
            if (distance(start, end, tuple(center_point)) <= 7.0):
                line.update(center_point)
                # cv2.putText(img, str(index) + "-" + str(i), tuple(center_point), cv2.FONT_HERSHEY_PLAIN, 1, (25,112,3), 1)
                i += 1

    #Sắp xếp line theo giá trị y của điểm đầu
    line_array = sorted(line_array, key=lambda x: x.points[1])
   
    #Hiển thị lên ảnh
    i = 0
    for line_object in line_array:
        x1, y1 = line_object.start
        x2, y2 = line_object.end
        cv2.putText(img, str(i), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (25,112,3), 1)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        j = 0
        for point in line_object.points:
            cv2.putText(img, str(i) + "-" + str(j), tuple(point), cv2.FONT_HERSHEY_PLAIN, 1, (25,112,3), 1)
            j += 1
        i += 1
    cv2.imshow("Result Image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite("/home/manhas/Desktop/ressd.jpg", img)

res = json.load(open("/home/manhas/Desktop/result_image/save_folder/84.json"))
bbs = res["bounding_box"]
rect = res["boxes"]

img = cv2.imread("/home/manhas/Desktop/84.jpg")
cv2.imshow("sss", img)
cv2.waitKey()
cv2.destroyAllWindows()
index(img, bbs, rect)
