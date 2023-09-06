import cv2
import numpy as np

#物件導向寫法的pixels_to_ratio

class CentroidDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)

    def find_centroids(self):
        # 轉成HSV格式來偵測綠色
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 43, 46])
        upper_green = np.array([77, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # 找到輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroids = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        print('cx = '+str(cx))
        print('cy = '+str(cy))
        return centroids
    
class CentroidAnalyzer:
    def __init__(self, centroids):
        self.centroids = centroids
        
    def get_two_nearest_centroids(self):
        if len(self.centroids) < 2:
            return None, None
        
        min_distance = float('inf')
        c1, c2 = None, None
        sorted_centroids = sorted(self.centroids, key=lambda x: (x[0], x[1]))

        for i in range(len(sorted_centroids)):
            for j in range(i + 1, len(sorted_centroids)):
                distance = ((sorted_centroids[i][0] - sorted_centroids[j][0])**2 + 
                            (sorted_centroids[i][1] - sorted_centroids[j][1])**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    c1, c2 = sorted_centroids[i], sorted_centroids[j]

        return c1, c2
    def calculate_scale_ratio(self, c1, c2):
        triangle_edge = np.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
        real_height = 8.0  # 8公分
        real_edge = np.sqrt((8*8)+1)

        if triangle_edge == 0:
            print("Error: Triangle height is zero!")
            return
        else:
            ratio = real_edge / triangle_edge

        print(f"Scale Ratio: {ratio} cm/pixel")

        return ratio
    
    def calculate_real_area(self, ratio, contours):

        real_area = contours * ratio* ratio
        print(f"real_area: {real_area} cm*cm")

def main(path,contours):

    path = path
    #path = r'C:\Users\88698\Desktop\Image Rectification\1.jpg'
    centroid_detector = CentroidDetector(path)

    
    centroids = centroid_detector.find_centroids()
    # 從質心中選取兩個最靠近的
    if len(centroids) < 2:
        print("Cannot find enough green squares!")
        return
    
    centroid_analyzer = CentroidAnalyzer(centroids)
    c1, c2 = centroid_analyzer.get_two_nearest_centroids()

    if c1 is None or c2 is None:
        print("Error: Cannot find two distinct centroids!")
        return
    
    cv2.line(centroid_detector.img, c1, c2, (0, 0, 255), 2)
    ratio = centroid_analyzer.calculate_scale_ratio(c1, c2)
    centroid_analyzer.calculate_real_area(ratio, contours)
    
    img = cv2.imread(path)
    for centroid in centroids:
        cv2.circle(img, centroid, 5, (0, 0, 255), -1)
    
    ###########依原比例縮放##############

    # 取得原始圖像的高度和寬度
    height, width = img.shape[:2]

    # 指定希望調整視窗大小的最大寬度和高度
    max_width = 800
    max_height = 600

    # 根據縱橫比例計算新的視窗大小
    if width > max_width or height > max_height:
        if width / max_width > height / max_height:
            new_width = max_width
            new_height = int(height / (width / max_width))
        else:
            new_height = max_height
            new_width = int(width / (height / max_height))
    else:
        new_width = width
        new_height = height
        
    # 在質心間繪製線段
    cv2.line(img, c1, c2, (0, 0, 255), 2)

    cv2.namedWindow('result', 0)
    cv2.resizeWindow('result', new_width, new_height)
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()