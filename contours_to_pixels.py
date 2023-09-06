
import cv2
import numpy as np
# 使用方式
# 1. path 為輸入影像
# 2. 輸出即為畫好輪廓的影像

# # 讀取圖像
path = r'C:\Users\88698\Desktop\Image Rectification\4.jpg'

def contours_to_pixels(image_path):

    image = cv2.imread(image_path)

    # 將圖像轉換為HSV色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定義紅色的HSV範圍
    lower_red = np.array([0, 10, 10])  # 調整下界
    upper_red = np.array([11, 255, 255])  # 調整上界

    # 創建二值圖像，其中紅色區域是白色，其餘是黑色
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    # 使用形態學操作來清除雜訊
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    ###line###

    # 高斯模糊去雜訊##
    blurred = cv2.GaussianBlur(red_mask, (5, 5), 0)

    # 查找紅色區域的輪廓
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化正方形計數和像素計數
    square_count = 0
    square_pixel_count = 0
    # 初始化正方形面積和最大面積
    max_square_area = 0

    # 遍歷每個輪廓
    for contour in contours:
        # 計算輪廓的面積
        area = cv2.contourArea(contour)

        # 過濾掉小輪廓
        if area > 100:  # 調整閾值根據需要
            # 計算輪廓的外接矩形
            x, y, w, h = cv2.boundingRect(contour)

            # 檢查外接矩形是否接近正方形
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:
                # 如果找到更大的正方形，更新最大面積和輪廓
                if area > max_square_area:
                    max_square_area = area
                    max_square_contour = contour

  # 在原始圖像中繪製最大正方形輪廓
    if max_square_contour is not None:
        cv2.drawContours(image, [max_square_contour], -1, (10, 255, 0), 2)
    
    print('-----contours_to_pixels-----')

    print(f"area:{max_square_area} pixels")

    print(' ')

    ###########依原比例縮放##############

    # 取得原始圖像的高度和寬度
    height, width = image.shape[:2]

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

    # 重新調整視窗大小
    cv2.namedWindow('result', 0)
    cv2.resizeWindow('result', new_width, new_height)

    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return max_square_area

if __name__ == "__main__":
    contours_to_pixels()