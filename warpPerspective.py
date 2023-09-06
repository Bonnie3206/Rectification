# import cv2
# import numpy as np

# for 透視轉換
# 1. PATH為輸入影像
# 2. 選擇以尺圍出的正方形 的四個頂點

import cv2
import numpy as np

selected_points = []

def select_point(event, x, y, flags, param):
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param, (x,y), 10, (0,255,0), -1)
        cv2.imshow("Select L Ruler Corners", param)
        selected_points.append((x,y))
        if len(selected_points) == 4:  # Change to 4 points
            cv2.destroyAllWindows()

def warpPerspective(image_path):
    
    path = image_path
    img = cv2.imread(path)
    clone = img.copy()

    height, width = img.shape[:2]

    #選取圖片的window
    max_width = 800
    max_height = 600

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

    cv2.namedWindow('Select L Ruler Corners', 0)
    cv2.resizeWindow('Select L Ruler Corners', new_width, new_height)

    cv2.setMouseCallback("Select L Ruler Corners", select_point, clone)
    cv2.imshow("Select L Ruler Corners", clone)
    cv2.waitKey(0)

    pts1 = np.array(selected_points, dtype="float32")

    w, h = 500, 500

    # Define the new corners for the transformed L ruler
    pts2 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")  

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, matrix, (w, h))

    #顯示圖片的window
    height, width = img.shape[:2]

    max_width = w
    max_height = h
    #因為會切只有校正卡那邊 此時用原本圖像比例的眶反而會失真
    new_width = max_width
    new_height = max_height

    cv2.namedWindow('Warped L Ruler', 0)
    cv2.resizeWindow('Warped L Ruler', new_width, new_height)

    cv2.imshow("Warped L Ruler", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

