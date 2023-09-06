from pixels_to_ratio import *
from warpPerspective import *
from contours_to_pixels import *

image_path = r'C:\Users\88698\Desktop\Image Rectification\4.jpg'

def main():
    #warpPerspective(image_path)
    max_square_area = contours_to_pixels(image_path)
    pixels_to_ratio(image_path,max_square_area)
    
    

if __name__ == "__main__":
    main()

################################################################
# from importlib.machinery import SourceFileLoader

# class PathManager:
#     def __init__(self, path):
#         self.path = path

#     def run(self):

#         # contours_to_pixels

#         module_contours_to_pixels = "contours_to_pixels"
#         module_contours_to_pixels = SourceFileLoader(module_contours_to_pixels, f"{module_contours_to_pixels}.py").load_module()
#         area_contours = module_contours_to_pixels.contours_to_pixels(self.path)
#         print(max)


#         # 呼叫refi.py檔案中的main函式並傳遞path值
#         module_refi = "refi"
#         module_refi = SourceFileLoader(module_refi, f"{module_refi}.py").load_module()
#         module_refi.main(self.path,area_contours)

#         # 呼叫warpPerspective.py檔案中的main函式並傳遞path值
#         # module_warpPerspective = "warpPerspective"
#         # module_warpPerspective = SourceFileLoader(module_warpPerspective, f"{module_warpPerspective}.py").load_module()
#         # module_warpPerspective.warpPerspective(self.path)

# if __name__ == "__main__":
#     path = r'C:\Users\88698\Desktop\Image Rectification\5.jpg'
#     path_manager = PathManager(path)
#     path_manager.run()
