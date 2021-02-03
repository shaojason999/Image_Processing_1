"""
reference
https://www.learnopencv.com/camera-calibration-using-opencv/
https://vimsky.com/zh-tw/examples/detail/python-method-cv2.findChessboardCorners.html
https://blog.csdn.net/qq_40475529/article/details/89409303
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html
https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
https://kknews.cc/zh-tw/code/e6brxxr.html
"""

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from hw1 import Ui_Dialog
import cv2
import sys
import numpy as np

class Main(QMainWindow, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.init()

    def init(self):
        self.CHECKERBOARD = (11,8)
        self.Q1_objp_list,self.Q2_objp_list=[],[]
        self.Q1_corner_list,self.Q2_corner_list=[],[]
        self.Q1_ret_list,self.Q2_ret_list=[],[]
        self.Q1_img_path_list,self.Q2_img_path_list=[],[]
        self.Q1_img_list,self.Q2_img_list=[],[]
        self.Q1_gray_list,self.Q2_gray_list=[],[]
        self.Q1_rotation_vectors,self.Q2_rotation_vectors=[],[]
        self.Q1_translation_vectors,self.Q2_translation_vectors=[],[]
        self.selected_image_index=0

        #information = [str(i)+".bmp" for i in range(1,16)]
        information = [str(i) for i in range(1,16)]
        self.comboBox.addItems(information)

        print("Calculating, please wait")
        self.event_set()
        self.find_corner_and_calibrate()
        print("Finished")

    def event_set(self):
        self.pushButton.clicked.connect(self.draw_corner)
        self.pushButton_2.clicked.connect(self.show_intrinsic)
        self.pushButton_3.clicked.connect(self.show_extrinsic)
        self.pushButton_4.clicked.connect(self.show_distortion)
        self.pushButton_5.clicked.connect(self.close_windows)
        self.comboBox.activated[int].connect(self.image_select)
        self.pushButton_11.clicked.connect(self.show_images)
        self.pushButton_12.clicked.connect(self.stereo_disparity_map)
        self.pushButton_13.clicked.connect(self.find_keypoints)
        self.pushButton_14.clicked.connect(self.keypoints_match)

    def image_select(self, index):
        self.selected_image_index=index

    def find_keypoints(self):
        sift = cv2.xfeatures2d.SIFT_create()
        
        self.img1 = cv2.imread('Q4_Image/Aerial1.jpg',0)
        self.keypoints1 = sift.detect(self.img1,None)
        self.keypoints1 =sorted(self.keypoints1, key=lambda x:x.size, reverse=True)
        #print(i.pt,i.size,i.octave,i.response,i.angle,i.class_id)
        self.keypoints1=self.keypoints1[0:7]    # I only need the top 6 points, but two of these are exactly the same,(except the angle) so I need to take the top 7 in order to show 6 on the image
        self.img1=cv2.drawKeypoints(self.img1,self.keypoints1,self.img1)
        #self.img1=cv2.drawKeypoints(self.img1,self.keypoints1,self.img1,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        self.img2 = cv2.imread('Q4_Image/Aerial2.jpg',0)
        self.keypoints2 = sift.detect(self.img2,None)
        self.keypoints2 =sorted(self.keypoints2, key=lambda x:x.size, reverse=True)
        self.keypoints2=self.keypoints2[:7]
        self.img2=cv2.drawKeypoints(self.img2,self.keypoints2,self.img2)

        cv2.imwrite('Q4_Image/FeatureAerial1.jpg',self.img1)
        cv2.resizeWindow('FeatureAerial1.jpg', int(self.img1.shape[1]), int(self.img1.shape[0]))
        cv2.imshow('FeatureAerial1.jpg',self.img1)
        cv2.imwrite('Q4_Image/FeatureAerial2.jpg',self.img2)
        cv2.resizeWindow('FeatureAerial2.jpg', int(self.img2.shape[1]), int(self.img2.shape[0]))
        cv2.imshow('FeatureAerial2.jpg',self.img2)

    def keypoints_match(self):
        sift = cv2.xfeatures2d.SIFT_create()
        self.keypoints1,descriptors1=sift.compute(self.img1,self.keypoints1)
        self.keypoints2,descriptors2=sift.compute(self.img2,self.keypoints2)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1,descriptors2)
        matches =sorted(matches, key=lambda x:x.distance)
        img3 = cv2.drawMatches(self.img1, self.keypoints1, self.img2, self.keypoints2, matches[:4], None, flags=2)
        cv2.resizeWindow('keypoints_match', int(img3.shape[1]), int(img3.shape[0]))
        cv2.imshow("keypoints_match",img3)

    def stereo_disparity_map(self):
        imgL = cv2.imread('Q3_Image/imL.png',0)
        imgR = cv2.imread('Q3_Image/imR.png',0)
        temp = cv2.imread('Q3_Image/imL.png')

        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL,imgR)
        disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.namedWindow("Stereo Disparity Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Stereo Disparity Map", int(imgL.shape[1]), int(imgL.shape[0]))
        cv2.imshow("Stereo Disparity Map",disparity)

    def show_extrinsic(self):
        index=self.selected_image_index
        dst,jacobian=cv2.Rodrigues(self.Q1_rotation_vectors[index])    # convert from 3x1 to 3x3 # https://blog.csdn.net/qq_40475529/article/details/89409303
        rotation=np.array(dst)
        translation=np.array(self.Q1_translation_vectors[index])
        print("\nExtrinsic Matrix:")
        print(np.c_[rotation,translation])

    def show_images(self):
        i=0
        cv2.namedWindow("Slide", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Slide", int(self.Q2_img_list[i].shape[0]/4), int(self.Q2_img_list[i].shape[1]/4))

        point_vector_list=[]
        axis = np.float32([[3,3,-3], [1,1,0], [3,5,0], [5,1,0]])    # must be float
        for i in range(5):
            point_vector,jacobian = cv2.projectPoints(axis, np.array(self.Q2_rotation_vectors[i]), np.array(self.Q2_translation_vectors[i]), np.array(self.Q2_intrinsic), np.array(self.Q2_distortion))
            # draw line
            for j in range(len(point_vector)-1):
                for k in range(j+1,len(point_vector)):
                    cv2.line(self.Q2_img_list[i],tuple(point_vector[j][0]),tuple(point_vector[k][0]),(0, 0, 255),10,cv2.FILLED)
        
        while cv2.getWindowProperty("Slide", 0) >= 0:   # check is window closed
            cv2.imshow("Slide",self.Q2_img_list[i])
            cv2.waitKey(500)
            i=(i+1)%5

    def find_corner_and_calibrate(self):
        objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

        #Q1
        for i in range(1,16):
            self.Q1_img_path_list.append('Q1_Image/'+str(i)+'.bmp')

        for img_path in self.Q1_img_path_list:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, None)

            if(ret):
                self.Q1_objp_list.append(objp)
                self.Q1_gray_list.append(gray)
                self.Q1_corner_list.append(corners)
                self.Q1_ret_list.append(ret)
                self.Q1_img_list.append(img)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.Q1_objp_list, self.Q1_corner_list, self.Q1_gray_list[0].shape[::-1], None, None)
        self.Q1_intrinsic=mtx
        self.Q1_distortion=dist
        self.Q1_rotation_vectors=rvecs
        self.Q1_translation_vectors=tvecs

        #Q2
        for i in range(1,6):
            self.Q2_img_path_list.append('Q2_Image/'+str(i)+'.bmp')

        for img_path in self.Q2_img_path_list:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, None)

            if(ret):
                self.Q2_objp_list.append(objp)
                self.Q2_gray_list.append(gray)
                self.Q2_corner_list.append(corners)
                self.Q2_ret_list.append(ret)
                self.Q2_img_list.append(img)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.Q2_objp_list, self.Q2_corner_list, self.Q2_gray_list[0].shape[::-1], None, None)
        self.Q2_intrinsic=mtx
        self.Q2_distortion=dist
        self.Q2_rotation_vectors=rvecs
        self.Q2_translation_vectors=tvecs

    def draw_corner(self):
        for i in range(len(self.Q1_img_path_list)):
            cv2.drawChessboardCorners(self.Q1_img_list[i], self.CHECKERBOARD, self.Q1_corner_list[i], self.Q1_ret_list[i])

            cv2.namedWindow(self.Q1_img_path_list[i], cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.Q1_img_path_list[i], int(self.Q1_img_list[i].shape[0]/4), int(self.Q1_img_list[i].shape[1]/4))
            cv2.imshow(self.Q1_img_path_list[i],self.Q1_img_list[i])

    def close_windows(self):
        cv2.destroyAllWindows()

    def show_intrinsic(self):
        print("\nIntrinsic Matrix:")
        print(self.Q1_intrinsic)

    def show_distortion(self):
        print("\nDistortion Matrix:")
        print(self.Q1_distortion)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()

    sys.exit(app.exec_())