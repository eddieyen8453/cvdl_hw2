import cv2
import numpy as np
import matplotlib.pyplot as plt


class Question2:
    def __init__(self) -> None:
        pass

    def preprocessing(self, videoPath):
        
        # first frame
        img_name = "first_frame.jpg"
        # videoPath = "C:\\Users\\user\\Desktop\\Dataset_CvDl_Hw2\\Q2\\optical_flow.mp4"
        capture = cv2.VideoCapture(videoPath)
        FPS = capture.get(cv2.CAP_PROP_FPS)
        ret, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  ## BGR to RGB
        cv2.imwrite('Q2\\{}'.format(img_name), frame)


        # add cross to first frame
        dirPath = "C:\\Users\\user\\Desktop\\class_NCKU\\cv\\Hw2\\hw2\\Q2\\{}".format(img_name)
        img_gray = cv2.imread(dirPath,0)
        maxCorners = 1
        qualityLevel = 0.3
        minDistance = 7
        blockSize = 7

        corners = cv2.goodFeaturesToTrack(img_gray, maxCorners, qualityLevel, minDistance, blockSize)
        x = int(corners[0][0][0])
        y = int(corners[0][0][1])

        img = cv2.imread(dirPath)

        # cv2.line(影像, 開始座標, 結束座標, 顏色, 線條寬度)
        img = cv2.line(img, (x-20, y), (x+20, y), (255, 0, 0), 4)
        img = cv2.line(img, (x, y-20), (x, y+20), (255, 0, 0), 4)
        
        cv2.imwrite('Q2\\withcross_{}'.format(img_name), img)

        plt.imshow(img)
        plt.show()

    def videoTracking(self, videoPath):
        capture = cv2.VideoCapture(videoPath)
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 1,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        # color = np.random.randint(0,255,(100,3))

        # Take first frame and find corners in it
        ret, old_frame = capture.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        x = int(corners[0][0][0])
        y = int(corners[0][0][1])
        # old_gray = cv2.line(old_gray, (x-20, y), (x+20, y), (255, 0, 0), 4)
        # old_gray = cv2.line(old_gray, (x, y-20), (x, y+20), (255, 0, 0), 4)


        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while(1):
            ret,frame = capture.read()
            print(frame)
            if frame is None:
                cv2.destroyAllWindows()
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, corners, None, **lk_params)

            # Select good points
            good_new = p1[st==1]
            good_old = corners[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                # print()
                mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), (0,100,255), 4)
                x_line = cv2.line(frame, (int(a)-20, int(b)), (int(a)+20, int(b)), (0, 0, 255), 5)
                y_line = cv2.line(frame, (int(a), int(b)-20), (int(a), int(b)+20), (0, 0, 255), 5)
                # frame = cv2.circle(frame,(int(a),int(b)),5,(0,100,255),-1)
            img = cv2.add(x_line, mask)

            cv2.imshow('frame',img)
            k = cv2.waitKey(30) & 0xff
            # print(k)
            if k == 27:
                break

            # # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            corners = good_new.reshape(-1,1,2)


if __name__ == "__main__":
    pass