from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import ui_cvdl2023_hw2 as ui
import os
import torch
from torchvision import transforms
from torchsummary import summary
from PIL import Image
import matplotlib.pyplot as plt

from Q1.Q1 import Question1
from Q2.Q2 import Question2
from Q3.Q3 import Question3
from Q4.Q4 import Question4
from Q5.Q5 import Question5


class Main(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.imagePath = None
        self.videoPath = None
        self.dirPath = None

        # load data
        self.pushButtonLoadVideo.clicked.connect(self.getVideoPath)
        self.pushButtonLoadImage.clicked.connect(self.getImagePath)
        # self.pushButtonLoadFolder.clicked.connect(self.selectDir)

        # question 1
        self.pushButtonBackgroundSubtraction.clicked.connect(lambda: Q1Object.backgroundSubtraction(self.videoPath))

        # # question 2
        self.pushButtonPreprocessing.clicked.connect(lambda: Q2Object.preprocessing(self.videoPath))
        self.pushButtonVideoTracking.clicked.connect(lambda: Q2Object.videoTracking(self.videoPath))

        # # question 3
        self.pushButtonImageReconstruction.clicked.connect(lambda: Q3Object.imageReconstruction(self.imagePath))

        # question 4
        self.pushButtonShowModelStructure.clicked.connect(Q4Object.showModelStructure)        
        self.pushButtonShowAccuracyanLoss.clicked.connect(self.showAccuracyAndLoss)
        image_path = 'Q4/handwrite.png'  # 替換成你的圖片路徑
        self.pushButtonPredict.clicked.connect(lambda: self.showInference(image_path))

        # question 5
        self.pushButtonLoadImage_1.clicked.connect(self.showImageOnGUI)
        self.pushButtonShowImages.clicked.connect(Q5Object.showImages)
        self.pushButtonShowModelStructure_2.clicked.connect(Q5Object.showModelStructure)
        self.pushButtonShowComparsion.clicked.connect(Q5Object.showComparison)
        self.pushButtonInference.clicked.connect(lambda : self.showInference_5(self.imagePath))
        # self.pushButtonInference.clicked.connect(Q5Object.showInference)
        
    
    def selectVideo(self):
        fileName = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(None, caption='Choose a File', directory='C:\\', filter='Video Files (*.mp4)')[0])  # get turple[0] which is file name
        return fileName
    
    def getVideoPath(self):
        self.videoPath = self.selectVideo()

    def selectImage(self):
        fileName = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getOpenFileName(None, caption='Choose a File', directory='C:\\', filter='Image Files (*.png *.jpg *.bmp)')[0])  # get turple[0] which is file name
        return fileName
    
    def getImagePath(self):
        self.imagePath = self.selectImage()

    def selectDir(self):
        self.dirPath = QtCore.QDir.toNativeSeparators(QtWidgets.QFileDialog.getExistingDirectory(None, caption='Select a folder:', directory='C:\\', options=QtWidgets.QFileDialog.ShowDirsOnly))

    # overide to force exit
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        os._exit(0)

    def showAccuracyAndLoss(self):
        Q4Object.makeAccuracyAndLoss()

    def showInference(self, imgPath):
        label,output = Q4Object.showInference(imgPath)
        print(label)
        self.textArea1.setText(str(label))
        # 顯示機率分佈的直方圖
        plt.figure()
        plt.bar(range(10), output.tolist(), tick_label=range(10))
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title('Probability Distribution')
        plt.show()

    def showImageOnGUI(self):
        self.getImagePath()
        print(self.imagePath)
        pixmap = QtGui.QPixmap(self.imagePath)
        pixmap = pixmap.scaled(128,128,QtCore.Qt.KeepAspectRatio)
        self.graphicsView.setPixmap(pixmap)
        # scaled_pixmap = pixmap.scaled(224, 224)
        # pixmap_item = QtWidgets.QGraphicsPixmapItem(scaled_pixmap)
        # print(pixmap_item)
        # self.scene.addItem(pixmap_item)

    def showInference_5(self, imgPath):

        if imgPath == None:
            print("please load the image which want to predict")
            return


        print("predict start...")

        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        # resnet50_model = torch.load('Q5/model_40.pt', map_location ='cpu')
        model = torch.load("Q5/model_40.pt", map_location= "cpu")

        # 將模型切換為評估模式
        model.eval()

        # 載入並預處理單張圖片
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 需要根據模型的輸入尺寸進行調整
            transforms.ToTensor(),
        ]   )
        print("A:",imgPath)
        img = Image.open(imgPath)
        input_tensor = transform(img).unsqueeze(0)

        # 如果你的模型在 GPU 上，將輸入數據移動到 GPU 上
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        model = model.to(device)

        # 進行預測
        with torch.no_grad():
            output = model(input_tensor)

        # print(output)
        threshold = 0.5
        predictions = (output > threshold).float()
        # print(predictions)
        result = predictions.item()
        # print(result)

        if result < 0.5:
            predOutcome = 'Cat'
        else :
            predOutcome = 'Dog'

        print(predOutcome)
        self.textArea.setText('Prediction Label : ' + predOutcome)
        return predOutcome

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Q1Object = Question1()
    Q2Object = Question2()
    Q3Object = Question3()
    Q4Object = Question4()
    Q5Object = Question5()
    
    window = Main()
    window.show()
    sys.exit(app.exec_())