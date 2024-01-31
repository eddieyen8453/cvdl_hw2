import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from torchsummary import summary
import torchvision.models as models

# from .model import predict_image    #, show_train_images


class Question4:
    def showDataAugmentation(self, imgPath):
        if imgPath == None:
            print('Please load the image.')
        else:
            imgRotation = self.showRandomRotation(imgPath)
            imgResized = self.showRandomResizedCrop(imgPath)
            imgFlipped = self.showRandomHorizontalFlip(imgPath)
            result = self.getConcatH(imgRotation, imgResized, imgFlipped)
            result.show()
            # result.save('Q5/augmentation.png')


    def showRandomRotation(self, imgPath):
        img = Image.open(imgPath)
        transfrom = transforms.RandomRotation(degrees=(0, 180))    # rotated degree from 0 to 180
        img = transfrom(img)
        # img.show()
        return img

    def showRandomResizedCrop(self, imgPath):
        img = Image.open(imgPath)
        transfrom = transforms.RandomResizedCrop(size=img.size, scale=(0.05, 0.99)) # random cropped size is 0.05x to 0.99x
        img = transfrom(img)
        # img.show()
        return img

    def showRandomHorizontalFlip(self, imgPath):
        img = Image.open(imgPath)
        transfrom = transforms.RandomHorizontalFlip(p=0.5)   # filp rate is 1/2
        img = transfrom(img)
        # img.show()
        return img

    # mix the images horizontally
    def getConcatH(self, img1, img2, img3):
        concatenated = Image.new('RGB', (img1.width + img2.width + img3.width, img1.height))
        concatenated.paste(img1, (0, 0))
        concatenated.paste(img2, (img1.width, 0))
        concatenated.paste(img3, (img1.width + img2.width, 0))
        return concatenated

    # def showTrainImages(self):
    #     show_train_images()

    def showModelStructure(self):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        # if device == 'cuda:0':
        #     model = YourModel()  # 创建你的模型实例
        #     model.load_state_dict(torch.load('Q4/VGG19_BN_model.pth'))
        #     model.to('cuda')
        # else:
        model = torch.load('Q4/VGG19_BN_model.pth', map_location ='cpu')
        # model.to('cuda')

        summary(model, (1, 224, 224))   # show model structure

    def makeAccuracyAndLoss(self):
        imgAcc = cv2.imread('Q4/training_accuracy_validation_accuracy.png')
        imgLoss = cv2.imread('Q4/training_loss_validation_loss.png')
        result = np.concatenate((imgAcc, imgLoss), axis=0)  # concat two pictures together
        # image1 = cv2.imread('Q4/result.png')
        cv2.imwrite('Q4/result.png', result)
        cv2.imshow('Accuracy and Loss ', result)

    def showInference(self, image_path):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定義預處理步驟
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # VGG19 需要 224x224 的輸入
            transforms.ToTensor(),
        ])

        # 載入完整模型
        model_path = 'Q4/VGG19_BN_model.pth'  # 替換成你的模型路徑
        VGG19_BN_model = torch.load(model_path, map_location=device)
        VGG19_BN_model = VGG19_BN_model.to(device)

        # 將模型切換為評估模式
        VGG19_BN_model.eval()

        # 載入並預處理單張圖片
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 需要根據模型的輸入尺寸進行調整
            transforms.Grayscale(num_output_channels=1),  # 將圖像轉換為單通道
            transforms.ToTensor(),
        ]   )

        img = Image.open(image_path)
        input_tensor = transform(img).unsqueeze(0)

        # 如果你的模型在 GPU 上，將輸入數據移動到 GPU 上
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        VGG19_BN_model = VGG19_BN_model.to(device)
        # 進行預測
        with torch.no_grad():
            output = VGG19_BN_model(input_tensor)

        # 處理預測結果
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        self.predicted_class = torch.argmax(probabilities).item()

        print("predicted_class : ",self.predicted_class)
        print("probabilities : ",probabilities.tolist())

        # label,output
        return self.predicted_class,probabilities

if __name__ == '__main__':
    print('This is Q5')
    print('Do run run this file')