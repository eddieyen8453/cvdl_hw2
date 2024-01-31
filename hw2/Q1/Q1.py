import cv2
import numpy as np

class Question1:
    def __init__(self) -> None:
        pass

    def backgroundSubtraction(self, videoPath):

        # Create a VideoCapture object
        capture = cv2.VideoCapture(videoPath)

        # Create an empty list to store the gaussian models for each pixel
        gaussians = []
        # Create an empty list to store the mask for each frame
        masks = []
        video_frames = []

        FPS = capture.get(cv2.CAP_PROP_FPS)
        print("FPS :" ,FPS)
        # load first 25 frames
        for count in range(25):
            # Read the next frame from the video
            ret, frame = capture.read()

            # Check if the frame was successfully read
            if not ret:
                break

            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if count == 0:
                pixels = np.empty((gray.shape[0], gray.shape[1]), object)
                # Loop through all the pixels in the frame
                for i in range(gray.shape[0]):
                    for j in range(gray.shape[1]):
                        pixels[i, j] = []

            for i in range(gray.shape[0]):
                for j in range(gray.shape[1]):
                    # Get the current pixel value
                    pixels[i, j].append(gray[i, j])

            video_frames.append(frame)
            masks.append(mask)
            print("loading frame :" + str(count + 1))

                        
        # bulid the gaussian model with first 25 frames
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                mean = sum(pixels[i, j]) / len(pixels[i, j])
                std = 5 if np.std(pixels[i, j]) < 5 else np.std(pixels[i, j])
                # Create a gaussian model for this pixel and add it to the list
                gaussians.append((mean, std))

        print("gaussian model is built")

        # load the rest frames
        while capture.isOpened():
            ret, frame = capture.read()

            # Check if the frame was successfully read
            if not ret:
                break
            # initialise the mask matrix
            mask = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Test the current pixel value against the gaussian model
            for i in range(gray.shape[0]):
                for j in range(gray.shape[1]):
                    mean, std = gaussians[i * gray.shape[1] + j]
                    if abs(gray[i, j] - mean) > 5 * std:
                    # Set the pixel value to 255 (foreground, white)
                        mask[i, j] = 255
                    else:
                        # Set the pixel value to 0 (background, black)
                        mask[i, j] = 0
            video_frames.append(frame)
            masks.append(mask)
            print("loading frame :" + str(count + 1))
            count += 1

        # write the concatenated output to the video file
        output_path = videoPath[:-4] + "_output.mp4"
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), FPS, (round(
            3 * capture.get(cv2.CAP_PROP_FRAME_WIDTH)), round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))


        for i in range(len(masks)):
            subtracted_frame = cv2.bitwise_and(video_frames[i], video_frames[i], mask=masks[i])
            concatenated_output = cv2.hconcat([video_frames[i], cv2.cvtColor(masks[i], cv2.COLOR_GRAY2BGR), subtracted_frame])
            video_writer.write(concatenated_output.astype(np.uint8))
            print("generating video in frame: " + str(i + 1))

        print("video generation is done")

        # release
        capture.release()
        video_writer.release()

        # play the video
        self.playVideo(output_path=output_path, fps=FPS)

    def playVideo(self, output_path, fps):
        capture_out = cv2.VideoCapture(output_path)

        while capture_out.isOpened():
            ret, frame = capture_out.read()

            # if frame is read correctly ret is True
            if not ret:
                print("Video played")
                break
            
            cv2.imshow("Background Subtraction", frame)  # resize to show completelly
            # press "q" to exit playing
            if cv2.waitKey(round(fps)) == ord('q'):
                break
        
        capture_out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # for testing
    Q1 = Question1()
    Q1.backgroundSubtraction("C:\\Users\\Xi\\Desktop\\cvdl_hw2\\Dataset_CvDl_Hw2\\Q1_Image\\traffic.mp4")