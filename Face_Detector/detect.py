import cv2
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

faceNet = cv2.dnn.readNet(faceModel, faceProto)

frame = cv2.imread(args.image)

resultImg, faceBoxes = highlightFace(faceNet, frame)

if not faceBoxes:
    print("No face detected")

for faceBox in faceBoxes:
    # Draw a rectangle around the face without displaying gender and age
    cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), (0, 255, 0), 2)

# Display the original image without resizing
cv2.namedWindow("Detecting the faces", cv2.WINDOW_NORMAL)
cv2.imshow("Detecting the faces", resultImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
