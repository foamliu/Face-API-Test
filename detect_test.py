import json
from io import BytesIO

import cv2 as cv
import requests
from PIL import Image


def draw_landmarks(image, faces):
    for face in faces:
        landmarks = face['faceLandmarks']
        pupilLeft = landmarks['pupilLeft']
        pupilRight = landmarks['pupilRight']
        noseTip = landmarks['noseTip']
        mouthLeft = landmarks['mouthLeft']
        mouthRight = landmarks['mouthRight']
        eyebrowLeftOuter = landmarks['eyebrowLeftOuter']
        eyebrowLeftInner = landmarks['eyebrowLeftInner']
        eyeLeftOuter = landmarks['eyeLeftOuter']
        eyeLeftTop = landmarks['eyeLeftTop']
        eyeLeftBottom = landmarks['eyeLeftBottom']
        eyeLeftInner = landmarks['eyeLeftInner']
        eyebrowRightInner = landmarks['eyebrowRightInner']
        eyebrowRightOuter = landmarks['eyebrowRightOuter']
        eyeRightInner = landmarks['eyeRightInner']
        eyeRightTop = landmarks['eyeRightTop']
        eyeRightBottom = landmarks['eyeRightBottom']
        eyeRightOuter = landmarks['eyeRightOuter']
        noseRootLeft = landmarks['noseRootLeft']
        noseRootRight = landmarks['noseRootRight']
        noseLeftAlarTop = landmarks['noseLeftAlarTop']
        noseRightAlarTop = landmarks['noseRightAlarTop']
        noseLeftAlarOutTip = landmarks['noseLeftAlarOutTip']
        noseRightAlarOutTip = landmarks['noseRightAlarOutTip']
        upperLipTop = landmarks['upperLipTop']
        upperLipBottom = landmarks['upperLipBottom']
        underLipTop = landmarks['underLipTop']
        underLipBottom = landmarks['underLipBottom']
        landmark_points = [pupilLeft, pupilRight, noseTip, mouthLeft, mouthRight, eyebrowLeftOuter, eyebrowLeftInner,
                           eyeLeftOuter, eyeLeftTop, eyeLeftBottom, eyeLeftInner,
                           eyebrowRightInner, eyebrowRightOuter, eyeRightInner, eyeRightTop, eyeRightBottom,
                           eyeRightOuter, noseRootLeft, noseRootRight, noseLeftAlarTop, noseRightAlarTop,
                           noseLeftAlarOutTip, noseRightAlarOutTip, upperLipTop, upperLipBottom, underLipTop,
                           underLipBottom]
        for pt in landmark_points:
            cv.circle(image, (int(pt['x']), int(pt['y'])), 3, (0, 255, 0), -1)

        segments = [[eyebrowLeftOuter, eyebrowLeftInner],
                    [eyebrowRightInner, eyebrowRightOuter],
                    [eyeRightOuter, eyeRightTop, eyeRightInner, eyeRightBottom, eyeRightOuter],
                    [eyeLeftOuter, eyeLeftTop, eyeLeftInner, eyeLeftBottom, eyeLeftOuter],
                    [noseRootLeft, noseLeftAlarTop, noseLeftAlarOutTip, noseTip, noseRightAlarOutTip, noseRightAlarTop,
                     noseRootRight],
                    [mouthLeft, upperLipTop, mouthRight, upperLipBottom, mouthLeft],
                    [mouthLeft, underLipTop, mouthRight, underLipBottom, mouthLeft]]

        for seg in segments:
            for i in range(len(seg) - 1):
                pt1 = (int(seg[i]['x']), int(seg[i]['y']))
                pt2 = (int(seg[i + 1]['x']), int(seg[i + 1]['y']))
                cv.line(image, pt1, pt2, (0, 255, 0), 1)


if __name__ == '__main__':
    url = 'http://localhost:5000/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=true'

    image_file = BytesIO()
    img = Image.open('images/Donald-Trump.jpg')
    img.save(image_file, "JPEG")
    image_file.seek(0)

    files = [('images', ('test.jpg', image_file, 'image/jpeg'))]
    r = requests.post(url, files=files)
    print(r.text)
    faces = json.loads(r.text)
    print(faces)

    image = cv.imread('images/Donald-Trump.jpg')
    draw_landmarks(image, faces)
    cv.imwrite('images/out.jpg', image)
