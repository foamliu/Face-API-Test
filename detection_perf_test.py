import json

import cv2 as cv
import requests


def draw_boxes(image, faces):
    for face in faces:
        box = face['faceRectangle']
        xmin = box['left']
        ymin = box['top']
        width = box['width']
        height = box['height']
        xmax = xmin + width
        ymax = ymin + height
        cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)


if __name__ == '__main__':
    video = 'video/movie.mp4'
    url = 'http://localhost:5000/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=true'
    cap = cv.VideoCapture(video)
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out = cv.VideoWriter('video/output.avi', fourcc, 24.0, (1920, 1080))
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        vis = frame

        cv.imwrite('temp.jpg', frame)
        files = [('images', ('temp.jpg', open('temp.jpg', 'rb'), 'image/jpeg'))]
        r = requests.post(url, files=files)
        faces = json.loads(r.text)

        draw_boxes(vis, faces)

        cv.imshow('frame', vis)
        out.write(vis)
        ch = cv.waitKey(1)
        if ch == 27:
            break
