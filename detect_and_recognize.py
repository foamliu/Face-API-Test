import json
from datetime import datetime
from io import BytesIO

import cv2 as cv
import requests
from PIL import Image


def get_group():
    name2personId = dict()
    name2personId['Clair'] = '757f5a38-fb98-4762-a92c-0a3de467dfa9'
    name2personId['Jim'] = 'd8977fbc-fcbc-4d57-9d15-7ec36e81e5a7'
    name2personId['Lina'] = 'b8e24131-0418-45ad-af36-61f91c02d678'
    name2personId['May'] = '17546e10-4cef-4f4a-bdc5-2f6fa892204b'
    name2personId['Mike'] = 'd7b0d0e0-b208-4532-b074-576844e3ed62'
    name2personId['Ning'] = '12471913-3c5c-4b75-9a9d-12493258aa7f'
    name2personId['Sam'] = '214b4cf1-27dd-4fde-827f-51420d550730'

    personId2name = dict()
    for key in name2personId.keys():
        personId2name[name2personId[key]] = key

    return personId2name, name2personId


def draw_boxes(image, faceRectangle, name):
    xmin = faceRectangle['left']
    ymin = faceRectangle['top']
    width = faceRectangle['width']
    height = faceRectangle['height']
    pt1 = (xmin, ymin)
    pt2 = (xmin + width, ymin + height)
    cv.rectangle(image, pt1, pt2, (0, 255, 0), 1)
    cv.putText(image, name, (xmin + 1, ymin + 1), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(image, name, (xmin, ymin), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), lineType=cv.LINE_AA)


def process_one_frame(image):
    detect_url = 'http://localhost:5000/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=true'

    image_file = BytesIO()
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img = Image.fromarray(image_rgb)
    img.save(image_file, "JPEG")
    image_file.seek(0)

    files = [('images', ('test.jpg', image_file, 'image/jpeg'))]
    r = requests.post(detect_url, files=files)
    # print(r.text)
    face_boxes = json.loads(r.text)

    faceId_list = []
    for face in face_boxes:
        faceId_list.append(face['faceId'])

    # print(len(faceId_list))
    # print(faceId_list)

    req = {
        "largePersonGroupId": "f91aa966-b175-11e8-aeb1-d89ef339b7b0",
        "faceIds": faceId_list,
        "maxNumOfCandidatesReturned": 3,
        "confidenceThreshold": 0.5
    }
    identify_url = 'http://localhost:5000/face/v1.0/identify'
    r = requests.post(identify_url, json=req)
    # print(r.text)

    face_identities = json.loads(r.text)
    for face in face_identities:
        if face['candidates']:
            faceId = face['faceId']
            personId = face['candidates'][0]['personId']
            name = personId2name[personId]
            faceRectangle = [f['faceRectangle'] for f in face_boxes if f['faceId'] == faceId][0]
            # print(name)
            # print(faceRectangle)
            draw_boxes(image, faceRectangle, name)

    return image


# if __name__ == '__main__':
#     personId2name, name2personId = get_group()
#     image = cv.imread('images/scene.jpg')
#     image = process_one_frame(image)
#     cv.imwrite('images/face_recognition.png', image)

if __name__ == '__main__':
    personId2name, name2personId = get_group()
    video = 'video/movie_1080p.mp4'
    cap = cv.VideoCapture(video)
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out = cv.VideoWriter('video/output.avi', fourcc, 24.0, (1920, 1080))
    frame_idx = 0
    t1 = datetime.now()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        vis = process_one_frame(frame)
        cv.imshow('frame', vis)
        out.write(vis)
        ch = cv.waitKey(1)
        frame_idx += 1
        if ch == 27:
            break

    t2 = datetime.now()
    delta = t2 - t1
    elapsed = delta.seconds + delta.microseconds / 1E6

    print('fps: ' + str(frame_idx / elapsed))
