import cv2
import numpy as np
from itertools import combinations
from scipy.spatial import distance as dist
import math
camera = cv2.VideoCapture('vtest.avi')

classesfile = 'coco.names'
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

n = 10
confthreshold = 0.3
nmsthreshold = 0.3
modelConf = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def Check(a,  b):
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    calibration = (a[1] + b[1]) / 2       
    if 0 < dist < 0.2*calibration:
        return True
    else:
        return False

def findObjects(outputs, frame):
    H, W, cT = frame.shape
    box = []
    classIds = []
    conf = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classNames[classId] == 'person':
                if confidence > confthreshold:
                    w, h = int(detect[2]*W), int(detect[3]*H)
                    x, y = int((detect[0]*W) - w/2), int((detect[1]*H) - h/2)
                    box.append([x,y,w,h])
                    classIds.append(classId)
                    conf.append(float(confidence))

    print(len(box))
    box_line = cv2.dnn.NMSBoxes(box, conf, confthreshold, nmsthreshold)
    if len(box_line) > 0:
            flat_box = box_line.flatten()
            pairs = []
            center = [] 
            status = [] 
            for i in flat_box:
                (x, y) = (box[i][0], box[i][1])
                (w, h) = (box[i][2], box[i][3])
                center.append([int(x + w / 2), int(y + h / 2)])
                status.append(0)

            for i in range(len(center)):
                for j in range(i+1, len(center)):
                    close = Check(center[i], center[j])
                    # D = dist.euclidean((center[j][0], center[j][1]), (center[i][0], center[i][1])) / W
                    # print(D)
                    if close:
                        pairs.append([center[i], center[j]])
                        status[i] += 1
                        status[j] += 1
            index = 0
            danger = 0
            for i in flat_box:
                (x, y) = (box[i][0], box[i][1])
                (w, h) = (box[i][2], box[i][3])
                if status[index] > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
                    cv2.putText(frame, str(status[index]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,255), 2, cv2.LINE_AA)
                    danger += 1
                elif status[index] == 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                index += 1
            for h in pairs:
                cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
    text = "No of at-risk people: %s" % str(danger) 
    location = (10,25)
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,255), 2, cv2.LINE_AA) 
    cv2.imshow('video',frame)
    processedImg = frame.copy()




while True:
    sucess, img = camera.read()
    # img = cv2.imread('people-pedestrian-man-woman.jpg')
    blob = cv2.dnn.blobFromImage(img,1/255,(320,320),[0,0,0],1,crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    # print(layerNames)
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    output = net.forward(outputNames)
    print(output[0].shape)

    findObjects(output,img)
    # cv2.imshow('Image', img)
    cv2.waitKey(1)
