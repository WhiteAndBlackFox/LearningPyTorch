# -*- coding: utf-8 -*-
import argparse
from facenet_pytorch import MTCNN

from FaceDetectionAndHandGestures.FaceDetection import FaceDetection
from FaceDetectionAndHandGestures.HandGestures.HandGesturesTrain import HandGesturesTrain

parser = argparse.ArgumentParser(description='Face detection and hand gestures')
parser.add_argument('-t', '--train', dest='path', help='train NN')
parser.add_argument('-p', '--path', help='Path', required=True)
parser.add_argument('-v', '--video', dest='path', help='Start video recognition cam')
args = parser.parse_args()
print(args)

if args.train is not None:
    if args.path is None:
        print("Please set params `path` where data train")
    hgt = HandGesturesTrain(args.path, 10)
    hgt.fit()
    hgt.save('HandGestures.pth')
elif args.video:
    if args.path is None:
        print("Please set params `path` where trained model")
    mtcnn = MTCNN(keep_all=True).eval()
    fd = FaceDetection(mtcnn, args.path)
    fd.run()
