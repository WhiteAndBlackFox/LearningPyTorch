# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch


class FaceDetection(object):
    """
        Face detector class and recognise hand
    """

    def __init__(self, mtcnn, path_model, device='cpu'):
        self.mtcnn = mtcnn
        self.device = device
        self.path_model = path_model

    def _draw(self, frame, boxes, probs, landmarks, gestures):
        """
        Draw landmarks and boxes for each face detected
        """
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                # Draw rectangle on frame
                cv2.rectangle(frame,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 0, 255),
                              thickness=2)

                # Show probability
                cv2.putText(frame,
                            gestures, (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1 / 2, (0, 0, 255), 2,
                            cv2.LINE_AA)

        except Exception as ex:
            pass

        return frame

    @staticmethod
    def digit_to_classname(digit):
        if digit == 0:
            return 'palm'
        elif digit == 1:
            return 'finger'
        elif digit == 2:
            return 'fist'
        elif digit == 3:
            return 'fist_moved'
        elif digit == 4:
            return 'thumb'
        elif digit == 5:
            return 'index'
        elif digit == 6:
            return 'ok'
        elif digit == 7:
            return 'palm_moved'
        elif digit == 8:
            return 'C'
        elif digit == 9:
            return 'down'

    def run(self):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        net = torch.load(self.path_model)
        net.eval()

        while True:
            ret, frame = cap.read()
            try:

                h, w, c = frame.shape
                norm_img = np.zeros((w, h))
                frame = cv2.normalize(frame, norm_img, 0, 255, cv2.NORM_MINMAX)

                # detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                if boxes is None:
                    continue

                img_frame = cv2.resize(frame, (170, 64))
                img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
                img_frame = torch.from_numpy(img_frame).unsqueeze(0).to(self.device).float()
                hand = net(img_frame[None, ...])
                gestures = self.digit_to_classname(hand.argmax())

                # draw on frame
                self._draw(frame, boxes, probs, landmarks, f"Maybe you show {gestures}?")
            except Exception as e:
                pass

            # Show the frame
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
