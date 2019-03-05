import numpy as np
import cv2
import torch
from PIL import Image
from mtcnn_pytorch import PNet, RNet, ONet
from mtcnn_pytorch import nms, calibrate_box, get_image_boxes, convert_to_square
from mtcnn_pytorch import run_first_stage
from mtcnn_pytorch import get_reference_facial_points, warp_and_crop_face

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys

# device = 'cpu'

class MTCNN:
    def __init__(self):
        self._p_net = PNet().to(device)
        self._r_net = RNet().to(device)
        self._o_net = ONet().to(device)
        self._o_net.eval()
        self._r_net.eval()
        self._o_net.eval()
        self._reference = get_reference_facial_points(default_square=True)

    def align(self, image, crop_size=(128, 128), scale=3.0):
        assert isinstance(image, np.ndarray)
        pil_image = Image.fromarray(image)
        _, landmarks = self.detect_faces(pil_image)
        sys.stdout.write("len(landmarks[0]): %d \n" % len(landmarks[0]))
        face_5_key_points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        # self.view_landmarks(image, np.array(face_5_key_points))
        warped_face = self.warp_and_crop_face(image, face_5_key_points, crop_size, scale)
        # warped_face = warp_and_crop_face(np.array(image), face_5_key_points, self._reference, crop_size=crop_size)
        return warped_face

    def view_landmarks(self, image, landmarks):
        """
        :type image: np.ndarray
        :type landmarks: np.ndarray
        """
        im = np.asarray(image, dtype=np.uint8).copy()
        landmarks = np.float32(landmarks)
        for i in range(landmarks.shape[0]):
            cv2.circle(im, tuple(landmarks[i]), 1, (0, 255, 0))
            cv2.imshow('cv2.circle', im)
        cv2.waitKey(50)

    def warp_and_crop_face(self, image, landmarks, crop_size=(128, 128), scale=3.0):
        """
        :param scale:
        :param image:
        :type image
        :param landmarks:
        :type landmarks list
        :param crop_size: tuple
        :return: warp_and_crop_face
        """
        landmarks = np.array(landmarks)
        # compute rotate angle, r_angle=arctan((y1-y2)/(x1-x2))
        r_angle = np.arctan((landmarks[0][1]-landmarks[1][1]) /
                            (landmarks[0][0]-landmarks[1][0]))
        r_angle = 180*r_angle/np.pi
        # get rotation matrix
        rot_mat = cv2.getRotationMatrix2D(tuple(landmarks[2]), r_angle, scale=1)
        # rotate image
        rotated_image = cv2.warpAffine(image, rot_mat, dsize=image.shape[0:2])

        # compute crop size
        cropped_image = self._crop_image(rotated_image, landmarks, scale)
        # compute box
        # resize image
        resize_image = cv2.resize(cropped_image, crop_size)

        return resize_image

    def _crop_image(self, image, landmarks, scale=3.0):
        # compute the distance between left eye(landmarks[0]) and left mouth point[landmarks[3]]
        distance = np.sqrt(np.sum(np.square(landmarks[0]-landmarks[3])))
        size = distance * scale
        # compute row_start, row_end, col_start, col_end
        nose_point = landmarks[2]
        row_start = int(nose_point[1]-size/2)
        row_end = int(nose_point[1]+size/2)
        col_start = int(nose_point[0]-size/2)
        col_end = int(nose_point[0]+size/2)
        # make range valid
        row_start = row_start if row_start > 0 else 0
        row_end = row_end if row_end < image.shape[0] else image.shape[0]
        col_start = col_start if col_start > 0 else 0
        col_end = col_end if col_end < image.shape[1] else image.shape[1]
        # crop image
        cropped_image = image[row_start:row_end, col_start:col_end]
        rows, cols, _ = cropped_image.shape
        _min = np.min((rows, cols))
        if _min < cols:
            # rows is smaller than cols
            padding = np.zeros(shape=(cols-_min, cols, 3), dtype=np.uint8)
            cropped_image = np.vstack((cropped_image, padding))
        elif _min < rows:
            # cols is smaller than rows
            padding = np.zeros(shape=(rows, rows - _min, 3), dtype=np.uint8)
            cropped_image = np.hstack((cropped_image, padding))
        return cropped_image

    def align_multi(self, img, limit=None, min_face_size=30.0):
        assert isinstance(img, np.ndarray)
        img = Image.fromarray(img)
        boxes, landmarks = self.detect_faces(img, min_face_size)
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, self._reference, crop_size=(112, 112))
            faces.append(warped_face)
        return boxes, faces

    def detect_faces(self, image, min_face_size=20.0,
                     thresholds=(0.6, 0.7, 0.8),
                     nms_thresholds=(0.7, 0.7, 0.7)):
        """
        Arguments:
            image: an instance of PIL.Image.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """
        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        with torch.no_grad():
            # run P-Net on different scales
            for s in scales:
                boxes = run_first_stage(image, self._p_net, scale=s, threshold=thresholds[0])
                bounding_boxes.append(boxes)

            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            bounding_boxes = np.vstack(bounding_boxes)

            keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
            bounding_boxes = bounding_boxes[keep]

            # use offsets predicted by pnet to transform bounding boxes
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            # shape [n_boxes, 5]

            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 2

            img_boxes = get_image_boxes(bounding_boxes, image, size=24)
            img_boxes = torch.FloatTensor(img_boxes).to(device)

            output = self._r_net(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]

            keep = nms(bounding_boxes, nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 3

            img_boxes = get_image_boxes(bounding_boxes, image, size=48)
            if len(img_boxes) == 0:
                return [], []
            img_boxes = torch.FloatTensor(img_boxes).to(device)
            output = self._o_net(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]

        return bounding_boxes, landmarks
