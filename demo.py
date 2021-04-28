import os
from  detection.retinaface import RetinaNetDetector
import mlchain
from skimage import transform as trans
import cv2
import numpy as np 
from copy import deepcopy

from models.net import ModelAgeGender


import onnxruntime


def align_face_by_landmarks(cv_img, dst, dst_w, dst_h):
    if dst_w == 96 and dst_h == 112:
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32)
    elif dst_w == 112 and dst_h == 112:
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] ], dtype=np.float32)
    elif dst_w == 150 and dst_h == 150:
        src = np.array([
            [51.287415, 69.23612],
            [98.48009, 68.97509],
            [75.03375, 96.075806],
            [55.646385, 123.7038],
            [94.72754, 123.48763]], dtype=np.float32)
    elif dst_w == 160 and dst_h == 160:
        src = np.array([
            [54.706573, 73.85186],
            [105.045425, 73.573425],
            [80.036, 102.48086],
            [59.356144, 131.95071],
            [101.04271, 131.72014]], dtype=np.float32)
    elif dst_w == 224 and dst_h == 224:
        src = np.array([
            [76.589195, 103.3926],
            [147.0636, 103.0028],
            [112.0504, 143.4732],
            [83.098595, 184.731],
            [141.4598, 184.4082]], dtype=np.float32)
    else:
        return None
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

    if M is None:
        img = cv_img
        
        #use center crop
        det = np.zeros(4, dtype=np.int32)
        det[0] = int(img.shape[1]*0.0625)
        det[1] = int(img.shape[0]*0.0625)
        det[2] = img.shape[1] - det[0]
        det[3] = img.shape[0] - det[1]

        margin = 44
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        ret = cv2.resize(ret, (dst_w, dst_h))
        return ret 
        
    face_img = cv2.warpAffine(cv_img,M,(dst_w,dst_h), borderValue = 0.0)
    return face_img


def get_face(frame, coord):
    frame = deepcopy(frame)
    height, width = frame.shape[0:2]
    l, t, r, b, _ = coord
    w_f = int(r-l)
    h_f = int(b-t)

    l_f = max(0, int(l - 0.2*w_f))
    t_f = max(0, int(t - 0.2*h_f))
    r_f = min(width, int(r + 0.2*w_f))
    b_f = min(height, int(b + 0.05*h_f))

    face_img = frame[t_f:b_f, l_f:r_f, :]
    return face_img, (l_f, t_f, r_f, b_f)


def draw_face(frame, face_crd, age, gender, keypoint=None):
    x1, y1, x2, y2 = face_crd
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
   
    info_face = "Age:{}-Gender:{}".format(age, gender)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)
    thickness = 1

    if keypoint is not None:
        # l_y = (keypoint[0], keypoint[5])
        # r_y = (keypoint[1], keypoint[6])
        # nos = (keypoint[2], keypoint[7])
        # l_m = (keypoint[3], keypoint[8])
        # r_m = (keypoint[4], keypoint[9])

        # l_y, r_y, nos, l_m, r_m = [(keypoint[i], keypoint[i+5]) for i in range(5)]
        pass
        # cv2.circle(frame, l_y, 3, (255, 0, 0), 2)
        # cv2.circle(frame, r_y, 3, (0, 0, 255), 2)
        # cv2.circle(frame, nos, 3, (0, 255, 0), 2)
        # cv2.circle(frame, l_m, 3, (255, 255, 0), 2)
        # cv2.circle(frame, r_m, 3, (255, 255, 255), 2)

    cv2.putText(frame, info_face, (x1, y2), font, font_scale, color, thickness)
    return frame


def demo_video(video_path=None):
    if video_path is None:
        video_path = 0
    vid = cv2.VideoCapture(video_path)
    detector = RetinaNetDetector()
    estimator = ModelAgeGender()
    estimator.init_model("mobilenet_v2", num_age_classes=81, widen_factor=0.25, pretrained=False)
    # estimator.load_statedict("weights/age_gender_06112020_mbnet0.25.pt")
    # estimator.load_statedict("./weights/orinal_regression_04112020.pt")
    estimator.load_statedict("weights/36_0.8979109327926814_gender_5.966053009033203_age.pt")

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        det_faces = detector.predict(frame)
        if len(det_faces) == 0:
            cv2.imshow("", frame)
            key = cv2.waitKey(1) & 0xff
            if key == ord("q"):
                break
        
        try:
            for ind in range(len(det_faces[0])):
                face_info = det_faces[0][ind]
                keypoint = det_faces[1][ind]
                face_img, face_crd = get_face(frame, face_info)
                
                small_face = detector.predict(face_img)
                keypoint = small_face[1][0]
                l_y, r_y, nos, l_m, r_m = [(keypoint[i], keypoint[i+5]) for i in range(5)]
                keypoint = np.array((l_y, r_y, nos, l_m, r_m))
                face_align = align_face_by_landmarks(face_img, keypoint, 224, 224)
                # face_align = face_img
                cv2.imshow(f"image_{ind}", face_align)
                # age, gender = estimator.predict_image(face_img[:,:, ::-1])
                age, gender = estimator.predict_image(face_align[:,:,::-1])
                frame = draw_face(frame, face_crd, age, gender, keypoint)
        except IndexError:
            print("None face detected!")
        cv2.namedWindow("", cv2.WINDOW_NORMAL)
        cv2.imshow("", frame)
        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            break


def demo_image(image_path):
    detector = RetinaNetDetector()
    estimator = ModelAgeGender()
    estimator.init_model("mobilenet_v2", num_age_classes=81, widen_factor=0.25)
    estimator.load_statedict("weights/ordinal_regression_04112020.pt")

    frame = cv2.imread(image_path)

    det_faces = detector.predict(frame)
    if len(det_faces) != 0:
        for ind in range(len(det_faces)):
        # for face_info in det_faces[0]:
            face_info = det_faces[0][ind]
            keypoint = det_faces[1][ind]
            face_img, face_crd = get_face(frame, face_info)
            age, gender = estimator.predict_image(face_img[:,:, ::-1])
            frame = draw_face(frame, face_crd, age, gender, keypoint)
    
    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    cv2.imshow("", frame)
    key = cv2.waitKey(0) & 0xff


def demo_video_onnx(model_path):
    ort_session = onnxruntime.InferenceSession(model_path)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.resize(frame, (224, 224))
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        ort_inputs = {ort_session.get_inputs()[0].name: image}
        ort_outputs = ort_session.run(None, ort_inputs)
        import ipdb; ipdb.set_trace()
        cv2.imshow("", frame)
        cv2.waitKey(1)



if __name__ == "__main__":
    # demo_video()
    # demo_image("test_dataset/images.jpeg")
    demo_video_onnx("weights/age_gender_mb0.25_08112020.onnx")