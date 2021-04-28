import cv2
import numpy as np 
import imgaug.augmenters as iaa
from custom_augmentation import (LightFlare, ParallelLight, SpotLight)

aug = iaa.Sequential([
    iaa.OneOf([
        iaa.Sometimes(0.2, LightFlare()),
        iaa.Sometimes(0.2, ParallelLight()),
        iaa.Sometimes(0.2, SpotLight())
    ]),
    # iaa.Sometimes(0.025, iaa.PerspectiveTransform(scale=(0.01, 0.1), keep_size=True)),
    iaa.Sometimes(0.1,
        iaa.OneOf([
            iaa.GaussianBlur((0, 1.5)),
            iaa.AverageBlur(k=(3, 5)),
            iaa.MedianBlur(k=(3, 5)),
            iaa.MotionBlur(k=(3, 5), angle=(-45, 45))
        ])
    ),
    iaa.Sometimes(0.2,
        iaa.Affine(
            # scale=(0.98, 1.5),
            # translate_percent=(0.08),
            rotate=(-15, 15),
            #shear=(-5, 5)
        )
    )
])

# img = cv2.imread
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    aug_frame = aug.augment_image(frame)
    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    cv2.imshow("", aug_frame)
    key = cv2.waitKey(500)
    if key == 27:
        break