#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import os
sys.path.append(os.getcwd() + "/augmentor")

import numpy as np

from . import constants as const
from . import noise as noise
from . import shadow_ellipse as ellipse
from . import shadow_polygon as polygon
from . import shadow_single as single


def add_n_random_shadows(image, n_shadow  = 4, blur_scale = 1.0):
	intensity = np.random.uniform(const.MIN_SHADOW, const.MAX_SHADOW)
	return add_n_shadows(image, n_shadow, intensity, blur_scale)


def add_n_shadows(image, n_shadow = 4, intensity = 0.5, blur_scale = 1.0):
	for i in range(n_shadow ):
		blur_width = noise.get_blur_given_intensity(intensity, blur_scale)

		choice = np.random.uniform(0, 6)
		if choice < 1:
			image = polygon.add_n_triangles_shadow(image, intensity, blur_width)
		elif choice < 2:
			image = polygon.add_n_triangles_light(image, intensity, blur_width)
		elif choice < 3:
			image = single.add_single_light(image, intensity, blur_width)
		elif choice < 4:
			image = single.add_single_shadow(image, intensity, blur_width)
		elif choice < 5:
			image = ellipse.add_ellipse_light(image, intensity, blur_width)
		else:
			image = ellipse.add_ellipse_shadow(image, intensity, blur_width)

	return image
