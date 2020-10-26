#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import os
# sys.path.append(os.getcwd() + "/augmentor")

import random

import cv2
import numpy as np
from PIL import ImageEnhance, Image

from . import constants
from . import channels as channels
from . import shadow as shadow


def get_enhanced_image(image, enhancement = None):
	if enhancement == None:
		enhancement = constants.Enhancement.get_random()

	pil_im = Image.fromarray(image)
	if enhancement == constants.Enhancement.brightness:
		factor = np.random.uniform(
					constants.MIN_BRIGHTNESS,
					constants.MAX_BRIGHTNESS,
					)
		enhancer = ImageEnhance.Brightness(pil_im)

	elif enhancement == constants.Enhancement.contrast:
		factor = np.random.uniform(
					constants.MIN_CONTRAST,
					constants.MAX_CONTRAST,
					)
		enhancer = ImageEnhance.Contrast(pil_im)

	else:
		factor = np.random.uniform(
					constants.MIN_SHARPNESS,
					constants.MAX_SHARPNESS,
					)
		enhancer = ImageEnhance.Sharpness(pil_im)

	enhanced = enhancer.enhance(factor)
	return np.array(enhanced)


def random_color_shade(image, channels_format = None):
	if channels_format == None:
		channels_format = constants.Channels.get_random()

	intensity = np.random.uniform(
					constants.MIN_COLOR_SHADE,
				      	constants.MAX_COLOR_SHADE,
				      	)
	if channels.is_monochannel(image):
	 	image = change_random_channel_gray(image, intensity)
	elif channels_format == constants.Channels.bgr:
		image = change_random_bgrchannel(image, intensity)
	elif channels_format == constants.Channels.hsv:
		image = change_random_hsvchannel(image, intensity)
	else:
		image = change_random_hlschannel(image, intensity)

	return image


def change_random_channel_gray(image, intensity = 0.2):
	image = change_random_bgrchannel(image, intensity)
	return channels.get_gray_image(image)


def change_random_bgrchannel(image, intensity = 0.2):
	b, g, r = channels.get_bgr_channels(image)
	channel_choice = random.randint(1, 3)
	n_shadow = random.randint(1, 4)
	return __change_channel(channel_choice, intensity, n_shadow, b, g, r)


# makes senses only with color images
def change_random_hsvchannel(image, intensity = 0.2):
	h, s, v = channels.get_hsv_channels(image)
	channel_choice = random.randint(1, 3)
	if channel_choice == 2:
		intensity *= 3
	elif channel_choice == 3:
		intensity *= 0.7

	n_shadow = random.randint(1, 4)
	image = __change_channel(channel_choice, intensity, n_shadow, h, s, v)

	return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


# makes senses only with color images
def change_random_hlschannel(image, intensity = 0.2):
	h, l, s = channels.get_hls_channels(image)
	channel_choice = random.choice((1, 3))
	if channel_choice == 3:
		intensity *= 3

	n_shadow = random.randint(1, 4)
	image = __change_channel(channel_choice, intensity, n_shadow, h, l, s)

	return cv2.cvtColor(image, cv2.COLOR_HLS2BGR)


def __change_channel(channel_choice, intensity, n_shadow, channel_1, channel_2,  channel_3):
	blur_scale = 1.2
	if channel_choice == 1:
		channel_1 = shadow.add_n_shadows(channel_1, n_shadow, intensity, blur_scale)
	elif channel_choice == 2:
		channel_2 = shadow.add_n_shadows(channel_2, n_shadow, intensity, blur_scale)
	else:
		channel_3 = shadow.add_n_shadows(channel_3, n_shadow, intensity, blur_scale)

	return cv2.merge((channel_1, channel_2, channel_3))
