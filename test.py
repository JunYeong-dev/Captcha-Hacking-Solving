import cv2
import utils

image = cv2.imread('1.png', cv2.IMREAD_COLOR)
blue = utils.get_chars(image.copy(), utils.BLUE)
green = utils.get_chars(image.copy(), utils.GREEN)
red = utils.get_chars(image.copy(), utils.RED)
cv2.imshow('Image Gray', blue)
cv2.waitKey()
cv2.imshow('Image Gray', green)
cv2.waitKey()
cv2.imshow('Image Gray', red)
cv2.waitKey()