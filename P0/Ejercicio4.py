import cv2
import numpy as np
from matplotlib import pyplot as plt

def setColor(img):
	prt = img.shape

	if len(prt) == 2:
		img = setColorGrey(img,prt)
	else:
		img = setColorRGB(img,prt)

	return img

def setColorGrey(img, prt):
	for i in range(0,prt[0]):
		for j in range(0,prt[1]):
			img[i,j] = (img[i,j] + 20) % 256

	return img;

def setColorRGB(img,prt):
	for i in range(0,prt[0]):
                for j in range(0,prt[1]):
                        img[i,j] = (img[i,j] + (10,10,10)) % (256,256,256)
	return img

def main():
	namefile = input("Ingresa el path de la imagen: ")
	flags_color = input("0 - Blanco & Negro\n1 - Color RGB\nIntroduzca el modo de lectura: ")


	img = cv2.imread(namefile, int(flags_color))

	img = setColor(img) 

	cv2.imshow("img",img)
	cv2.waitKey(0)

if __name__ == '__main__':
	main()
