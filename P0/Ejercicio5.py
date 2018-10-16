import cv2
import numpy as np
from matplotlib import pyplot as plt

def pintaIM(imgs, titles):
	max_idx = len(imgs);
	idx = 1

	
	for i in range(0, max_idx):
		plt.subplot(1, max_idx, idx)
		
		if len(imgs[i].shape) == 3:
			b,g,r = cv2.split(imgs[i])
			imgs[i] = cv2.merge((r,g,b))

		plt.imshow(imgs[i], 'gray')
		plt.title(titles[i])
		idx = idx + 1

	return plt


def main():
	imgs = []

	namefile = input("Ingresa el path de la imagen: ")
	flags_color = input("0 - Blanco & Negro\n1 - Color RGB\nIntroduzca el modo de lectura: ")


	img = cv2.imread(namefile, int(flags_color))
	imgs.append(img)

	namefile = input("Ingresa el path de la imagen: ")
	flags_color = input("0 - Blanco & Negro\n1 - Color RGB\nIntroduzca el modo de lectura: ")

	img = cv2.imread(namefile, int(flags_color))
	imgs.append(img)

	pintaIM(imgs, ("img1","img2"))

	plt.show()

	cv2.waitKey(0)
 
if __name__ == '__main__':
	main()
