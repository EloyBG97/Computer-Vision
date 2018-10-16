import cv2
import numpy as np

def pintaIM(imgs):
	idx = 0
	for img in imgs:
		cv2.imshow("img" + idx,img)
		idx = idx + 1

	return img


def main():

	namefile = input("Ingresa el path de la imagen: ")
	flags_color = input("0 - Blanco & Negro\n1 - Color RGB\nIntroduzca el modo de lectura: ")

	imgs = np.array(cv2.imread(namefile, int(flags_color)))

	namefile = input("Ingresa el path de la imagen: ")
	flags_color = input("0 - Blanco & Negro\n1 - Color RGB\nIntroduzca el modo de lectura: ")

	np.append(imgs, cv2.imread(namefile, int(flags_color)))

	pintaIM(imgs)

	cv2.waitKey(0)
 
if __name__ == '__main__':
	main()
