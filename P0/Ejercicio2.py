import cv2

def main():
	namefile = input("Ingresa el path de la imagen: ")
	flags_color = input("1 - Blanco & Negro\n2 - Color RGB\nIntroduzca el modo de lectura: ")

	img = cv2.imread(namefile, int(flags_color))

	print img

if __name__ == '__main__':
	main()

