import cv2 as cv
import sys

def main():
	if len(sys.argv) != 5:
		print 'Uso: ' + sys.argv[0] + '<ruta_imagen> <tamano_kernel> <sigma> <write_bit>\n'
		sys.exit(0)

	ruta_imagen = sys.argv[1]
	kernel = (int(sys.argv[2]), int(sys.argv[2]))
	sigma = int(sys.argv[3])
	write_bit = int(sys.argv[4])
	
	img = cv.imread(ruta_imagen, cv.IMREAD_GRAYSCALE )
	img =  cv.GaussianBlur(img,kernel,sigma)

	if(write_bit == 1):
		cv.imwrite('Documentacion/gaussiana' + str(kernel) + str(sigma) + '.bmp', img)

	else:
		cv.imshow('gaussiana',img)
		cv.waitKey(0)
		cv.destroyAllWindows()

	
if __name__ == '__main__':
	main()
