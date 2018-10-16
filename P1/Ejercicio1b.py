import cv2 as cv
import sys

def main():
	if len(sys.argv) != 4:
		print 'Uso: ' + sys.argv[0] + ' <dx> <dy> <tamano_kernel>\n'

	dx = int(sys.argv[1])
	dy = int(sys.argv[2])
	ksize = int(sys.argv[3])
	

	kx, ky = cv.getDerivKernels(dx,dy,ksize)

	print 'dx = ' , dx , ' , dy = ', dy, '\n'
	print 'Deriv X: \n' , kx
	print 'Deriv Y: \n' , ky

if __name__ == '__main__':
	main()
