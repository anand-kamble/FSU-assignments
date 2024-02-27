all:
	mpiCC -o ColorToGray.exe Jpegfile.cpp ColorToGray.cpp JpegLib/libjpeg.a

run:
	mpirun -np 4 ./ColorToGray.exe