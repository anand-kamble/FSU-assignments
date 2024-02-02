/**
 * @file ColorToGray.cpp
 * @brief Convert a color image to a gray image using OpenMP parallelization.
 *
 * This program reads a color image in RGB format, converts it to gray luminance,
 * and saves the result as a new jpg file. The conversion process is parallelized
 * using OpenMP to achieve better performance.
 *
 * @name Student Name: Anand Kamble
 * @date Date: 25th Jan 2024
 *
 */

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "Jpegfile.h"


int main()
{
	// Variables to store image properties and pixel data
	UINT height;
	UINT width;
	BYTE* dataBuf;

	//read the file to dataBuf with RGB format
	dataBuf = JpegFile::JpegFileToRGB("test-huge.jpg", &width, &height);

	//the following code convert RGB to gray luminance.
	int row, col;

	int threads[] = { 1, 2, 4, 8, 12 };

	for (int t : threads)
	{
		// Record the start time for benchmarking
		double start = omp_get_wtime();
		// OpenMP parallelization for the RGB to gray luminance conversion
#pragma omp parallel num_threads(t) default(none) firstprivate(col) shared(dataBuf, width, height)
		{
#pragma omp for  
			for (row = 0; row < static_cast<int>(height); row++) {
				for (col = 0; col < static_cast<int>(width); col++) {
					BYTE* pRed, * pGrn, * pBlu;
					pRed = dataBuf + row * width * 3 + col * 3;
					pGrn = dataBuf + row * width * 3 + col * 3 + 1;
					pBlu = dataBuf + row * width * 3 + col * 3 + 2;

					// luminance
					int lum = (int)(.299 * (double)(*pRed) + .587 * (double)(*pGrn) + .114 * (double)(*pBlu));

					*pRed = (BYTE)lum;
					*pGrn = (BYTE)lum;
					*pBlu = (BYTE)lum;
				}
			}
		}
#pragma omp barrier
		// Record the end time for benchmarking
		double end = omp_get_wtime();
		// Print the execution time for the current thread count
		printf("Time for %2d threads : %f\n", t, end - start);
	}



	//write the gray luminance to another jpg file
	JpegFile::RGBToJpegFile("testmono.jpg", dataBuf, width, height, true, 75);

	delete dataBuf;
	return 0;
}