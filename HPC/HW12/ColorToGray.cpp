#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "Jpegfile.h"

int main()
{
	UINT height;
	UINT width;
	BYTE *dataBuf;
	// read the file to dataBuf with RGB format
	dataBuf = JpegFile::JpegFileToRGB("test-huge.jpg", &width, &height);

	// the following code convert RGB to gray luminance.
	UINT row, col;

	/**
	 * I'm using chrono to measure the time taken by the function.
	 * Found the following code from https://chat.openai.com/share/16b73d07-0a31-441e-9f36-b781e5b4d61c
	 */
	auto start = std::chrono::high_resolution_clock::now();

//
/**
 * Using #pragma acc kernels, gave following message:
 * Complex loop carried dependence of pBlu->,pGrn->,pRed-> prevents parallelization
 *
 * Thats why I am using #pragma acc parallel loop collapse(2)
 */
#pragma acc data copy(dataBuf[0 : width * height * 3])
	{
#pragma acc parallel loop collapse(2)
		for (row = 0; row < height; row++)
		{
			for (col = 0; col < width; col++)
			{
				BYTE *pRed, *pGrn, *pBlu;
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
	} // Here the data is copied back to the host

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); // This returns in microseconds
	printf("Time taken by function: %d ms\n", duration.count() / 1000);					 // Dividing by 1000 to get milliseconds
	// write the gray luminance to another jpg file
	JpegFile::RGBToJpegFile("testmono.jpg", dataBuf, width, height, true, 75);

	delete dataBuf;
	printf("ColorToGray done.\n");
	return 1;
}