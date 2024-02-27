
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "Jpegfile.h"

#define MASTER 0

int main(int argc, char *argv[])
{

	int id, p;
	double wtime;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	UINT height = 0;
	UINT width = 0;
	BYTE *dataBuf;

	if (id == MASTER)
	{
		dataBuf = JpegFile::JpegFileToRGB("test-huge.jpg", &width, &height);
		printf("id: %d, width: %d, height: %d\n", id, width, height);
	}

	MPI_Bcast(&height, 1, MPI_UINT16_T, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&width, 1, MPI_UINT16_T, MASTER, MPI_COMM_WORLD);

	printf("id: %d, width: %d, height: %d\n", id, width, height);

	int rowsPerProcess = height / p;
	int remainingRows = height % p;

	printf("id: %d, rowsPerProcess: %d\n", id, rowsPerProcess);

	auto workerRows = height / p;
	// printf("id: %d, rowsPerProcess: %d, remainingRows: %d\n", id, rowsPerProcess, remainingRows);
	auto rowsToProcess = workerRows;
	if (id == MASTER)
	{
		rowsToProcess += remainingRows;
	}

	auto workerDataSize = (int)(rowsToProcess * width * 3 * sizeof(BYTE));
	auto workerData = (BYTE *)malloc(workerDataSize);
	auto scatterCount = workerRows * width * 3;
	// printf("id: %d, rows: %d, scatterCount: %d\n", id, workerRows, scatterCount);
	// if (id == MASTER)
	// {
	MPI_Scatter(dataBuf, scatterCount, MPI_BYTE, workerData, scatterCount, MPI_BYTE, MASTER, MPI_COMM_WORLD);
	// }

	// the following code convert RGB to gray luminance.
	UINT row, col;
	printf("id: %d, starting to process.\n", id);
	for (row = 0; row < rowsToProcess; row++)
	{
		for (col = 0; col < width; col++)
		{
			BYTE *pRed, *pGrn, *pBlu;
			pRed = workerData + row * width * 3 + col * 3;
			pGrn = workerData + row * width * 3 + col * 3 + 1;
			pBlu = workerData + row * width * 3 + col * 3 + 2;

			// luminance
			int lum = (int)(.299 * (double)(*pRed) + .587 * (double)(*pGrn) + .114 * (double)(*pBlu));

			*pRed = (BYTE)lum;
			*pGrn = (BYTE)lum;
			*pBlu = (BYTE)lum;
		}
	}
	printf("id: %d, done with process.\n", id);

	// if (id == MASTER)
	// {

	MPI_Gather(workerData, scatterCount, MPI_BYTE, dataBuf, scatterCount, MPI_BYTE, MASTER, MPI_COMM_WORLD);
	// }

	if (id == MASTER)
	{
		// write the gray luminance to another jpg file
		JpegFile::RGBToJpegFile("testmono.jpg", dataBuf, width, height, true, 75);
		delete dataBuf;
	}

	delete workerData;
	MPI_Finalize();

	return 0;
}