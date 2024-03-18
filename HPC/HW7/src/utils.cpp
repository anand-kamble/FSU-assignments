#include <cmath>
#ifndef __JpegFile
#include "../includes/Jpegfile.h"
#endif
typedef double DIST; // Type for distance between pixels

/**
 * @brief This function computes the Euclidean distance between two pixels.
 *
 * @param dataBuf Pointer to the image pixel data.
 * @param p1 Index of the first pixel.
 * @param generator RGB values of the generator pixel.
 * @return DIST The Euclidean distance between the two pixels.
 */
DIST distanceBetweenPexels(BYTE *dataBuf, BYTE p1, int *generator)
{
    BYTE *pRed1, *pGrn1, *pBlu1;
    pRed1 = dataBuf + p1 * 3;
    pGrn1 = dataBuf + p1 * 3 + 1;
    pBlu1 = dataBuf + p1 * 3 + 2;

    int pRed2, pGrn2, pBlu2;
    pRed2 = generator[0];
    pGrn2 = generator[1];
    pBlu2 = generator[2];

    return sqrt(pow((int)(*pRed1 - pRed2), 2) + pow((int)(*pGrn1 - pGrn2), 2) + pow((int)(*pBlu1 - pBlu2), 2));
}