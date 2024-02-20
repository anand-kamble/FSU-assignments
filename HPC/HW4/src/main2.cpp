/**
 * @file main.cpp
 * @brief This file contains the main function for image segmentation using the k-means algorithm with OpenMP parallelization.
 *
 * This program performs image segmentation using the k-means clustering algorithm.
 * It reads an image file, divides the pixels into k groups, and iteratively updates
 * the group centroids until convergence. Finally, it colors the pixels based on the
 * average color of each group and saves the segmented image to a file.
 *
 * @name Author: Student Name: Anand Kamble
 * @date Date: 12th Feb 2024
 *
 * @note Make sure to include the necessary dependencies:
 *   - iostream
 *   - cmath
 *   - omp.h
 *   - Jpegfile.h (Assuming it's the header for handling JPEG files)
 *   - Group.h (Assuming it's the header for the Group class)
 *
 * @note Classes:
 *   - `Group`: Class for storing groups of pixels and their generators.
 *
 * @note Functions:
 *   - `DIST distanceBetweenPexels(BYTE *dataBuf, int p1, int *generator)`: Computes the distance between two pixels.
 *   - `int main()`: Main function for image segmentation using k-means algorithm with OpenMP parallelization.
 *
 * @note Usage:
 *   - Set the image file path, number of clusters (k), and the desired number of threads in the main function.
 *   - Compile the code with appropriate flags for OpenMP support.
 *   - Adjust the number of threads using the `omp_set_num_threads` function call.
 *
 * @note Parallelization Details:
 *   - Three parallel loops are used: for grouping pixels, updating group centroids, and coloring pixels.
 *   - OpenMP directives are used to parallelize the loops and manage thread synchronization.
 *
 * @warning Ensure proper compilation with OpenMP support for parallel execution.
 */

#include <iostream>
#include <cmath>
#include <omp.h>
#include "../includes/Jpegfile.h"
#include "Group.cpp"

typedef double DIST; // Type for distance between pixels

/**
 * @brief This function computes the Euclidean distance between two pixels.
 *
 * @param dataBuf Pointer to the image pixel data.
 * @param p1 Index of the first pixel.
 * @param generator RGB values of the generator pixel.
 * @return DIST The Euclidean distance between the two pixels.
 */
DIST distanceBetweenPexels(BYTE *dataBuf, int p1, int *generator)
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

/**
 * @brief The main function of the program.
 *
 * This function performs image segmentation using the k-means clustering algorithm.
 * It reads an image file, divides the pixels into k groups, and iteratively updates
 * the group centroids until convergence. Finally, it colors the pixels based on the
 * average color of each group and saves the segmented image to a file.
 *
 * @return int The exit status of the program.
 */
int main()
{
    // Variables to store image properties and pixel data
    UINT height;
    UINT width;
    BYTE *dataBuf;

    DIST i = 0;

    // read the file to dataBuf with RGB format
    dataBuf = JpegFile::JpegFileToRGB("Images/test2.jpg", &width, &height);

    int k = 3;              // Number of clusters
    int N = height * width; // Total number of pixels

    int *pexelGroup = new int[N];
    int *generators = new int[k * 3];
    int *average = new int[k * 3];

    for (int i = 0; i < k; i++)
    {
        // groups[i].setGenerator(dataBuf[(((N * 3) / k) * i)],
        //                        dataBuf[(((N * 3) / k) * i) + 1],
        //                        dataBuf[(((N * 3) / k) * i) + 2]);

        generators[i] = dataBuf[(((N * 3) / k) * i)];
        generators[i + 1] = dataBuf[(((N * 3) / k) * i) + 1];
        generators[i + 2] = dataBuf[(((N * 3) / k) * i) + 2];

        // groups[i].setpixels((int *)malloc(sizeof(int) * (N * 3))); // N * 3 is the maximum number of pixels,
        // using max number since there can be a situation where all the pixels are in one group.
    }

    for (int i = 0; i < N; i++)
    {
        DIST minDistance = INFINITY;
        for (int j = 0; j < k; j++)
        {
            auto distance = distanceBetweenPexels(dataBuf, i, &generators[j * 3]);
            if (distance < minDistance)
            {
                minDistance = distance;
                pexelGroup[i] = j;
            }
        }
    }

    for (int i = 0; i < k; i++)
    {
        auto ar = average[i * 3];
        auto ag = average[i * 3 + 1];
        auto ab = average[i * 3 + 2];

        for (int j = 0; j < N; j++)
        {
            if (pexelGroup[j] == i)
            {
                ar += dataBuf[pexelGroup[j]];
                ag += dataBuf[pexelGroup[j] + 1];
                ab += dataBuf[pexelGroup[j] + 2];
            }
        }
        // std::cout << "Runnign\n";
        ar /= N;
        ag /= N;
        ab /= N;
    }

    for (int i = 0; i < N; i++)
    {
        dataBuf[i] = average[pexelGroup[i] * 3];
        dataBuf[i + 1] = average[pexelGroup[i + 1] * 3];
        dataBuf[i + 2] = average[pexelGroup[i + 2] * 3];
    }

    double end = omp_get_wtime();

    // cout << "Time for " << NUM_THREADS << " threads : " << end - start << endl;

    // write the gray luminance to another jpg file
    JpegFile::RGBToJpegFile("testseg.jpg", dataBuf, width, height, true, 100);

    std::cout << "Complete\n";

    delete dataBuf;

    return 0;
}