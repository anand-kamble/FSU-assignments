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

    // Read the file to dataBuf with RGB format. The width and height of the image are stored in the respective variables.
    dataBuf = JpegFile::JpegFileToRGB("Images/test3.jpg", &width, &height);

    const int N = height * width; // Total number of pixels

    // Set the parameters for the k-means algorithm
    const int k = 5;           // Number of clusters
    const int ITERATIONS = 10; // Number of iterations for the k-means algorithm

    Group *groups = new Group[k]; // Array to store groups

    for (int i = 0; i < k; i++)
    {
        groups[i].setGenerator(dataBuf[(((N * 3) / k) * i)],
                               dataBuf[(((N * 3) / k) * i) + 1],
                               dataBuf[(((N * 3) / k) * i) + 2]);

        groups[i].setpixels((int *)malloc(sizeof(int) * (N * 3))); // N * 3 is the maximum number of pixels,
        // using max number since there can be a situation where all the pixels are in one group.
    }

    int NUM_THREADS = 32;                                  // Number of threads to use for parallelization
    NUM_THREADS = min(NUM_THREADS, omp_get_max_threads()); // Make sure that we don't use more threads than available
    omp_set_num_threads(NUM_THREADS);                      // Set the number of threads

    double start = omp_get_wtime(); // Record the start time for benchmarking

    for (int i = 0; i < ITERATIONS; i++)
    {
#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (int i = 0; i < k; i++)
            {
                groups[i].clearPixels();    // Clear the pixels in each group
                groups[i].setPixelCount(0); // Set the pixel count of each group to 0
            }
#pragma omp for schedule(static)
            for (int i = 0; i < N; i++)
            {
                BYTE *pTest = dataBuf + i * 3;         // Pointer to the current pixel
                DIST minimumDistance = INFINITY;       // Initialize the minimum distance to infinity
                Group *minimumDistanceGroup = nullptr; // Pointer to the group with the minimum distance
                for (int j = 0; j < k; j++)
                {
                    DIST distance = distanceBetweenPexels(dataBuf, i, groups[j].generator); // Compute the distance between the pixel and the generator of the group
                    if (distance < minimumDistance)                                         // If the distance is less than the minimum distance
                    {
                        minimumDistance = distance;        // Update the minimum distance
                        minimumDistanceGroup = &groups[j]; // Update the pointer to the group with the minimum distance
                    }
                }
                if (minimumDistanceGroup != nullptr)
                {
                    minimumDistanceGroup->setPixelCount(minimumDistanceGroup->pixelCount + 1); // Increment the pixel count of the group with the minimum distance
                    minimumDistanceGroup->addPixel(i);                                         // Add the pixel to the group with the minimum distance
                }
            }

#pragma omp for schedule(dynamic)
            for (int i = 0; i < k; i++)
            {
                int numOfPixels = groups[i].getPixelCount();            // Get the number of pixels in the group
                long long int averageR = 0, averageG = 0, averageB = 0; // Initialize the average color of the group to 0
                                                                        // #pragma omp parallel for schedule(static) reduction(+ : averageR, averageG, averageB)
                for (int j = 0; j < numOfPixels; j++)
                {
                    BYTE *pTest = dataBuf + (groups[i].getpixels()[j] * 3); // Pointer to the current pixel

                    // Compute the average color of the group
                    averageR += (*pTest);       // Red
                    averageG += (*(pTest + 1)); // Green
                    averageB += (*(pTest + 2)); // Blue
                }

                if (numOfPixels > 0) // If the group has pixels
                {
                    averageR /= numOfPixels;                              // Normalize the average color red of the group by dividing by the number of pixels
                    averageG /= numOfPixels;                              // Normalize the average color green of the group by dividing by the number of pixels
                    averageB /= numOfPixels;                              // Normalize the average color blue of the group by dividing by the number of pixels
                    groups[i].setGenerator(averageR, averageG, averageB); // Update the generator of the group
                    groups[i].average[0] = averageR;                      // Update the average color red of the group
                    groups[i].average[1] = averageG;                      // Update the average color green of the group
                    groups[i].average[2] = averageB;                      // Update the average color blue of the group
                }
            }
// Color pixels based on group averages
#pragma omp for schedule(dynamic)
            for (int i = 0; i < k; i++)
            {
                auto groupPixels = groups[i].getpixels();
                auto pixelCount = groups[i].getPixelCount();
                auto average = groups[i].average;
                // #pragma omp for schedule(static) // private(i, groupPixels, pixelCount, average)
                for (int j = 0; j < pixelCount; j++)
                {
                    auto pTest = dataBuf + (groupPixels[j] * 3);
                    // std::cout << "Running" << endl;
                    // Coloring the groups
                    *pTest = average[0];
                    *(pTest + 1) = average[1];
                    *(pTest + 2) = average[2];
                }
            }
        }
    }
    double end = omp_get_wtime();

    std::cout << "\rTime for " << NUM_THREADS << " threads : " << end - start << endl;

    // Write the segmented image to a file with the suffix "_seg"
    JpegFile::RGBToJpegFile("test_seg.jpg", dataBuf, width, height, true, 100);

    delete dataBuf;
    delete[] groups;

    return 0;
}