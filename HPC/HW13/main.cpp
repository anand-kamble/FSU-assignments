#include <cmath>
#include <stdint.h>
#include <iostream>
#include <chrono>
#include "Jpegfile.h"

// Defining a struct to represent a pixel
struct Pixel
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

// Function to calculate the euclidean distance between two pixels
double euclideanDistance(const Pixel *p1, const Pixel *p2)
{
    return sqrt(pow(p1->r - p2->r, 2) + pow(p1->g - p2->g, 2) + pow(p1->b - p2->b, 2));
}

int main()
{
    UINT width, height;
    uint8_t *hostDataBuf = JpegFile::JpegFileToRGB("test-large.jpg", &width, &height);

    const int N = height * width;

    Pixel *colors = new Pixel[N];
    // #pragma acc parallel loop present(hostDataBuf) gang vector

    for (int i = 0; i < N; i++)
    {
        colors[i].r = hostDataBuf[i * 3];
        colors[i].g = hostDataBuf[i * 3 + 1];
        colors[i].b = hostDataBuf[i * 3 + 2];
    }

    int k;
    std::cout << "Enter the number of clusters: ";
    std::cin >> k;
    Pixel *generators = new Pixel[k];

    auto start = std::chrono::high_resolution_clock::now();

// #pragma acc enter data create(generators[0 : k])
#pragma acc data copyin(hostDataBuf[0 : N * 3], generators[0 : k])
    {
#pragma acc parallel loop gang vector
        for (int i = 0; i < k; i++)
        {
            generators[i].r = hostDataBuf[(N * 3 / k) * i];
            generators[i].g = hostDataBuf[(N * 3 / k) * i + 1];
            generators[i].b = hostDataBuf[(N * 3 / k) * i + 2];
        }

        int *groupColorSum; // = new int[k * 3];
        int *groupCount;    // = new int[k];
#pragma acc enter data create(groupColorSum[0 : k * 3], groupCount[0 : k])
#pragma acc data create(colors[0 : N], generators[0 : k], groupColorSum[0 : k * 3], groupCount[0 : k])
        {
            for (int iter = 0; iter < 10; iter++)
            {

                // #pragma acc kernels
#pragma acc parallel loop gang vector reduction(+ : groupColorSum[ : k * 3]) reduction(+ : groupCount[ : k]) present(colors, generators, groupColorSum, groupCount)
                for (int i = 0; i < N; i++)
                {
                    double minDist = INFINITY;
                    int minIndex = 0;

                    for (int j = 0; j < k; j++)
                    {
                        double dist = euclideanDistance(&colors[i], &generators[j]);
                        if (dist < minDist)
                        {
                            minDist = dist;
                            minIndex = j;
                        }
                    }

                    groupColorSum[minIndex * 3] += colors[i].r;
                    groupColorSum[minIndex * 3 + 1] += colors[i].g;
                    groupColorSum[minIndex * 3 + 2] += colors[i].b;
                    groupCount[minIndex]++;
                }

#pragma acc parallel loop gang vector present(generators, groupColorSum, groupCount)
                for (int i = 0; i < k; i++)
                {
                    generators[i].r = groupColorSum[i * 3] / groupCount[i];
                    generators[i].g = groupColorSum[i * 3 + 1] / groupCount[i];
                    generators[i].b = groupColorSum[i * 3 + 2] / groupCount[i];
                }

#pragma acc parallel loop gang vector
                for (int i = 0; i < k * 3; i++)
                    groupColorSum[i] = 0;
#pragma acc parallel loop gang
                for (int i = 0; i < k; i++)
                    groupCount[i] = 0;
            }

#pragma acc parallel loop gang vector present(colors, generators)
            for (int i = 0; i < N; i++)
            {
                double minDist = INFINITY;
                int minIndex = 0;

                for (int j = 0; j < k; j++)
                {
                    double dist = euclideanDistance(&colors[i], &generators[j]);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        minIndex = j;
                    }
                }

                colors[i] = generators[minIndex];
            }
        }

#pragma acc data copyout(colors[0 : N])
        {
#pragma acc parallel loop present(colors, hostDataBuf) gang vector
            for (int i = 0; i < N; i++)
            {
                hostDataBuf[i * 3] = colors[i].r;
                hostDataBuf[i * 3 + 1] = colors[i].g;
                hostDataBuf[i * 3 + 2] = colors[i].b;
            }
        }

        delete[] generators;
// delete[] groupColorSum;
// delete[] groupCount;
#pragma acc exit data delete (groupColorSum[0 : k * 3], groupCount[0 : k])
#pragma acc data copyout(hostDataBuf[0 : N * 3])
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); // This returns in microseconds
    printf("Time taken by function: %d ms\n", duration.count() / 1000);                  // Dividing by 1000 to get milliseconds

    JpegFile::RGBToJpegFile("output.jpg", hostDataBuf, width, height, 100, false);

    delete[] colors;
    free(hostDataBuf);

    return 0;
}