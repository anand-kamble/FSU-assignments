#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "../includes/Jpegfile.h"

struct PIXEL
{
    int x, y, z;
};

// Kernel for grouping pixels to nearest generator
__global__ void groupingKernel(UINT width, UINT height, int k, PIXEL *colors, PIXEL *generators, int *groupColorSum, int *groupCount)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        PIXEL pixel = colors[index];

        float minDist = INFINITY;
        int minIndex = 0;

        for (int i = 0; i < k; i++)
        {
            PIXEL gen = generators[i];
            float dist = sqrtf((pixel.x - gen.x) * (pixel.x - gen.x) + (pixel.y - gen.y) * (pixel.y - gen.y) + (pixel.z - gen.z) * (pixel.z - gen.z));
            if (dist < minDist)
            {
                minDist = dist;
                minIndex = i;
            }
        }

        atomicAdd(&groupColorSum[minIndex * 3 + 0], (int)pixel.x);
        atomicAdd(&groupColorSum[minIndex * 3 + 1], (int)pixel.y);
        atomicAdd(&groupColorSum[minIndex * 3 + 2], (int)pixel.z);
        atomicAdd(&groupCount[minIndex], 1);
    }
}

// Kernel for updating generators
__global__ void updateGeneratorsKernel(int k, PIXEL *generators, int *groupColorSum, int *groupCount)
{
    int i = threadIdx.x;
    if (i < k)
    {
        generators[i].x = (float)groupColorSum[i * 3 + 0] / groupCount[i];
        generators[i].y = (float)groupColorSum[i * 3 + 1] / groupCount[i];
        generators[i].z = (float)groupColorSum[i * 3 + 2] / groupCount[i];
    }
}

// Kernel for replacing pixel colors with group generators
__global__ void replaceColorKernel(UINT width, UINT height, int k, PIXEL *colors, PIXEL *generators, int *groupCount)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        PIXEL pixel = colors[index];

        float minDist = INFINITY;
        int minIndex = 0;

        for (int i = 0; i < k; i++)
        {
            PIXEL gen = generators[i];
            float dist = sqrtf((pixel.x - gen.x) * (pixel.x - gen.x) + (pixel.y - gen.y) * (pixel.y - gen.y) + (pixel.z - gen.z) * (pixel.z - gen.z));
            if (dist < minDist)
            {
                minDist = dist;
                minIndex = i;
            }
        }

        colors[index] = generators[minIndex];
    }
}

int main()
{
    // 1. Host reads the jpg file
    UINT width, height;
    uint8_t *hostDataBuf = JpegFile::JpegFileToRGB("test-large.jpg", &width, &height);

    // 2. Allocate memory in Device, copy colors from Host to Device
    PIXEL *d_colors;
    cudaMalloc(&d_colors, width * height * sizeof(PIXEL));
    cudaMemcpy(d_colors, hostDataBuf, width * height * sizeof(PIXEL), cudaMemcpyHostToDevice);

    // 3. Host generates k generates, allocate Device memory for generators, and copy the generates from Host to Device
    int k = 8;
    auto generators = new PIXEL[k];
    for (int i = 0; i < k; i++)
    {
        generators[i].x = hostDataBuf[i * 3];
        generators[i].y = hostDataBuf[i * 3 + 1];
        generators[i].z = hostDataBuf[i * 3 + 2];
    }
    PIXEL *d_generators;
    cudaMalloc(&d_generators, k * sizeof(PIXEL));
    cudaMemcpy(d_generators, generators, k * sizeof(PIXEL), cudaMemcpyHostToDevice);

    // 4. Allocate Device memory for each group color sum, group count, and initialize it to zero
    int *d_groupColorSum;
    int *d_groupCount;
    cudaMalloc(&d_groupColorSum, k * 3 * sizeof(int));
    cudaMalloc(&d_groupCount, k * sizeof(int));
    cudaMemset(d_groupColorSum, 0, k * 3 * sizeof(int));
    cudaMemset(d_groupCount, 0, k * sizeof(int));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 5. Start of iterations from Host
    for (int iter = 0; iter < 10; iter++)
    {
        // 6. Launch the grouping kernel

        groupingKernel<<<gridSize, blockSize>>>(width, height, k, d_colors, d_generators, d_groupColorSum, d_groupCount);
        cudaDeviceSynchronize();

        // 7. Launch the update generators kernel
        updateGeneratorsKernel<<<1, k>>>(k, d_generators, d_groupColorSum, d_groupCount);
        cudaDeviceSynchronize();

        // Reset group color sum and count
        cudaMemset(d_groupColorSum, 0, k * 3 * sizeof(int));
        cudaMemset(d_groupCount, 0, k * sizeof(int));
    }

    // 9. Launch the replace color kernel
    replaceColorKernel<<<gridSize, blockSize>>>(width, height, k, d_colors, d_generators, d_groupCount);
    cudaDeviceSynchronize();

    // 10. Copy colors from Device back to Host
    cudaMemcpy(hostDataBuf, d_colors, width * height * sizeof(PIXEL), cudaMemcpyDeviceToHost);

    // 11. Host write the gray image to a jpg file
    JpegFile::RGBToJpegFile("output.jpg", hostDataBuf, width, height, 100, false);

    // 12. Free memory
    cudaFree(d_colors);
    cudaFree(d_generators);
    cudaFree(d_groupColorSum);
    cudaFree(d_groupCount);
    delete[] hostDataBuf;

    return 0;
}