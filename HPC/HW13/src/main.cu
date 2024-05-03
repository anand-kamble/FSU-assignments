#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>
#include <iostream>
#include "../includes/Jpegfile.h"

// Defining a struct to represent a pixel
struct Pixel
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

// Function to calculate the euclidean distance between two pixels
__device__ double euclideanDistance(const Pixel *p1, const Pixel *p2)
{
    double sum = 0.0;
    return sqrt(pow(p1->r - p2->r, 2) + pow(p1->g - p2->g, 2) + pow(p1->b - p2->b, 2));
}

// Grouping kernel using shared memory
__global__ void groupingKernel(UINT width, UINT height, int k, Pixel *colors, Pixel *generators, int *groupColorSum, int *groupCount)
{
    extern __shared__ Pixel s_generators[];
    extern __shared__ float s_groupColorSum[];
    extern __shared__ int s_groupCount[];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        Pixel pixel = colors[index];

        // Load generators into shared memory
        if (threadIdx.x < k)
        {
            s_generators[threadIdx.x] = generators[threadIdx.x];
            s_groupColorSum[threadIdx.x * 3 + 0] = 0;
            s_groupColorSum[threadIdx.x * 3 + 1] = 0;
            s_groupColorSum[threadIdx.x * 3 + 2] = 0;
            s_groupCount[threadIdx.x] = 0;
        }
        __syncthreads();

        float minDist = INFINITY;
        int minIndex = 0;

        // Find the nearest generator
        for (int i = 0; i < k; i++)
        {
            Pixel gen = s_generators[i];
            float dist = sqrtf((pixel.r - gen.r) * (pixel.r - gen.r) + (pixel.g - gen.g) * (pixel.g - gen.g) + (pixel.b - gen.b) * (pixel.b - gen.b));
            if (dist < minDist)
            {
                minDist = dist;
                minIndex = i;
            }
        }

        // Accumulate the color sum and count in shared memory
        atomicAdd(&s_groupColorSum[minIndex * 3 + 0], (int)pixel.r);
        atomicAdd(&s_groupColorSum[minIndex * 3 + 1], (int)pixel.g);
        atomicAdd(&s_groupColorSum[minIndex * 3 + 2], (int)pixel.b);
        atomicAdd(&s_groupCount[minIndex], 1);

        __syncthreads();

        // Write the shared memory results back to global memory
        if (threadIdx.x == 0)
        {
            for (int i = 0; i < k; i++)
            {
                generators[i] = s_generators[i];
                groupColorSum[i * 3 + 0] = s_groupColorSum[i * 3 + 0];
                groupColorSum[i * 3 + 1] = s_groupColorSum[i * 3 + 1];
                groupColorSum[i * 3 + 2] = s_groupColorSum[i * 3 + 2];
                groupCount[i] = s_groupCount[i];
            }
        }
    }
}

// Update generators kernel
__global__ void updateGeneratorsKernel(int k, Pixel *generators, int *groupColorSum, int *groupCount)
{
    int i = threadIdx.x;
    if (i < k)
    {
        generators[i].r = groupColorSum[i * 3 + 0] / groupCount[i];
        generators[i].g = groupColorSum[i * 3 + 1] / groupCount[i];
        generators[i].b = groupColorSum[i * 3 + 2] / groupCount[i];
    }
}

// Replace color kernel
__global__ void replaceColorKernel(UINT width, UINT height, int k, Pixel *colors, Pixel *generators, int *groupCount)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        Pixel pixel = colors[index];

        float minDist = INFINITY;
        int minIndex = 0;

        for (int i = 0; i < k; i++)
        {
            Pixel gen = generators[i];
            float dist = sqrtf((pixel.r - gen.r) * (pixel.r - gen.r) + (pixel.g - gen.g) * (pixel.g - gen.g) + (pixel.b - gen.b) * (pixel.b - gen.b));
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
    UINT width, height;
    uint8_t *hostDataBuf = JpegFile::JpegFileToRGB("test-large.jpg", &width, &height);

    const int N = height * width;

    Pixel *device_colors;
    cudaMalloc(&device_colors, width * height * sizeof(Pixel));
    cudaMemcpy(device_colors, hostDataBuf, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

    int k;
    std::cout << "Enter the number of clusters: ";
    std::cin >> k;

    auto generators = new Pixel[k];
    printf("Generators: \n");
    for (int i = 0; i < k; i++)
    {
        auto G = generators[i];
        G.r = hostDataBuf[(N * 3 / k) * i];
        G.g = hostDataBuf[(N * 3 / k) * i + 1];
        G.b = hostDataBuf[(N * 3 / k) * i + 2];
        printf("Generator %d: %d %d %d\n", i, G.r, G.g, G.b);
    }

    Pixel *device_generators;
    cudaMalloc(&device_generators, k * sizeof(Pixel));
    cudaMemcpy(device_generators, generators, k * sizeof(Pixel), cudaMemcpyHostToDevice);

    int *device_groupColorSum;
    int *device_groupCount;
    cudaMalloc(&device_groupColorSum, k * 3 * sizeof(int));
    cudaMalloc(&device_groupCount, k * sizeof(int));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int iter = 0; iter < 10; iter++)
    {
        groupingKernel<<<gridSize, blockSize, (k * (sizeof(Pixel) + 2 * sizeof(int)))>>>(width, height, k, device_colors, device_generators, device_groupColorSum, device_groupCount);
        cudaDeviceSynchronize();

        updateGeneratorsKernel<<<1, k>>>(k, device_generators, device_groupColorSum, device_groupCount);
        cudaDeviceSynchronize();

        // Reset group color sum and count
        cudaMemset(device_groupColorSum, 0, k * 3 * sizeof(int));
        cudaMemset(device_groupCount, 0, k * sizeof(int));
    }

    replaceColorKernel<<<gridSize, blockSize>>>(width, height, k, device_colors, device_generators, device_groupCount);
    cudaDeviceSynchronize();

    cudaMemcpy(hostDataBuf, device_colors, N * sizeof(Pixel), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Time for CUDA execution: %f ms\n", elapsedTime);

    JpegFile::RGBToJpegFile("output.jpg", hostDataBuf, width, height, 100, false);

    delete[] generators;
    free(hostDataBuf);
    cudaFree(device_colors);
    cudaFree(device_generators);
    cudaFree(device_groupColorSum);
    cudaFree(device_groupCount);

    return 0;
}