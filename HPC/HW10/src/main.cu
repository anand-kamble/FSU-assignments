#include <cuda_runtime.h> // I am not sure if this is needed when using nvcc, but vscode was complaining about not being able to find cuda_runtime.h
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


/**
 * @brief Function to calculate the euclidean distance between two pixels
 * 
 * @param p1 
 * @param p2 
 * @return double 
 */
double euclideanDistance(const Pixel *p1, const Pixel *p2)
{
    double sum = 0.0;
    return sqrt(pow(p1->r - p2->r, 2) + pow(p1->g - p2->g, 2) + pow(p1->b - p2->b, 2));
};


/**
 * @brief Function to calculate the sum of all pixels in a group
 * 
 * @param width 
 * @param height 
 * @param k 
 * @param colors 
 * @param generators 
 * @param groupColorSum 
 * @param groupCount 
 * @return __global__ 
 */
__global__ void groupingKernel(UINT width, UINT height, int k, Pixel *colors, Pixel *generators, int *groupColorSum, int *groupCount)
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

        atomicAdd(&groupColorSum[minIndex * 3 + 0], (int)pixel.r);
        atomicAdd(&groupColorSum[minIndex * 3 + 1], (int)pixel.g);
        atomicAdd(&groupColorSum[minIndex * 3 + 2], (int)pixel.b);
        atomicAdd(&groupCount[minIndex], 1);
    }
}


/**
 * @brief Function to update the generators
 * 
 * @param k 
 * @param generators 
 * @param groupColorSum 
 * @param groupCount 
 * @return __global__ 
 */
__global__ void updateGeneratorsKernel(int k, Pixel *generators, int *groupColorSum, int *groupCount)
{
    int i = threadIdx.x;
    if (i < k)
    {
        generators[i].r = (int)groupColorSum[i * 3 + 0] / groupCount[i];
        generators[i].g = (int)groupColorSum[i * 3 + 1] / groupCount[i];
        generators[i].b = (int)groupColorSum[i * 3 + 2] / groupCount[i];
    }
}

/**
 * @brief Function to replace the colors of the pixels with the color of the nearest generator
 * 
 * @param width 
 * @param height 
 * @param k 
 * @param colors 
 * @param generators 
 * @param groupCount 
 * @return __global__ 
 */
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
    UINT width, height; // Image width and height
    uint8_t *hostDataBuf = JpegFile::JpegFileToRGB("test-large.jpg", &width, &height); // Load the image

    const int N = height * width; // Total number of pixels

    Pixel *device_colors; // Device buffer for the image

    // Record the start time for benchmarking
    cudaEvent_t start, stop; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); // Start recording

    cudaMalloc(&device_colors, width * height * sizeof(Pixel)); // Allocate memory on the device
    cudaMemcpy(device_colors, hostDataBuf, width * height * sizeof(Pixel), cudaMemcpyHostToDevice); // Copy the image to the device

    int k; // Number of clusters
    std::cout << "Enter the number of clusters: ";
    std::cin >> k;

    auto generators = new Pixel[k]; // Array to store the generators
    printf("Generators: \n");
    for (int i = 0; i < k; i++)
    {
        // Selectnig the initial generators
        auto G = generators[i]; 
        G.r = hostDataBuf[(N * 3 / k) * i];
        G.g = hostDataBuf[(N * 3 / k) * i + 1];
        G.b = hostDataBuf[(N * 3 / k) * i + 2];
        printf("Generator %d: %d %d %d\n", i, G.r, G.g, G.b);
    }

    Pixel *device_generators; // Device buffer for the generators
    cudaMalloc(&device_generators, k * sizeof(Pixel)); // Allocate memory on the device
    cudaMemcpy(device_generators, generators, k * sizeof(Pixel), cudaMemcpyHostToDevice); // Copy the generators to the device

    int *device_groupColorSum; // Device buffer for the sum of colors in a group
    int *device_groupCount; // Device buffer for the count of pixels in a group
    cudaMalloc(&device_groupColorSum, k * 3 * sizeof(int)); // Allocate memory on the device
    cudaMalloc(&device_groupCount, k * sizeof(int)); // Allocate memory on the device
    cudaMemset(device_groupColorSum, 0, k * 3 * sizeof(int)); // Initialize the group color sum to 0
    cudaMemset(device_groupCount, 0, k * sizeof(int)); // Initialize the group count to 0

    dim3 blockSize(16, 16); // Block size
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); // Grid size
    for (int iter = 0; iter < 10; iter++)
    {
        // Launch the grouping kernel
        groupingKernel<<<gridSize, blockSize>>>(width, height, k, device_colors, device_generators, device_groupColorSum, device_groupCount);
        cudaDeviceSynchronize(); // Wait for the kernel to finish

        // Launch the update generators kernel
        updateGeneratorsKernel<<<1, k>>>(k, device_generators, device_groupColorSum, device_groupCount);
        cudaDeviceSynchronize(); // Wait for the kernel to finish

        // Reset group color sum and count
        cudaMemset(device_groupColorSum, 0, k * 3 * sizeof(int));
        cudaMemset(device_groupCount, 0, k * sizeof(int));
    }

    replaceColorKernel<<<gridSize, blockSize>>>(width, height, k, device_colors, device_generators, device_groupCount); // Launch the replace color kernel
    cudaDeviceSynchronize(); // Wait for the kernel to finish

    cudaMemcpy(hostDataBuf, device_colors, N * sizeof(Pixel), cudaMemcpyDeviceToHost); // Copy the image back to the host

    // Record the end time for benchmarking
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Time for CUDA execution: %f ms\n", elapsedTime);

    JpegFile::RGBToJpegFile("output.jpg", hostDataBuf, width, height, 100, false); // Save the image

    // Free memory
    delete[] generators;
    free(hostDataBuf);
    cudaFree(device_colors);
    cudaFree(device_generators);
    cudaFree(device_groupColorSum);
    cudaFree(device_groupCount);

    return 0;
}