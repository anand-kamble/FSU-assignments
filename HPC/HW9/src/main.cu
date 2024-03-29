/**
 * @file ColorToGrayCUDA.cu
 * @brief Convert a color image to a gray image using CUDA parallelization.
 *
 * This program reads a color image in RGB format, converts it to gray luminance,
 * and saves the result as a new jpg file. The conversion process is parallelized
 * using CUDA to achieve better performance.
 *
 * @name Student Name: Anand Kamble
 * @date Date: 28th March 2024
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../includes/Jpegfile.h"

// CUDA kernel function to convert a single pixel to grayscale
__global__ void
convertToGrayKernel(uint8_t *inputData, uint8_t *outputData, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int pixelIndex = row * width * 3 + col * 3;

        uint8_t red = inputData[pixelIndex];
        uint8_t green = inputData[pixelIndex + 1];
        uint8_t blue = inputData[pixelIndex + 2];

        // Convert to grayscale using luminance formula
        uint8_t gray = static_cast<uint8_t>(0.299 * red + 0.587 * green + 0.114 * blue);

        outputData[pixelIndex] = gray;
        outputData[pixelIndex + 1] = gray;
        outputData[pixelIndex + 2] = gray;
    }
}

int main()
{
    // Variables to store image properties and pixel data
    UINT height, width;
    uint8_t *hostDataBuf;

    // Read the input color image in RGB format
    hostDataBuf = JpegFile::JpegFileToRGB("test-large.jpg", &width, &height);

    // Allocate device memory for input and output data
    uint8_t *deviceInputData, *deviceOutputData;
    size_t dataSize = width * height * 3 * sizeof(uint8_t);

    // Record the start time for benchmarking
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMalloc(&deviceInputData, dataSize);
    cudaMalloc(&deviceOutputData, dataSize);

    // Copy input data from host to device
    cudaMemcpy(deviceInputData, hostDataBuf, dataSize, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(1024, 1024);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel
    convertToGrayKernel<<<gridDim, blockDim>>>(deviceInputData, deviceOutputData, width, height);

    // Record the end time for benchmarking
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy output data from device to host
    cudaMemcpy(hostDataBuf, deviceOutputData, dataSize, cudaMemcpyDeviceToHost);

    // Print the execution time
    printf("Time for CUDA execution: %f ms\n", elapsedTime);

    // Write the grayscale image to a new JPG file
    JpegFile::RGBToJpegFile("testmono_cuda.jpg", hostDataBuf, width, height, true, 75);

    // Free device memory
    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);

    // Free host memory
    delete[] hostDataBuf;

    return 0;
}