#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device Number: " << i << std::endl;
        std::cout << "  Device name: " << prop.name << std::endl;
        std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
        std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << "  Peak Memory Bandwidth (GB/s): "
                  << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
        std::cout << "  Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Thread Dimensions (x, y, z): ("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max Grid Dimensions (x, y, z): ("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")" << std::endl;
    }
    return 0;
}
