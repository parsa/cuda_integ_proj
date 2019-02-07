#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void sub_kernel(double* result, double const* arr1,
    double const* arr2, size_t const arr_size)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const stride = blockDim.x * gridDim.x;

    for (size_t i = 0; i < arr_size; i += stride)
    {
        if (idx < arr_size)
        {
            result[idx] = arr1[idx] - arr2[idx];
        }
    }
}

std::vector<double> sub(std::vector<double> inarr1, std::vector<double> inarr2)
{
    assert(inarr1.size() == inarr2.size());

    size_t const arr_size = inarr1.size();

    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::printf("0 %s\n", cudaGetErrorString(error));
        std::exit(1);
    }

    double* arr1;
    double* arr2;
    double* calc_result;
    cudaMallocManaged(&arr1, arr_size * sizeof(double));
    cudaMallocManaged(&arr2, arr_size * sizeof(double));
    cudaMallocManaged(&calc_result, arr_size * sizeof(double));

    std::copy(inarr1.begin(), inarr1.end(), arr1);
    std::copy(inarr2.begin(), inarr2.end(), arr2);

    size_t const block_size = 256;
    size_t const num_blocks = (arr_size + block_size - 1) / block_size;
    sub_kernel<<<num_blocks, block_size>>>(calc_result, arr1, arr2, arr_size);
    cudaDeviceSynchronize();

    std::vector<double> result(calc_result, calc_result + arr_size);

    cudaFree(arr1);
    cudaFree(arr2);
    cudaFree(calc_result);

    return result;
}
