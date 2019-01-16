#include <algorithm>
#include <cstdio>
#include <cstdlib>

__global__ void add(double* result, double const* arr1, double const* arr2,
    size_t const arr_size)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const stride = blockDim.x * gridDim.x;

    for (size_t i = 0; i < arr_size; i += stride)
    {
        if (idx < arr_size)
        {
            result[idx] = arr1[idx] + arr2[idx];
        }
    }
}

bool verify_result(double const* result, size_t const arr_size)
{
    return std::all_of(result, result + arr_size,
        [arr_size](double x) { return x == arr_size; });
    //for (size_t i = 0; i < arr_size; i++)
    //{
    //    if (result[i] != arr_size)
    //    {
    //        std::printf("result[i] = %f is an invalid result.\n", result[i]);
    //        return false;
    //    }
    //}
    //std::printf("all results were valid.\n");

    //return true;
}

int main()
{
    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::printf("0 %s\n", cudaGetErrorString(error));
        std::exit(1);
    }

    constexpr size_t arr_size = 2 << 24;

    double* arr1;
    double* arr2;
    double* result;
    cudaMallocManaged(&arr1, arr_size * sizeof(double));
    cudaMallocManaged(&arr2, arr_size * sizeof(double));
    cudaMallocManaged(&result, arr_size * sizeof(double));

    for (size_t i = 0; i < arr_size; i++)
    {
        arr1[i] = static_cast<double>(i);
        arr2[i] = static_cast<double>(arr_size - i);
    }

    size_t const block_size = 256;
    size_t const num_blocks = (arr_size + block_size - 1) / block_size;
    add<<<num_blocks, block_size>>>(result, arr1, arr2, arr_size);
    cudaDeviceSynchronize();

    if (verify_result(result, arr_size))
    {
        std::printf("all results were valid.\n");
    }
    else
    {
        std::printf("at least one result is invalid.\n");
        std::exit(1);
    }

    cudaFree(arr1);
    cudaFree(arr2);
    cudaFree(result);
}
