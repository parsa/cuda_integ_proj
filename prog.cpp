#include <algorithm>
#include <iostream>
#include <vector>

std::vector<double> add(std::vector<double> inarr1, std::vector<double> inarr2);

void test_integration()
{
    constexpr size_t arr_size = 2 << 24;

    std::cout << "Initializing test arrays...\n";
    std::vector<double> arr1(arr_size);
    std::vector<double> arr2(arr_size);

    for (size_t i = 0; i < arr_size; i++)
    {
        arr1[i] = static_cast<double>(i);
        arr2[i] = static_cast<double>(arr_size - i);
    }

    std::cout << "Calling the kernel wrapper...\n";
    auto result = add(std::move(arr1), std::move(arr2));

    std::cout << "Verifying results...\n";
    if (std::all_of(result.begin(), result.end(),
            [arr_size](double x) { return x == arr_size; }))
    {
        std::cout << "All results were valid.\n";
    }
    else
    {
        std::cout << "At least one result is invalid.\n";
    }
}

int main()
{
    std::cout << "Test CUDA integration\n";
    test_integration();
    std::cout << "Finished testing\n";
    return 0;
}
