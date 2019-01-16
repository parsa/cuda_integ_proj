rd /s /q cmake-build-debug
cmake -H. -Bcmake-build-debug -G"Visual Studio 15 2017 Win64" -DBUILD_TESTING=ON
start cmake-build-debug\cuda_integ_proj.sln