cmake_minimum_required(VERSION 3.22)

project(dnn-mlir-installed-consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(DNNMLIR CONFIG REQUIRED)

add_executable(dnn-mlir-installed-consumer main.cpp)
target_link_libraries(dnn-mlir-installed-consumer PRIVATE
  DNNMLIR::DNNIR
  DNNMLIR::DNNTorchToDNN
)
