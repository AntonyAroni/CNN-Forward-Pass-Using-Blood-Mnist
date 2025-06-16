cmake_minimum_required(VERSION 3.31)
project(CNN_forward_pass)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Opciones de compilación
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

# Crear biblioteca estática con todas las clases
add_library(CNN_Library STATIC
        Image.cpp
        ConvolutionKernel.cpp
        ActivationFunction.cpp
        PoolingLayer.cpp
        ConvolutionLayer.cpp
        BloodMNISTLoader.cpp
        ConvolutionalNeuralNetwork.cpp
)

# Incluir directorios de headers
target_include_directories(CNN_Library PUBLIC .)

add_executable(CNN_forward_pass main.cpp)

target_link_libraries(CNN_forward_pass CNN_Library)
