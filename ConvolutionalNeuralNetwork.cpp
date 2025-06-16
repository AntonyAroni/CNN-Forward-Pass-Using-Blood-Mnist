//
// Created by antony on 6/15/25.
//

#include "ConvolutionalNeuralNetwork.h"
#include <iostream>

void ConvolutionalNeuralNetwork::addConvolutionLayer(int in_channels, int out_channels, int kernel_size,
                                                     ActivationType activation) {
    conv_layers.emplace_back(in_channels, out_channels, kernel_size, activation);
    std::cout << "Añadida capa convolucional: " << in_channels << " -> " << out_channels
              << " canales, kernel " << kernel_size << "x" << kernel_size << std::endl;
}

void ConvolutionalNeuralNetwork::addPoolingLayer(int pool_size, PoolingType type) {
    pool_layers.emplace_back(pool_size, pool_size, type); // stride = pool_size
    std::cout << "Añadida capa de pooling: " << pool_size << "x" << pool_size;
    switch (type) {
        case PoolingType::MAX: std::cout << " (MAX)"; break;
        case PoolingType::MIN: std::cout << " (MIN)"; break;
        case PoolingType::AVERAGE: std::cout << " (AVERAGE)"; break;
    }
    std::cout << std::endl;
}

std::vector<std::vector<std::vector<float>>> ConvolutionalNeuralNetwork::forward(const std::vector<std::vector<std::vector<float>>>& input) {
    auto current_output = input;

    // Procesar capas de convolución alternadas con pooling
    for (size_t i = 0; i < conv_layers.size(); i++) {
        std::cout << "Procesando capa convolucional " << (i + 1) << "..." << std::endl;
        std::cout << "Entrada: " << current_output.size() << "x"
                  << current_output[0].size() << "x" << current_output[0][0].size() << std::endl;

        current_output = conv_layers[i].forward(current_output);

        // Verificar si la convolución fue exitosa
        if (current_output.empty() || current_output[0].empty() || current_output[0][0].empty()) {
            std::cerr << "Error en la capa convolucional " << (i + 1) << ". Terminando procesamiento." << std::endl;
            break;
        }

        std::cout << "Salida de conv" << (i + 1) << ": " << current_output.size() << "x"
                  << current_output[0].size() << "x" << current_output[0][0].size() << std::endl;

        // Aplicar pooling si existe la capa correspondiente
        if (i < pool_layers.size()) {
            std::cout << "Aplicando pooling..." << std::endl;
            current_output = pool_layers[i].forward(current_output);
            std::cout << "Salida de pooling: " << current_output.size() << "x"
                      << current_output[0].size() << "x" << current_output[0][0].size() << std::endl;
        }
    }

    return current_output;
}

void ConvolutionalNeuralNetwork::printArchitecture() const {
    std::cout << "\n=== Arquitectura de la CNN ===" << std::endl;
    std::cout << "Capas de convolución: " << conv_layers.size() << std::endl;
    std::cout << "Capas de pooling: " << pool_layers.size() << std::endl;

    for (size_t i = 0; i < conv_layers.size(); i++) {
        std::cout << "Conv" << (i + 1) << ": ";
        conv_layers[i].printKernelInfo();
    }
}