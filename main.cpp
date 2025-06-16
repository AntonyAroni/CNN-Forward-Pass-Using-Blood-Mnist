#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include "Image.h"
#include "ConvolutionalNeuralNetwork.h"
#include "ConvolutionLayer.h"
#include "PoolingLayer.h"
#include "BloodMNISTLoader.h"
#include "Enums.h"

int main() {
    std::cout << "=== CNN para Blood-MNIST - Capas de Convolución ===" << std::endl;

    // Inicializar cargador de datos
    BloodMNISTLoader loader;

    // Intentar cargar datos (opcional - funciona sin datos reales)
    std::cout << "\nIntentando cargar datos de Blood-MNIST..." << std::endl;
    bool data_loaded = loader.loadFromCSV("blood_dataset.csv");

    if (data_loaded) {
        loader.printDatasetInfo();
    } else {
        std::cout << "No se encontraron datos reales. Creando datos de ejemplo..." << std::endl;

        // Crear imagen de ejemplo para demostración
        Image example_img(3, 28, 28);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    example_img.data[c][i][j] = dist(gen);
                }
            }
        }
        example_img.label = 0;

        std::cout << "Imagen de ejemplo creada: 3x28x28" << std::endl;
    }

    // Crear y configurar la CNN
    ConvolutionalNeuralNetwork cnn;

    // Arquitectura mejorada: 3+ capas convolucionales con dimensiones apropiadas
    std::cout << "\nConfigurando arquitectura de la CNN..." << std::endl;

    cnn.addConvolutionLayer(3, 32, 3, ActivationType::RELU);    // Conv1: 3->32 canales, kernel 3x3 (28x28 -> 26x26)
    cnn.addPoolingLayer(2, PoolingType::MAX);                   // MaxPool 2x2 (26x26 -> 13x13)

    cnn.addConvolutionLayer(32, 64, 3, ActivationType::RELU);   // Conv2: 32->64 canales, kernel 3x3 (13x13 -> 11x11)
    cnn.addPoolingLayer(2, PoolingType::MAX);                   // MaxPool 2x2 (11x11 -> 5x5)

    cnn.addConvolutionLayer(64, 128, 3, ActivationType::RELU);  // Conv3: 64->128 canales, kernel 3x3 (5x5 -> 3x3)
    cnn.addPoolingLayer(2, PoolingType::AVERAGE);               // AvgPool 2x2 (3x3 -> 1x1)

    // Nota: La 4ta capa se omite porque con 1x1 no se puede aplicar kernel 3x3
    // En su lugar, se podría usar kernel 1x1 (pointwise convolution) si se desea
    std::cout << "Nota: Arquitectura optimizada para evitar dimensiones inválidas" << std::endl;

    cnn.printArchitecture();

    // Demostración con imagen de ejemplo
    std::cout << "\n=== Procesamiento de Forward Pass ===" << std::endl;

    // Crear imagen de entrada
    std::vector<std::vector<std::vector<float>>> input_image(3,
        std::vector<std::vector<float>>(28, std::vector<float>(28, 0.5f)));

    std::cout << "Imagen de entrada: 3x28x28" << std::endl;

    // Procesar a través de la red
    auto output = cnn.forward(input_image);

    std::cout << "\nSalida final: " << output.size() << "x"
              << output[0].size() << "x" << output[0][0].size() << std::endl;

    // Mostrar estadísticas de la salida
    float min_val = output[0][0][0], max_val = output[0][0][0], sum = 0.0f;
    int total_elements = 0;

    for (const auto& channel : output) {
        for (const auto& row : channel) {
            for (float val : row) {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum += val;
                total_elements++;
            }
        }
    }

    std::cout << "Estadísticas de la salida:" << std::endl;
    std::cout << "  Mínimo: " << min_val << std::endl;
    std::cout << "  Máximo: " << max_val << std::endl;
    std::cout << "  Promedio: " << (sum / total_elements) << std::endl;

    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout << "\n====================================" << std::endl;
    std::cout << "\n=== Versión alternativa con 4 capas usando padding ===" << std::endl;

    // Crear segunda CNN con padding para mantener dimensiones
    ConvolutionalNeuralNetwork cnn_with_padding;

    // Definir las 4 capas convolucionales con padding
    ConvolutionLayer conv1_pad(3, 32, 3, ActivationType::RELU, 1, 1);   // stride=1, padding=1
    ConvolutionLayer conv2_pad(32, 64, 3, ActivationType::RELU, 1, 1);
    ConvolutionLayer conv3_pad(64, 128, 3, ActivationType::RELU, 1, 1);
    ConvolutionLayer conv4_pad(128, 256, 1, ActivationType::RELU, 1, 0); // kernel 1x1

    // Capas de pooling
    PoolingLayer pool1(2, 2, PoolingType::MAX);  // Pooling después de conv1 y conv2
    PoolingLayer pool2(2, 2, PoolingType::AVERAGE); // Pooling después de conv3

    std::cout << "CNN con padding configurada (4 capas completas)" << std::endl;

    // Demostrar el forward pass completo
    std::cout << "\n=== Procesamiento con 4 capas ===" << std::endl;
    auto padded_input = input_image;  // Imagen de entrada 3x28x28

    // --- Capa 1 ---
    std::cout << "Conv1 (con padding): 3x28x28 -> ";
    padded_input = conv1_pad.forward(padded_input);
    std::cout << "32x" << padded_input[0].size() << "x" << padded_input[0][0].size() << std::endl;

    padded_input = pool1.forward(padded_input);
    std::cout << "Pool1: 32x" << padded_input[0].size() << "x" << padded_input[0][0].size() << std::endl;

    // --- Capa 2 ---
    std::cout << "Conv2 (con padding): -> ";
    padded_input = conv2_pad.forward(padded_input);
    std::cout << "64x" << padded_input[0].size() << "x" << padded_input[0][0].size() << std::endl;

    padded_input = pool1.forward(padded_input);
    std::cout << "Pool2: 64x" << padded_input[0].size() << "x" << padded_input[0][0].size() << std::endl;

    // --- Capa 3 ---
    std::cout << "Conv3 (con padding): -> ";
    padded_input = conv3_pad.forward(padded_input);
    std::cout << "128x" << padded_input[0].size() << "x" << padded_input[0][0].size() << std::endl;

    padded_input = pool2.forward(padded_input);
    std::cout << "Pool3: 128x" << padded_input[0].size() << "x" << padded_input[0][0].size() << std::endl;

    // --- Capa 4 (1x1) ---
    std::cout << "Conv4 (1x1): -> ";
    padded_input = conv4_pad.forward(padded_input);
    std::cout << "256x" << padded_input[0].size() << "x" << padded_input[0][0].size() << std::endl;

    // Estadísticas de salida
    float min_val2 = std::numeric_limits<float>::max();
    float max_val2 = std::numeric_limits<float>::min();
    float sum2 = 0.0f;
    int total2 = 0;

    for (const auto& channel : padded_input) {
        for (const auto& row : channel) {
            for (float val : row) {
                min_val2 = std::min(min_val2, val);
                max_val2 = std::max(max_val2, val);
                sum2 += val;
                total2++;
            }
        }
    }

    std::cout << "\nSalida final: 256x" << padded_input[0].size() << "x" << padded_input[0][0].size() << std::endl;
    std::cout << "Estadísticas de salida:" << std::endl;
    std::cout << "  Mínimo: " << min_val2 << std::endl;
    std::cout << "  Máximo: " << max_val2 << std::endl;
    std::cout << "  Promedio: " << (sum2 / total2) << std::endl;

    std::cout << "\n=== Procesamiento de 4 capas completado ===" << std::endl;
    return 0;
}