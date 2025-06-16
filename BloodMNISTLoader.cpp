//
// Created by antony on 6/15/25.
//


#include "BloodMNISTLoader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

bool BloodMNISTLoader::loadFromCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line); // Ignora la primera línea (cabeceras)

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;

        // Leer la etiqueta (primer valor)
        std::getline(iss, token, ',');
        int label = std::stoi(token); // Ya debería ser 0-7

        // Leer píxeles (2352 valores)
        Image img(3, 28, 28);
        img.label = label;

        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    if (!std::getline(iss, token, ',')) {
                        std::cerr << "Error: Faltan píxeles en la línea." << std::endl;
                        return false;
                    }
                    img.data[c][i][j] = std::stof(token); // Ya está normalizado
                }
            }
        }

        images.push_back(img);
    }

    file.close();
    std::cout << "Cargadas " << images.size() << " imágenes de Blood-MNIST" << std::endl;
    return !images.empty();
}

const std::vector<Image>& BloodMNISTLoader::getImages() const {
    return images;
}

void BloodMNISTLoader::printDatasetInfo() const {
    if (images.empty()) return;

    std::cout << "\n=== Información del Dataset Blood-MNIST ===" << std::endl;
    std::cout << "Total de imágenes: " << images.size() << std::endl;
    std::cout << "Dimensiones: " << images[0].data.size() << "x"
              << images[0].data[0].size() << "x" << images[0].data[0][0].size() << std::endl;

    // Contar clases
    std::vector<int> class_count(8, 0);
    for (const auto& img : images) {
        if (img.label >= 0 && img.label < 8) {
            class_count[img.label]++;
        }
    }

    std::cout << "Distribución de clases:" << std::endl;
    std::vector<std::string> class_names = {
        "Basophil", "Eosinophil", "Erythroblast", "Immature granulocytes",
        "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"
    };

    for (int i = 0; i < 8; i++) {
        std::cout << "  Clase " << i << " (" << class_names[i] << "): " << class_count[i] << std::endl;
    }
}