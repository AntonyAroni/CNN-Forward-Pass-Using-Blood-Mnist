//
// Created by antony on 6/15/25.
//

#ifndef BLOOD_MNIST_LOADER_H
#define BLOOD_MNIST_LOADER_H

#include "Image.h"
#include <vector>
#include <string>

class BloodMNISTLoader {
private:
    std::vector<Image> images;

public:
    bool loadFromCSV(const std::string& filename);
    const std::vector<Image>& getImages() const;
    void printDatasetInfo() const;
};

#endif // BLOOD_MNIST_LOADER_H