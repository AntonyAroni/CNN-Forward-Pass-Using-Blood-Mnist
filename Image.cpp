//
// Created by antony on 6/15/25.
//

#include "Image.h"

Image::Image(int channels, int height, int width) {
    data.resize(channels, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));
    label = 0;
}