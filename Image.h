//
// Created by antony on 6/15/25.
//

#ifndef IMAGE_H
#define IMAGE_H

#include <vector>

struct Image {
    std::vector<std::vector<std::vector<float>>> data; // [channels][height][width]
    int label;

    Image(int channels, int height, int width);
};

#endif // IMAGE_H