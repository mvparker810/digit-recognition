#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <vector>
#include <string>

//black = 1.0
//white = 0.0
class ImageLoader {
public:
    //load image and convert to 28x28 grayscale with values 0-1
    static bool loadImage(const std::string& filename, std::vector<double>& pixels);
private:
    static bool loadBMP(const std::string& filename, std::vector<double>& pixels);
};

#endif
