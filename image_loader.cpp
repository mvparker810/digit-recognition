#include "image_loader.h"
#include <fstream>
#include <iostream>
#include <cmath>

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType;
    uint32_t fileSize;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offsetData;
};

struct BMPInfoHeader {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitCount;
    uint32_t compression;
    uint32_t sizeImage;
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    uint32_t colorsUsed;
    uint32_t colorsImportant;
};
#pragma pack(pop)

bool ImageLoader::loadBMP(const std::string& filename, std::vector<double>& pixels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open image file: " << filename << std::endl;
        return false;
    }

    BMPHeader header;
    BMPInfoHeader infoHeader;

    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

    if (header.fileType != 0x4D42) {
        std::cerr << "Error: Not a valid BMP file!" << std::endl;
        return false;
    }

    int width = infoHeader.width;
    int height = std::abs(infoHeader.height);
    int bitCount = infoHeader.bitCount;

    std::cout << "Loaded image: " << width << "x" << height << " (" << bitCount << " bits)" << std::endl;
    if (bitCount != 24 && bitCount != 8 && bitCount != 32 && bitCount != 1 && bitCount != 4) {
        std::cerr << "Error: Unsupported bit depth (" << bitCount << " bits)!" << std::endl;
        std::cerr << "Supported formats: 1-bit, 4-bit, 8-bit, 24-bit, 32-bit" << std::endl;
        std::cerr << "To fix: Open in Paint, Save As -> BMP -> 24-bit Bitmap" << std::endl;
        return false;
    }

    std::vector<unsigned char> palette;
    if (bitCount <= 8) {
        int numColors = infoHeader.colorsUsed;
        if (numColors == 0) {
            numColors = 1 << bitCount;
        }
        palette.resize(numColors * 4);
        file.read(reinterpret_cast<char*>(palette.data()), numColors * 4);
    }

    file.seekg(header.offsetData, std::ios::beg);
    std::vector<std::vector<double>> imageData(height, std::vector<double>(width));

    for (int y = height - 1; y >= 0; y--) {
        int rowBytes;
        if      (bitCount == 24)    rowBytes = width * 3;
        else if (bitCount == 32)    rowBytes = width * 4;
        else if (bitCount == 8)     rowBytes = width;
        else if (bitCount == 4)     rowBytes = (width + 1) / 2;
        else if (bitCount == 1)     rowBytes = (width + 7) / 8;
        
        int rowPadding = (4 - (rowBytes % 4)) % 4;

        for (int x = 0; x < width; x++) {
            if (bitCount == 24) {
                unsigned char bgr[3];
                file.read(reinterpret_cast<char*>(bgr), 3);
                double gray = (bgr[2] + bgr[1] + bgr[0]) / 3.0;

                double normalized = gray / 255.0;
                imageData[y][x] = (normalized > 0.5) ? 1.0 : 0.0;
            } else if (bitCount == 32) {
                unsigned char bgra[4];
                file.read(reinterpret_cast<char*>(bgra), 4);
                double gray = (bgra[2] + bgra[1] + bgra[0]) / 3.0;

                double normalized = gray / 255.0;
                imageData[y][x] = (normalized > 0.5) ? 1.0 : 0.0;
            } else if (bitCount == 8) {
                unsigned char paletteIndex;
                file.read(reinterpret_cast<char*>(&paletteIndex), 1);

                int idx = paletteIndex * 4;
                double gray = (palette[idx + 2] + palette[idx + 1] + palette[idx]) / 3.0;
                double normalized = gray / 255.0;
                imageData[y][x] = (normalized > 0.5) ? 1.0 : 0.0;
            } else if (bitCount == 4) {
                if (x % 2 == 0) {
                    unsigned char byte;
                    file.read(reinterpret_cast<char*>(&byte), 1);

                    int paletteIndex1 = (byte >> 4) & 0x0F;
                    int idx1 = paletteIndex1 * 4;
                    double gray1 = (palette[idx1 + 2] + palette[idx1 + 1] + palette[idx1]) / 3.0;
                    double normalized1 = gray1 / 255.0;
                    imageData[y][x] = (normalized1 > 0.5) ? 1.0 : 0.0;

                    if (x + 1 < width) {
                        int paletteIndex2 = byte & 0x0F;
                        int idx2 = paletteIndex2 * 4;
                        double gray2 = (palette[idx2 + 2] + palette[idx2 + 1] + palette[idx2]) / 3.0;
                        double normalized2 = gray2 / 255.0;
                        imageData[y][x + 1] = (normalized2 > 0.5) ? 1.0 : 0.0;
                    }
                }
            } else if (bitCount == 1) {
                if (x % 8 == 0) {
                    unsigned char byte;
                    file.read(reinterpret_cast<char*>(&byte), 1);
                    for (int bit = 0; bit < 8 && (x + bit) < width; bit++) {
                        int pixelBit = (byte >> (7 - bit)) & 1;
                        int idx = pixelBit * 4;
                        double gray = (palette[idx + 2] + palette[idx + 1] + palette[idx]) / 3.0;
                        double normalized = gray / 255.0;
                        imageData[y][x + bit] = (normalized > 0.5) ? 1.0 : 0.0;
                    }
                }
            }
        }
        file.seekg(rowPadding, std::ios::cur);
    }
    file.close();

    
    pixels.resize(784);
    double scaleX = static_cast<double>(width) / 28.0;
    double scaleY = static_cast<double>(height) / 28.0;
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int srcX = static_cast<int>(x * scaleX);
            int srcY = static_cast<int>(y * scaleY);

            srcX = std::min(srcX, width - 1);
            srcY = std::min(srcY, height - 1);

            pixels[y * 28 + x] = imageData[srcY][srcX];
        }
    }

    std::cout << "Resized to 28x28 for neural network input" << std::endl;
    return true;
}




bool ImageLoader::loadImage(const std::string& filename, std::vector<double>& pixels) {
    size_t dotPos = filename.find_last_of('.');
    if (dotPos == std::string::npos) {
        std::cerr << "Error: No file extension found!" << std::endl;
        return false;
    }

    std::string ext = filename.substr(dotPos);


    for (char& c : ext)  c = tolower(c);

    if (ext == ".bmp") {
        return loadBMP(filename, pixels);
    } else {
        std::cerr << "Error: Unsupported format! Please use .bmp files." << std::endl;
        std::cerr << "Tip: You can convert PNG to BMP using paint or online converters." << std::endl;
        return false;
    }
}
