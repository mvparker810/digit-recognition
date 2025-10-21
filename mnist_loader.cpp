#include "mnist_loader.h"
#include <fstream>
#include <iostream>
#include <sstream>

bool MNISTLoader::loadCSV(
    const std::string& csvPath,
    std::vector<std::vector<double>>& images,
    std::vector<int>& labels
) {
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open CSV file: " << csvPath << std::endl;
        return false;
    }

    images.clear();
    labels.clear();

    std::string line;
    bool firstLine = true;
    int lineCount = 0;

    while (std::getline(file, line)) {
        //skip header
        if (firstLine && line.find("label") != std::string::npos) {
            firstLine = false;
            continue;
        }
        firstLine = false;

        std::stringstream ss(line);
        std::string value;

        if (!std::getline(ss, value, ',')) {
            continue;
        }

        int label = std::stoi(value);
        labels.push_back(label);

        std::vector<double> image;
        while (std::getline(ss, value, ',')) {
            double pixel = std::stod(value) / 255.0;
            image.push_back(pixel);
        }

        if (image.size() != 784) {
            std::cerr << "Warning: Line " << lineCount << " has " << image.size()
                      << " pixels (expected 784)" << std::endl;
        }

        images.push_back(image);
        lineCount++;

        if (lineCount % 10000 == 0)
            std::cout << "Loaded " << lineCount << " samples..." << std::endl;
    }

    file.close();
    std::cout << "Loaded " << lineCount << " samples total" << std::endl;

    return lineCount > 0;
}
