#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>

class MNISTLoader {
public:
    static bool loadCSV(
        const std::string& csvPath,
        std::vector<std::vector<double>>& images,
        std::vector<int>& labels
    );
};

#endif
