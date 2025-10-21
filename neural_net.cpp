#include "neural_net.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

NeuralNet::NeuralNet(const std::vector<int>& layerSizes) : layerSizes_(layerSizes) {
    initializeWeights();
}

void NeuralNet::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());

    for (size_t i = 1; i < layerSizes_.size(); i++) {
        int prevLayerSize = layerSizes_[i - 1];
        int currLayerSize = layerSizes_[i];

        double limit = std::sqrt(6.0 / (prevLayerSize + currLayerSize));
        std::uniform_real_distribution<> dis(-limit, limit);

        std::vector<std::vector<double>> layerWeights(currLayerSize, std::vector<double>(prevLayerSize));
        std::vector<double> layerBiases(currLayerSize, 0.0);

        for (int j = 0; j < currLayerSize; j++) {
            for (int k = 0; k < prevLayerSize; k++) {
                layerWeights[j][k] = dis(gen);
            }
        }

        weights_.push_back(layerWeights);
        biases_.push_back(layerBiases);
    }
}

double NeuralNet::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNet::sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

std::vector<double> NeuralNet::softmax(const std::vector<double>& input) {
    std::vector<double> result(input.size());
    double maxVal = *std::max_element(input.begin(), input.end());
    double sum = 0.0;

    for (size_t i = 0; i < input.size(); i++) {
        result[i] = std::exp(input[i] - maxVal);
        sum += result[i];
    }

    for (size_t i = 0; i < result.size(); i++)
        result[i] /= sum;
    

    return result;
}

std::vector<double> NeuralNet::oneHot(int label, int numClasses) {
    std::vector<double> result(numClasses, 0.0);
    result[label] = 1.0;
    return result;
}

std::vector<double> NeuralNet::forward(const std::vector<double>& input) {
    activations_.clear();
    zValues_.clear();
    activations_.push_back(input);
    std::vector<double> current = input;

    for (size_t layer = 0; layer < weights_.size() - 1; layer++) {
        std::vector<double> z(layerSizes_[layer + 1]);
        std::vector<double> activation(layerSizes_[layer + 1]);

        for (size_t i = 0; i < weights_[layer].size(); i++) {
            double sum = biases_[layer][i];
            for (size_t j = 0; j < current.size(); j++) {
                sum += weights_[layer][i][j] * current[j];
            }
            z[i] = sum;
            activation[i] = sigmoid(sum);
        }



        zValues_.push_back(z);
        activations_.push_back(activation);
        current = activation;
    }

    size_t lastLayer = weights_.size() - 1;
    std::vector<double> z(layerSizes_[lastLayer + 1]);

    for (size_t i = 0; i < weights_[lastLayer].size(); i++) {
        double sum = biases_[lastLayer][i];
        for (size_t j = 0; j < current.size(); j++) {
            sum += weights_[lastLayer][i][j] * current[j];
        }



        z[i] = sum;
    }

    zValues_.push_back(z);
    std::vector<double> output = softmax(z);
    activations_.push_back(output);

    return output;
}

void NeuralNet::backpropagate(const std::vector<double>& input, const std::vector<double>& target, double learningRate) {
    std::vector<double> output = forward(input);
    std::vector<std::vector<double>> deltas(weights_.size());
    deltas[deltas.size() - 1].resize(output.size());

    for (size_t i = 0; i < output.size(); i++)
        deltas[deltas.size() - 1][i] = output[i] - target[i];
    
    for (int layer = weights_.size() - 2; layer >= 0; layer--) {
        deltas[layer].resize(layerSizes_[layer + 1]);
        for (size_t i = 0; i < layerSizes_[layer + 1]; i++) {
            double error = 0.0;
            for (size_t j = 0; j < layerSizes_[layer + 2]; j++)
                error += deltas[layer + 1][j] * weights_[layer + 1][j][i];

            deltas[layer][i] = error * sigmoidDerivative(zValues_[layer][i]);
        }
    }

    for (size_t layer = 0; layer < weights_.size(); layer++) {
        for (size_t i = 0; i < weights_[layer].size(); i++) {
            for (size_t j = 0; j < weights_[layer][i].size(); j++)
                weights_[layer][i][j] -= learningRate * deltas[layer][i] * activations_[layer][j];
            
            biases_[layer][i] -= learningRate * deltas[layer][i];
        }
    }
}

// ==================== DATA AUGMENTATION FUNCTIONS ====================

std::vector<double> NeuralNet::rotateImage(const std::vector<double>& image, double angleDeg) {
    std::vector<double> rotated(784, 0.0);
    double angleRad = angleDeg * M_PI / 180.0;
    double cosTheta = std::cos(angleRad);
    double sinTheta = std::sin(angleRad);
    double centerX = 14.0;
    double centerY = 14.0;

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            // Rotate around center
            double dx = x - centerX;
            double dy = y - centerY;
            double srcX = dx * cosTheta + dy * sinTheta + centerX;
            double srcY = -dx * sinTheta + dy * cosTheta + centerY;

            // Bilinear interpolation
            if (srcX >= 0 && srcX < 27 && srcY >= 0 && srcY < 27) {
                int x0 = static_cast<int>(srcX);
                int y0 = static_cast<int>(srcY);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                double wx = srcX - x0;
                double wy = srcY - y0;

                double val = (1 - wx) * (1 - wy) * image[y0 * 28 + x0] +
                             wx * (1 - wy) * image[y0 * 28 + x1] +
                             (1 - wx) * wy * image[y1 * 28 + x0] +
                             wx * wy * image[y1 * 28 + x1];

                rotated[y * 28 + x] = val;
            }
        }
    }
    return rotated;
}

std::vector<double> NeuralNet::translateImage(const std::vector<double>& image, int dx, int dy) {
    std::vector<double> translated(784, 0.0);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int srcX = x - dx;
            int srcY = y - dy;

            if (srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28) {
                translated[y * 28 + x] = image[srcY * 28 + srcX];
            }
        }
    }
    return translated;
}

std::vector<double> NeuralNet::scaleImage(const std::vector<double>& image, double scale) {
    std::vector<double> scaled(784, 0.0);
    double centerX = 14.0;
    double centerY = 14.0;

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            double dx = x - centerX;
            double dy = y - centerY;
            double srcX = dx / scale + centerX;
            double srcY = dy / scale + centerY;

            // Bilinear interpolation
            if (srcX >= 0 && srcX < 27 && srcY >= 0 && srcY < 27) {
                int x0 = static_cast<int>(srcX);
                int y0 = static_cast<int>(srcY);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                double wx = srcX - x0;
                double wy = srcY - y0;

                double val = (1 - wx) * (1 - wy) * image[y0 * 28 + x0] +
                             wx * (1 - wy) * image[y0 * 28 + x1] +
                             (1 - wx) * wy * image[y1 * 28 + x0] +
                             wx * wy * image[y1 * 28 + x1];

                scaled[y * 28 + x] = val;
            }
        }
    }
    return scaled;
}

std::vector<double> NeuralNet::addNoise(const std::vector<double>& image, double noiseLevel, std::mt19937& gen) {
    std::vector<double> noisy = image;
    std::normal_distribution<> dis(0.0, noiseLevel);

    for (size_t i = 0; i < noisy.size(); i++) {
        noisy[i] += dis(gen);
        noisy[i] = std::max(0.0, std::min(1.0, noisy[i])); // Clamp to [0, 1]
    }
    return noisy;
}

std::vector<double> NeuralNet::elasticDeform(const std::vector<double>& image, double alpha, double sigma, std::mt19937& gen) {
    std::vector<double> deformed(784, 0.0);
    std::normal_distribution<> dis(0.0, 1.0);

    // Create random displacement fields
    std::vector<std::vector<double>> dx(28, std::vector<double>(28));
    std::vector<std::vector<double>> dy(28, std::vector<double>(28));

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            dx[y][x] = dis(gen);
            dy[y][x] = dis(gen);
        }
    }

    // Gaussian smoothing (simple 3x3 kernel approximation)
    int kernelSize = 3;
    std::vector<std::vector<double>> dxSmooth(28, std::vector<double>(28, 0.0));
    std::vector<std::vector<double>> dySmooth(28, std::vector<double>(28, 0.0));

    for (int y = 1; y < 27; y++) {
        for (int x = 1; x < 27; x++) {
            double sumX = 0.0, sumY = 0.0, weight = 0.0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    double w = std::exp(-(kx * kx + ky * ky) / (2 * sigma * sigma));
                    sumX += dx[y + ky][x + kx] * w;
                    sumY += dy[y + ky][x + kx] * w;
                    weight += w;
                }
            }
            dxSmooth[y][x] = alpha * sumX / weight;
            dySmooth[y][x] = alpha * sumY / weight;
        }
    }

    // Apply deformation
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            double srcX = x + dxSmooth[y][x];
            double srcY = y + dySmooth[y][x];

            // Bilinear interpolation
            if (srcX >= 0 && srcX < 27 && srcY >= 0 && srcY < 27) {
                int x0 = static_cast<int>(srcX);
                int y0 = static_cast<int>(srcY);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                double wx = srcX - x0;
                double wy = srcY - y0;

                double val = (1 - wx) * (1 - wy) * image[y0 * 28 + x0] +
                             wx * (1 - wy) * image[y0 * 28 + x1] +
                             (1 - wx) * wy * image[y1 * 28 + x0] +
                             wx * wy * image[y1 * 28 + x1];

                deformed[y * 28 + x] = val;
            }
        }
    }
    return deformed;
}

std::vector<double> NeuralNet::augmentImage(const std::vector<double>& image, std::mt19937& gen) {
    std::vector<double> augmented = image;

    // Random rotation (-15 to +15 degrees)
    std::uniform_real_distribution<> rotDist(-15.0, 15.0);
    double angle = rotDist(gen);
    augmented = rotateImage(augmented, angle);

    // Random translation (-3 to +3 pixels)
    std::uniform_int_distribution<> transDist(-3, 3);
    int dx = transDist(gen);
    int dy = transDist(gen);
    augmented = translateImage(augmented, dx, dy);

    // Random scaling (0.85 to 1.15)
    std::uniform_real_distribution<> scaleDist(0.85, 1.15);
    double scale = scaleDist(gen);
    augmented = scaleImage(augmented, scale);

    // Add Gaussian noise (0 to 0.05 intensity)
    std::uniform_real_distribution<> noiseDist(0.0, 0.05);
    double noiseLevel = noiseDist(gen);
    augmented = addNoise(augmented, noiseLevel, gen);

    // Apply elastic deformation (50% chance)
    std::uniform_real_distribution<> probDist(0.0, 1.0);
    if (probDist(gen) > 0.5) {
        augmented = elasticDeform(augmented, 3.0, 0.5, gen);
    }

    return augmented;
}

// ==================== END DATA AUGMENTATION ====================

void NeuralNet::train(
    const std::vector<std::vector<double>>& trainImages,
    const std::vector<int>& trainLabels,
    int epochs,
    double learningRate,
    int batchSize
) {
    int numSamples = trainImages.size();
    int numClasses = layerSizes_.back();

    std::cout << "\nStarting training WITH DATA AUGMENTATION..." << std::endl;
    std::cout << "Epochs: " << epochs << ", Learning Rate: " << learningRate << ", Batch Size: " << batchSize << std::endl;
    std::cout << "Training samples: " << numSamples << std::endl;
    std::cout << "Augmentation: Rotation, Translation, Scaling, Noise, Elastic Deformation" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::vector<int> indices(numSamples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        double totalLoss = 0.0;

        std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs << std::endl;
        std::cout << "[";

        int barWidth = 50;
        for (int batch = 0; batch < numSamples; batch++) {
            int idx = indices[batch];
            std::vector<double> target = oneHot(trainLabels[idx], numClasses);

            // Apply data augmentation to training image
            std::vector<double> augmentedImage = augmentImage(trainImages[idx], gen);

            backpropagate(augmentedImage, target, learningRate);

            //cross entropy
            std::vector<double> output = forward(trainImages[idx]);
            double loss = 0.0;
            for (int i = 0; i < numClasses; i++) {
                if (target[i] == 1.0) {
                    loss -= std::log(output[i] + 1e-10);
                }
            }
            totalLoss += loss;

            //prog bar
            int progress = (batch + 1) * barWidth / numSamples;
            if ((batch + 1) % (numSamples / barWidth + 1) == 0 || batch == numSamples - 1) {
                std::cout << "\r[";
                for (int i = 0; i < barWidth; i++) {
                    if (i < progress) std::cout << "=";
                    else if (i == progress) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << (batch + 1) << "/" << numSamples << " samples";
                std::cout.flush();
            }
        }

        double avgLoss = totalLoss / numSamples;
        std::cout << " - Loss: " << avgLoss << std::endl;
    }

    std::cout << "\nTraining complete!" << std::endl;
}

int NeuralNet::predict(const std::vector<double>& input) {
    std::vector<double> output = forward(input);
    return std::max_element(output.begin(), output.end()) - output.begin();
}

std::vector<double> NeuralNet::predictWithConfidence(const std::vector<double>& input) {
    return forward(input);
}

double NeuralNet::test(
    const std::vector<std::vector<double>>& testImages,
    const std::vector<int>& testLabels
) {
    int correct = 0;
    int total = testImages.size();
    std::cout << "\nTesting on " << total << " samples..." << std::endl;

    for (size_t i = 0; i < testImages.size(); i++) {
        int prediction = predict(testImages[i]);
        if (prediction == testLabels[i]) correct++;
    }

    double accuracy = 100.0 * correct / total;
    std::cout << "Accuracy: " << correct << "/" << total << " (" << accuracy << "%)" << std::endl;

    return accuracy;
}

bool NeuralNet::saveModel(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for saving: " << filename << std::endl;
        return false;
    }

    size_t numLayers = layerSizes_.size();
    file.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));
    file.write(reinterpret_cast<const char*>(layerSizes_.data()), numLayers * sizeof(int));

    for (const auto& layer : weights_) {
        for (const auto& neuron : layer) {
            size_t neuronSize = neuron.size();
            file.write(reinterpret_cast<const char*>(&neuronSize), sizeof(neuronSize));
            file.write(reinterpret_cast<const char*>(neuron.data()), neuronSize * sizeof(double));
        }
    }

    for (const auto& layer : biases_) {
        size_t layerSize = layer.size();
        file.write(reinterpret_cast<const char*>(&layerSize), sizeof(layerSize));
        file.write(reinterpret_cast<const char*>(layer.data()), layerSize * sizeof(double));
    }

    file.close();
    std::cout << "Model saved to: " << filename << std::endl;
    return true;
}

bool NeuralNet::loadModel(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for loading: " << filename << std::endl;
        return false;
    }

    size_t numLayers;
    file.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));
    layerSizes_.resize(numLayers);
    file.read(reinterpret_cast<char*>(layerSizes_.data()), numLayers * sizeof(int));

    weights_.clear();
    biases_.clear();

    for (size_t i = 1; i < numLayers; i++) {
        std::vector<std::vector<double>> layerWeights;
        for (int j = 0; j < layerSizes_[i]; j++) {
            size_t neuronSize;
            file.read(reinterpret_cast<char*>(&neuronSize), sizeof(neuronSize));
            std::vector<double> neuron(neuronSize);
            file.read(reinterpret_cast<char*>(neuron.data()), neuronSize * sizeof(double));
            layerWeights.push_back(neuron);
        }
        weights_.push_back(layerWeights);
    }

    for (size_t i = 1; i < numLayers; i++) {
        size_t layerSize;
        file.read(reinterpret_cast<char*>(&layerSize), sizeof(layerSize));
        std::vector<double> layer(layerSize);
        file.read(reinterpret_cast<char*>(layer.data()), layerSize * sizeof(double));
        biases_.push_back(layer);
    }

    file.close();
    std::cout << "Model loaded from: " << filename << std::endl;
    return true;
}
