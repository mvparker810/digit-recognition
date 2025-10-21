#include "neural_net.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>

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

void NeuralNet::train(
    const std::vector<std::vector<double>>& trainImages,
    const std::vector<int>& trainLabels,
    int epochs,
    double learningRate,
    int batchSize
) {
    int numSamples = trainImages.size();
    int numClasses = layerSizes_.back();

    std::cout << "\nStarting training..." << std::endl;
    std::cout << "Epochs: " << epochs << ", Learning Rate: " << learningRate << ", Batch Size: " << batchSize << std::endl;
    std::cout << "Training samples: " << numSamples << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::vector<int> indices(numSamples);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        double totalLoss = 0.0;

        std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs << std::endl;
        std::cout << "[";

        int barWidth = 50;
        for (int batch = 0; batch < numSamples; batch++) {
            int idx = indices[batch];
            std::vector<double> target = oneHot(trainLabels[idx], numClasses);

            backpropagate(trainImages[idx], target, learningRate);

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
