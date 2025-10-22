#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <vector>
#include <random>

class NeuralNet {
public:

    NeuralNet(const std::vector<int>& layerSizes);

    std::vector<double> forward(const std::vector<double>& input);
    void train(
        const std::vector<std::vector<double>>& trainImages,
        const std::vector<int>& trainLabels,
        int epochs,
        double learningRate,
        int batchSize
    );
    double test(
        const std::vector<std::vector<double>>& testImages,
        const std::vector<int>& testLabels
    );

    int predict(const std::vector<double>& input);
    std::vector<double> predictWithConfidence(const std::vector<double>& input);
    bool saveModel(const std::string& filename) const;
    bool loadModel(const std::string& filename);

    std::vector<int> getLayerSizes() const { return layerSizes_; }

private:
    std::vector<int> layerSizes_;
    std::vector<std::vector<std::vector<double>>> weights_;
    std::vector<std::vector<double>> biases_;


    std::vector<std::vector<double>> activations_;
    std::vector<std::vector<double>> zValues_;

    double sigmoid(double x);
    double sigmoidDerivative(double x);

    std::vector<double> softmax(const std::vector<double>& input);
    std::vector<double> oneHot(int label, int numClasses);

    void initializeWeights();
    void backpropagate(const std::vector<double>& input, const std::vector<double>& target, double learningRate);

    std::vector<double> augmentImage(const std::vector<double>& image, std::mt19937& gen);
    std::vector<double> rotateImage(const std::vector<double>& image, double angleDeg);
    std::vector<double> translateImage(const std::vector<double>& image, int dx, int dy);
    std::vector<double> scaleImage(const std::vector<double>& image, double scale);
    std::vector<double> addNoise(const std::vector<double>& image, double noiseLevel, std::mt19937& gen);
    std::vector<double> elasticDeform(const std::vector<double>& image, double alpha, double sigma, std::mt19937& gen);
};

#endif
