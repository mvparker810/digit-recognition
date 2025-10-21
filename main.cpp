#include <iostream>
#include <string>
#include <cstring>
#include <memory>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <sys/stat.h>
#ifdef _WIN32
    #include <direct.h>
    #define MKDIR(dir)      _mkdir(dir)
#else
    #include <sys/types.h>
    #define MKDIR(dir)      mkdir(dir, 0755)
#endif
#include "neural_net.h"
#include "mnist_loader.h"
#include "image_loader.h"

#define NN_MODEL_FILE_SUFFIX ".mdl"

void print_cli_header() {
    std::cout << "\n";
    std::cout << "========================================" << std::endl;
    std::cout << "  Digit Recognition Neural Network" << std::endl;
    std::cout << "========================================" << std::endl;
}

bool dir_exists(const char* path) {
    struct stat info;
    if (stat(path, &info) != 0) return false;
    return (info.st_mode & S_IFDIR) != 0;
}



void print_net_architecture(const std::vector<int>& layers) {
    std::cout << "\nNetwork Architecture:" << std::endl;
    std::cout << "*  Input layer:  " << layers[0] << " neurons" << std::endl;
    for (size_t i = 1; i < layers.size() - 1; i++)
        std::cout << "*  Hidden layer " << i << ": " << layers[i] << " neurons" << std::endl;

    std::cout << "*  Output layer: " << layers[layers.size() - 1] << " neurons" << std::endl;
}

void print_input_visualization(const std::vector<double>& input) {
    std::cout << "\n28x28 input:" << std::endl;
    std::cout << "+----------------------------+" << std::endl;
    for (int y = 0; y < 28; y++) {
        std::cout << "|";
        for (int x = 0; x < 28; x++) {
            double val = input[y * 28 + x];
            if (val > 0.2) std::cout << "#";
            else std::cout << " ";
        }
        std::cout << "|" << std::endl;
    }
    std::cout << "+----------------------------+" << std::endl;
}

void print_probability_table(const std::vector<double>& confidences, int prediction, int actualLabel = -1) {
    const char* digitNames[] = {"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
    int barWidth = 30;

    std::cout << "\nOUTPUT MAP:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    for (int i = 0; i < 10; i++) {
        double percentage = confidences[i] * 100.0;
        int barLength = static_cast<int>((confidences[i] * barWidth));

        std::cout << digitNames[i];
        int nameLen = strlen(digitNames[i]);
        for (int j = nameLen; j < 5; j++) {
            std::cout << " ";
        }
        std::cout << " : [";

        //draw prog bar
        for (int j = 0; j < barWidth; j++) {
            if (j < barLength - 1)                          std::cout << "=";
            else if (j == barLength - 1 && barLength > 0)   std::cout << ">";
            else                                            std::cout << " ";
            
        }

        //%
        std::cout << "] ";
        char buffer[10];
        snprintf(buffer, sizeof(buffer), "%6.2f", percentage);
        std::cout << buffer << "%";

        // prediction vs actual
        if (actualLabel >= 0) {
            if (i == prediction && i == actualLabel)    std::cout << " <-- PREDICTION (CORRECT)";
            else if (i == prediction)                   std::cout << " <-- PREDICTION";
            else if (i == actualLabel)                  std::cout << " <-- ACTUAL";
            
        } else {
            if (i == prediction) std::cout << " <-- PREDICTION";
        }

        std::cout << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;
}

int main() {
    std::unique_ptr<NeuralNet> network = nullptr;
    std::string currentModelName = "";

    if (!dir_exists("models")) MKDIR("models");

    while (1) {
        print_cli_header();
        if (network != nullptr) {
            std::cout << "Current model: " << currentModelName << std::endl;
            print_net_architecture(network->getLayerSizes());
        } else {
            std::cout << "No model loaded" << std::endl;
        }

        std::cout << "\n--- Main Menu ---" << std::endl;
        std::cout << "1. Create new network" << std::endl;
        std::cout << "2. Load existing network" << std::endl;
        std::cout << "3. Save current network" << std::endl;
        std::cout << "4. Train network" << std::endl;
        std::cout << "5. Test network" << std::endl;
        std::cout << "6. Test random sample" << std::endl;
        std::cout << "7. Make prediction" << std::endl;
        std::cout << "8. Exit" << std::endl;
        std::cout << "\nChoice: ";

        int opcode;
        if (!(std::cin >> opcode)) {
            // input nan
            std::cout << "\nInvalid input! Please enter a number." << std::endl;
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "\nPress Enter to continue...";
            std::cin.get();
            continue;
        }
        std::cin.ignore();

        switch (opcode) {
            case 1: { //new network

                std::cout << "\n--- Create New Network ---" << std::endl;
                std::cout << "Input layer: 784 (training data images are 28x28 = 784)" << std::endl;
                std::cout << "Output layer: 10 (digits 0-9)" << std::endl;
                std::cout << "\nRecommended hidden layers:" << std::endl;
                std::cout << "  Lightweight == 128" << std::endl;
                std::cout << "  Medium ======= 128 64" << std::endl;
                std::cout << "  Complex ====== 256 128 64" << std::endl;
                std::cout << "\nEnter hidden layer sizes (space separated): ";

                std::string line;
                std::getline(std::cin, line);
                std::istringstream iss(line);

                std::vector<int> layers;
                layers.push_back(784); //input layer 28x28

                int size;
                while (iss >> size) {
                    if (size <= 0) {
                        std::cout << "Error: Layer size must be positive." << std::endl;
                        goto end_creation;
                    }
                    layers.push_back(size);
                }

                layers.push_back(10); // output layer 0-9

                network = std::make_unique<NeuralNet>(layers);
                currentModelName = "<unsaved>";
                std::cout << "\nNetwork created successfully!" << std::endl;
                print_net_architecture(layers);
                end_creation:;
            } break;
            case 2: { //load network
                std::cout << "\nEnter model filename: ";
                std::string filename;
                std::getline(std::cin, filename);
                auto tempNetwork = std::make_unique<NeuralNet>(std::vector<int>{1, 1});
                if (tempNetwork->loadModel("models/" + filename)) {
                    network = std::move(tempNetwork);
                    currentModelName = filename;
                    std::cout << "Network loaded successfully!" << std::endl;
                }
            } break;
            case 3: { //save network
                if (network == nullptr) {
                    std::cout << "Error: No network loaded!" << std::endl;
                    continue;
                }
                std::cout << "\nEnter filename to save (without extension): ";
                std::string filename;
                std::getline(std::cin, filename);
                filename += NN_MODEL_FILE_SUFFIX;
                if (network->saveModel("models/" + filename)) currentModelName = filename;

            } break;
            case 4: { //train network
                if (network == nullptr) {
                    std::cout << "Error: No network loaded!" << std::endl;
                    continue;
                }

                std::cout << "\n--- Train Network ---" << std::endl;
                std::cout << "Loading training data..." << std::endl;
                std::vector<std::vector<double>> trainImages;
                std::vector<int> trainLabels;

                if (!MNISTLoader::loadCSV("data/mnist_train.csv", trainImages, trainLabels)) {
                    std::cout << "Failed to load training data!" << std::endl;
                    continue;
                }

                int epochs;
                double learningRate;
                std::cout << "\nEnter number of epochs (recommended: 5-10, more = better but slower): ";
                std::cin >> epochs;
                if (epochs <= 0) {
                    std::cout << "Invalid value! Using default: 10" << std::endl;
                    epochs = 10;
                }
                std::cout << "Enter learning rate (recommended: 0.01-0.3, default 0.1): ";
                std::cin >> learningRate;
                if (learningRate <= 0 || learningRate > 1) {
                    std::cout << "Invalid value! Using default: 0.1" << std::endl;
                    learningRate = 0.1;
                }

                network->train(trainImages, trainLabels, epochs, learningRate, 32);
                std::cout << "\nTraining complete!" << std::endl;
                
                //autosave
                if (currentModelName == "<unsaved>") {
                    std::cout << "\nAuto-saving model..." << std::endl;
                    std::cout << "Enter filename (or press Enter to skip): ";
                    std::string filename;
                    std::getline(std::cin, filename);
                    if (!filename.empty()) {
                        filename += NN_MODEL_FILE_SUFFIX;
                        if (network->saveModel("models/" + filename)) {
                            currentModelName = filename;
                        }
                    } else {
                        std::cout << "Skipped saving. Model remains unsaved." << std::endl;
                    }
                } else {
                    std::cout << "\nAuto-saving to: " << currentModelName << std::endl;
                    network->saveModel("models/" + currentModelName);
                }
            } break;
            case 5: { //test network
                if (network == nullptr) {
                    std::cout << "Error: No network loaded!" << std::endl;
                    continue;
                }

                std::cout << "\n--- Test Network ---" << std::endl;
                std::cout << "Loading test data..." << std::endl;

                std::vector<std::vector<double>> testImages;
                std::vector<int> testLabels;

                if (!MNISTLoader::loadCSV("data/mnist_test.csv", testImages, testLabels)) {
                    std::cout << "Failed to load test data!" << std::endl;
                    continue;
                }
                network->test(testImages, testLabels);
            } break;
            case 6: { //test random smaple
                if (network == nullptr) {
                    std::cout << "Error: No network loaded!" << std::endl;
                    continue;
                }

                std::cout << "\n--- Test Random Sample ---" << std::endl;
                std::cout << "Loading test data..." << std::endl;
                std::vector<std::vector<double>> testImages;
                std::vector<int> testLabels;

                if (!MNISTLoader::loadCSV("data/mnist_test.csv", testImages, testLabels)) {
                    std::cout << "Failed to load test data!" << std::endl;
                    continue;
                }

                static std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
                std::uniform_int_distribution<> dis(0, testImages.size() - 1);

                bool test_sample = true;
                while (test_sample) {
                    int randomIndex = dis(gen);
                    std::vector<double> input = testImages[randomIndex];
                    int actualLabel = testLabels[randomIndex];

                    std::cout << "\nRandom sample #" << randomIndex << " from test set" << std::endl;
                    std::cout << "Actual label: " << actualLabel << std::endl;

                    print_input_visualization(input);

                    std::vector<double> confidences = network->predictWithConfidence(input);
                    int prediction = std::max_element(confidences.begin(), confidences.end()) - confidences.begin();
                    double maxConfidence = confidences[prediction];

                    const char* digitNames[] = {"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
                    char confBuffer[10];
                    snprintf(confBuffer, sizeof(confBuffer), "%.2f", maxConfidence * 100.0);

                    std::cout << "\n========================================" << std::endl;
                    std::cout << "Actual:     " << digitNames[actualLabel] << " (" << actualLabel << ")" << std::endl;
                    std::cout << "Predicted:  " << digitNames[prediction] << " (" << prediction << ")" << std::endl;
                    std::cout << "Confidence: " << confBuffer << "%" << std::endl;
                    if (prediction == actualLabel) {
                        std::cout << "Result: CORRECT :)" << std::endl;
                    } else {
                        std::cout << "Result: INCORRECT :(" << std::endl;
                    }
                    std::cout << "========================================" << std::endl;

                    print_probability_table(confidences, prediction, actualLabel);
                  
                    std::cout << "\nSample another random? (y/n): ";
                    char response;
                    std::cin >> response;
                    std::cin.ignore();
                    test_sample = (response == 'y' || response == 'Y');
                }
            } break;
            case 7: { //predict from image
                if (network == nullptr) {
                    std::cout << "Error: No network loaded!" << std::endl;
                    continue;
                }

                std::cout << "\n--- Make Prediction ---" << std::endl;
                std::cout << "Enter image filename (e.g., test_digit.bmp): ";
                std::string filename;
                std::getline(std::cin, filename);

                bool testAgain = true;
                while (testAgain) {
                    std::vector<double> input;
                    if (!ImageLoader::loadImage(filename, input)) {
                        std::cout << "Failed to load image!" << std::endl;
                        break;
                    }

                    if (input.size() != 784) {
                        std::cout << "Error: Image processing failed!" << std::endl;
                        break;
                    }

                    print_input_visualization(input);
                    std::vector<double> confidences = network->predictWithConfidence(input);
                    int prediction = std::max_element(confidences.begin(), confidences.end()) - confidences.begin();
                    double maxConfidence = confidences[prediction];

                    const char* digitNames[] = {"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
                    char confBuffer[10];
                    snprintf(confBuffer, sizeof(confBuffer), "%.2f", maxConfidence * 100.0);

                    std::cout << "\n========================================" << std::endl;
                    std::cout << "This model is " << confBuffer << "% sure that" << std::endl;
                    std::cout << "'" << filename << "' is a " << digitNames[prediction] << std::endl;
                    std::cout << "========================================" << std::endl;

                    print_probability_table(confidences, prediction);

                    std::cout << "\nReload image and predict again? (y/n): ";
                    char response;
                    std::cin >> response;
                    std::cin.ignore();
                    testAgain = (response == 'y' || response == 'Y');
                }
            } break;
            case 8: { //exit
                std::cout << "\nGoodbye!" << std::endl;
                return 0;
            }
            default: {
                std::cout << "\nInvalid opcode! Please select 1-8." << std::endl;
            } break;
        }

        if (opcode != 8) {
            std::cout << "\nPress Enter to continue...";
            std::cin.ignore();
            std::cin.get();
        }
    }

    return 0;
}
