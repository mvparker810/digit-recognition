# Digit Recognition Neural Network

A console-based neural network application for handwritten digit recognition using the MNIST dataset. Implemented from scratch with no external libraries.

## Features

- Interactive menu system for training, testing, and prediction
- Custom neural network architecture with configurable hidden layers
- Model serialization (save/load trained networks as .mdl files)
- Train on MNIST CSV dataset
- Test accuracy on validation set
- Test random samples from dataset with visualization
- Predict digits from BMP images
- Real-time training progress bar
- Confidence scores with visual probability bars

## Compilation

Use the provided batch file:
```
compile.bat
```

Or compile manually:
```
g++ -std=c++17 -O2 main.cpp neural_net.cpp mnist_loader.cpp image_loader.cpp -o digit_recognition.exe
```

Then run:
```
digit_recognition.exe
```

## Data Setup

Training and testing data are already provided in the `data` folder:
```
data/mnist_train.csv
data/mnist_test.csv
```

If you'd like to replace them with your own dataset, make sure they follow the CSV format below.

### CSV Format
- First column: label (0-9)
- Remaining 784 columns: pixel values (0-255)
- Optional header row (auto-detected and skipped)

### Where to Get MNIST Data
The MNIST dataset in CSV format can be downloaded from various sources online. Each row should contain one digit sample with 785 values total (1 label + 784 pixels). The training set typically contains 60,000 samples and the test set contains 10,000 samples.

## Menu Options

1. **New Network** - Create a new neural network
   - Input layer: 784 neurons (fixed, 28x28 image)
   - Hidden layers: customizable (recommended: 2 layers of 128, 64 neurons)
   - Output layer: 10 neurons (fixed, digits 0-9)
   - Recommended epochs: 10-20
   - Recommended learning rate: 0.1
   - Recommended batch size: 32

2. **Load Network** - Load a previously saved model from the `models` folder

3. **Train Network** - Train the current network on MNIST training data
   - Displays progress bar during training
   - Auto-saves model after training

4. **Test Network** - Evaluate accuracy on MNIST test set

5. **Test Random Sample** - View a random example from the test set
   - Shows ASCII visualization of the digit
   - Displays prediction vs actual label
   - Shows confidence scores for all digits
   - Option to reload and test the same image again

6. **Make Prediction** - Predict a digit from a BMP image
   - Reads from an image in the project root
   - Auto-resizes to 28x28 pixels for processing
   - Shows ASCII visualization of what the network sees
   - Displays confidence scores for all digits

7. **Save Network** - Manually save the current model to the `models` folder

8. **Exit** - Quit the application

## Creating Test Images

To test with custom images:

1. Create a BMP image, or edit the 'test_digit.bmp` already provided.
2. Draw your digit with:
   - White foreground (the digit itself)
   - Black background
3. Any size is supported (will be resized to 28x28)

## Model Files

Trained models are saved in the `models` folder with `.mdl` extension. These files contain:
- Network architecture (layer sizes)
- All weights and biases
- Binary format for fast loading

## Expected Performance

With recommended settings (2 hidden layers of 128 and 64 neurons, 10 epochs):
- Training accuracy: 95-98%
- Test accuracy: 95-97%
- Training time: varies by hardware (progress bar shows real-time status)

## Technical Details

- **Architecture**: Feedforward neural network
- **Hidden activation**: Sigmoid
- **Output activation**: Softmax
- **Loss function**: Cross Entropy
- **Optimization**: Minibatch gradient descent with backpropagation
- **Weight initialization**: Xavier initialization
- **Input normalization**: Pixels scaled to [0, 1]

## Sources

- **Neural Networks**: 3Blue1Brown - [Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- **But what is a neural netowrk?**: 3Blue1Brown - [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk)
- **MNIST Dataset**: LeCun, Y., Cortes, C., & Burges, C. (1998). [The MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/).
- **Cross-Entropy Loss**: [Cross entropy on Wikipedia](https://en.wikipedia.org/wiki/Cross_entropy) - Used for multi-class classification problems with softmax output.


