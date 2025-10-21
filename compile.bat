@echo off
echo Compiling Digit Recognition Neural Network...

g++ -std=c++17 -O2 main.cpp neural_net.cpp mnist_loader.cpp image_loader.cpp -o digit_recognition.exe

if %errorlevel% equ 0 (
    echo.
    echo Compilation successful! Run digit_recognition.exe to start.
) else (
    echo.
    echo Compilation failed!
)

pause
