cmake_minimum_required(VERSION 3.4)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_OSX_ARCHITECTURES "x86_64")
list(APPEND CMAKE_FRAMEWORK_PATH /Library/Frameworks) #Mac, why not auto?

add_definitions(-DsrcPath="${CMAKE_CURRENT_SOURCE_DIR}/")

add_executable(NeuralNetwork
    main.cpp
    neuralnetwork.h neuralnetwork.cpp
    layer.h

    activation.h
    layerfullyconnected.h layerfullyconnected.cpp
    dropoutlayer.h dropoutlayer.cpp
    layerbatchnorm.h layerbatchnorm.cpp
)
