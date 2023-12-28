cmake_minimum_required(VERSION 3.4)
project(NeuralNetwork)

set(CMAKE_OSX_ARCHITECTURES "x86_64")
list(APPEND CMAKE_FRAMEWORK_PATH /Library/Frameworks) #Mac, why not auto?

find_package(Imagine REQUIRED COMPONENTS Graphics)

add_executable(NeuralNetwork
    main.cpp
    neuralnetwork.h neuralnetwork.cpp
    Layer.h
    Layer.cpp
    activation.h

)
target_link_libraries(NeuralNetwork Imagine::Graphics)
