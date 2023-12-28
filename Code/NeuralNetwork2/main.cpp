#include "neuralnetwork.h"
#include "FullyConnectedLayer.h"
#include <iostream>
#include <vector>

int main() {
    NeuralNetwork nn; // Adjust the architecture as needed
    nn.addLayer(new FullyConnectedLayer(2, 3));
    nn.addLayer(new FullyConnectedLayer(3, 1));

    // AND dataset
    std::vector<std::pair<std::vector<double>, std::vector<double> > > dataset;

    // Manually add each pair to the dataset
    std::vector<double> input1 = {0, 0};
    std::vector<double> output1 = {1};
    dataset.push_back(std::make_pair(input1, output1));

    std::vector<double> input2 = {0, 1};
    std::vector<double> output2 = {0};
    dataset.push_back(std::make_pair(input2, output2));

    std::vector<double> input3 = {1, 0};
    std::vector<double> output3 = {0};
    dataset.push_back(std::make_pair(input3, output3));

    std::vector<double> input4 = {1, 1};
    std::vector<double> output4 = {1};
    dataset.push_back(std::make_pair(input4, output4));



    nn.train(dataset, 1000000, 0.1, NeuralNetwork::STOCHASTIC);

    // Test the trained network
    for (const auto& data : dataset) {
        auto output = nn.predict(data.first);
        std::cout << "Input: " << data.first[0] << ", " << data.first[1];
        std::cout << " - Predicted Output: " << output[0] << std::endl;
    }

    return 0;
}
