#include "neuralnetwork.h"
#include "LayerFullyConnected.h"
#include "DropoutLayer.h"
#include "LayerBatchNorm.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <Imagine/Graphics.h>
using namespace Imagine;

std::vector<std::pair<std::vector<double>, std::vector<double>>> readCSV(const std::string& filename) {
    std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset;
    std::ifstream file(filename);
    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<double> row;

        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                try {
                    row.push_back(std::stod(token));
                } catch (const std::invalid_argument& e) {
                    // Handle the case where the conversion fails
                    std::cerr << "Invalid argument: " << e.what() << '\n';
                    row.push_back(0.0); // or handle it appropriately
                }
            }
            else {
                row.push_back(0.0); // Default value for empty strings
            }
        }

        // Create a pair with all but the last element and the last element separately
        std::vector<double> features(row.begin(), row.end() - 1);
        std::vector<double> label(1, row.back());

        dataset.push_back(std::make_pair(features, label));
    }

    return dataset;
}

int main() {
    NeuralNetwork nn;

    // Neural Network architecture
    nn.addLayer(new DropoutLayer(0.01)); // To avoid overfitting
    nn.addLayer(new LayerBatchNorm(13));
    nn.addLayer(new LayerFullyConnected(13, 8, new TanH()));
    nn.addLayer(new LayerFullyConnected(8, 1, new Sigmoid()));

    // Training dataset
    const std::string filename = srcPath("./train.csv");
    auto dataset = readCSV(filename);

    // Training phase
    nn.train(dataset, 100, TrainingMode::STOCHASTIC, 0.05);




    // ----- RESULTS SECTION -----

    // Test dataset
    const std::string filenameTest = srcPath("./test.csv");
    auto dataset2 = readCSV(filenameTest);

    double totalLoss = 0.0;

    for (auto& v : dataset2) {
        const std::vector<double>& input = v.first;
        const std::vector<double>& target = v.second;

        // Calculate loss (e.g., Mean Squared Error)
        for (size_t j = 0; j < target.size(); ++j) {
            double error = nn.predict(input)[j] - target[j];
            totalLoss += error * error;
        }
    }

    // Calculate average loss over the dataset
    totalLoss /= dataset2.size();

    std::cout << std::endl;
    std::cout << "----- RESULTS  -----" << std::endl;
    std::cout << "Size training dataset: " << dataset.size() << std::endl;
    std::cout << "Size test dataset: " << dataset2.size() << std::endl;
    std::cout << "Loss in test dataset: " << totalLoss << std::endl;

    // ----------------------------------

    return 0;
}
