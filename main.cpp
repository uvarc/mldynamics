#include <iostream>
#include "lattice_base.h"
#include "analysis.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <memory>
#include <chrono>

int main(int argc, char* argv[]) {

    int lattice_length = argc > 1 ? atoi(argv[1]) : 30;
    double h = argc > 2 ? atof(argv[2]) : 0.01;
    double alpha = argc > 3 ? atof(argv[3]) : 1.0;
    int total_round = argc > 4 ? atoi(argv[4]) : 4;
    
    
    // load ML model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[5]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    
    // define lattice
    std::vector<Square_Site> test_lattice;
    test_lattice.reserve(lattice_length * lattice_length);
    for (int i = 0; i < lattice_length * lattice_length; ++i) {
        test_lattice.emplace_back(i, lattice_length);
    }
    
    
    // initialize
    initialize(test_lattice);
    std::cout << "Initialization finished.\n";
    
    
    // simulation
    std::cout << "Simulation begins:\n";
    
    // torch options
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    
    
    for (int round = 0; round < total_round; ++round) {
        // intermediate_passing
        auto start = std::chrono::high_resolution_clock::now();
        auto x = torch::zeros({lattice_length * lattice_length, 3}, options);
        copy_to_ml(test_lattice, x);
        x.requires_grad_(true);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x);
        at::Tensor output = module.forward(inputs).toTensor();
        output.backward();
        auto force = -x.grad();
        
        intermediate_passing(test_lattice, force, h, alpha);
        std::cout << "Round: " << round << " " << "intermediate passing done." << "\n";
        
        // final passing
        auto y = torch::zeros({lattice_length * lattice_length, 3}, options);
        copy_to_ml_2(test_lattice, y);
        y.requires_grad_(true);
        std::vector<torch::jit::IValue> inputs_2;
        inputs_2.push_back(y);
        at::Tensor output_2 = module.forward(inputs_2).toTensor();
        output_2.backward();
        auto force_2 = -y.grad();
        
        final_passing(test_lattice, force_2, h, alpha, round);
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Round: " << round << " " << "final passing done." << "\n";
        std::cout << "Round: " << round << " " << "done." << "\n";
        auto duration = duration_cast<std::chrono::seconds>(stop - start);
        std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
        std::cout << "**********************************" << "\n";
    }
    

    return 0;
}
