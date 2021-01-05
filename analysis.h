//
// Created by Puhan Zhang on 2/18/20.
//

#ifndef DE_C_PURE_ANALYSIS_H
#define DE_C_PURE_ANALYSIS_H

#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <random>
#include <ctime>
#include <chrono>
#include <sys/stat.h>
#include "lattice_base.h"
#include "armadillo"
#include <torch/script.h>
#include <torch/torch.h>

bool fileExists(const std::string& filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}

void generate_random_spins(std::vector<Square_Site>& test_lattice) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for (auto &site : test_lattice) {
        site.set_theta(2.0 * PI * distribution(generator));
        site.set_phi(std::acos(2.0 * distribution(generator) - 1));
        site.set_spins(std::sin(site.get_phi()) * std::cos(site.get_theta()),
                       std::sin(site.get_phi()) * std::sin(site.get_theta()), std::cos(site.get_phi()));
    }
}

void initial_from_file(std::vector<Square_Site>& test_lattice) {
    arma::mat initial(test_lattice.size(), 12);
    initial.load("../data_input/initial.csv", arma::csv_ascii);
    for (u_long i = 0; i < test_lattice.size(); ++i) {
        test_lattice[i].set_spins(initial.at(i, 0), initial.at(i, 1), initial.at(i, 2));
    }
}

void initialize(std::vector<Square_Site> &test_lattice) {
    std::ofstream initial_file("../data_output/initial.csv", std::ios::out);
    // randomize spins
    generate_random_spins(test_lattice);
    //initial_from_file(test_lattice);
    for (u_long i = 0; i < test_lattice.size(); ++i) {
        initial_file << test_lattice[i].get_spin_x() << ",";
        initial_file << test_lattice[i].get_spin_y() << ",";
        initial_file << test_lattice[i].get_spin_z() << "\n";
    }
    initial_file.close();
}

void calculate_mag(std::vector<Square_Site> &test_lattice, int round = 0) {
    std::ofstream mag_file;
    if (fileExists("../data_output/mag.csv")) {
        mag_file.open("../data_output/mag.csv", std::ios_base::app);
    }
    else {
        mag_file.open("../data_output/mag.csv", std::ios_base::out);
    }
    std::vector<double> mag(3, 0.0);
    for (auto &site : test_lattice) {
        mag[0] += site.get_spin_x();
        mag[1] += site.get_spin_y();
        mag[2] += site.get_spin_z();
    }

    mag[0] /= (test_lattice.size());
    mag[1] /= (test_lattice.size());
    mag[2] /= (test_lattice.size());
    mag_file << round << "," << std::sqrt(std::pow(mag[0], 2.0) + std::pow(mag[1], 2.0) + std::pow(mag[2], 2.0)) << std::endl;
    mag_file.close();
}

void copy_to_ml(std::vector<Square_Site> &test_lattice, at::Tensor &x) {
    for (u_long i = 0; i < test_lattice.size(); ++i) {
        x[i][0] = test_lattice[i].get_spin_x();
        x[i][1] = test_lattice[i].get_spin_y();
        x[i][2] = test_lattice[i].get_spin_z();
    }
}

void copy_from_ml(std::vector<Square_Site> &test_lattice, at::Tensor &force) {
    for (u_long i = 0; i < test_lattice.size(); ++i) {
        test_lattice[i].set_force(force[i][0].item().to<double>(), force[i][1].item().to<double>(), force[i][2].item().to<double>());
    }
}

void generate_t_vector(std::vector<Square_Site> &test_lattice, double alpha = 0.1) {
    for (auto &site : test_lattice) {
        arma::vec B_vector(3);
        arma::vec X_vector(3);
        B_vector[0] = site.get_force_x();
        B_vector[1] = site.get_force_y();
        B_vector[2] = site.get_force_z();
        X_vector[0] = site.get_spin_x();
        X_vector[1] = site.get_spin_y();
        X_vector[2] = site.get_spin_z();
        arma::vec T_vector = B_vector + alpha * arma::cross(X_vector, B_vector);
        site.set_t_vector(T_vector[0], T_vector[1], T_vector[2]);
    }
}

void generate_c_vector(std::vector<Square_Site> &test_lattice, double h = 0.01) {
    for (auto &site : test_lattice) {
        arma::mat k_matrix = arma::ones(3, 3);
        arma::vec X_vector(3);
        X_vector[0] = site.get_spin_x(); X_vector[1] = site.get_spin_y(); X_vector[2] = site.get_spin_z();
        k_matrix(0, 1) = 0.5 * h * site.get_t_z();
        k_matrix(0, 2) = -0.5 * h * site.get_t_y();
        k_matrix(1, 2) = 0.5 * h * site.get_t_x();
        k_matrix(1, 0) = -k_matrix(0, 1);
        k_matrix(2, 0) = -k_matrix(0, 2);
        k_matrix(2, 1) = -k_matrix(1, 2);
        arma::vec c_vector = k_matrix.i() * k_matrix.t() * X_vector;
        site.set_c_vector(c_vector[0], c_vector[1], c_vector[2]);
    }
}

void generate_half_spin_vector(std::vector<Square_Site> &test_lattice) {
    for (auto &site : test_lattice) {
        arma::vec X_vector(3);
        X_vector[0] = site.get_spin_x(); X_vector[1] = site.get_spin_y(); X_vector[2] = site.get_spin_z();
        arma::vec half_X_vector(3);
        half_X_vector[0] = site.get_c_x(); half_X_vector[1] = site.get_c_y(); half_X_vector[2] = site.get_c_z();
        arma::vec half_vector(3);
        half_vector = (X_vector + half_X_vector) * 0.5;
        site.set_i_spin_vector(half_vector[0], half_vector[1], half_vector[2]);
    }
}

//void intermediate_output(std::vector<Square_Site> &test_lattice, arma::mat &intermediate_mat) {
//    std::ofstream share_file("share_file.csv", std::ios::out);
//    for (u_long i = 0; i < test_lattice.size(); ++i) {
//        intermediate_mat.at(i, 6) = test_lattice[i].get_i_spin_x();
//        intermediate_mat.at(i, 7) = test_lattice[i].get_i_spin_y();
//        intermediate_mat.at(i, 8) = test_lattice[i].get_i_spin_z();
//    }
//    intermediate_mat.save(share_file, arma::csv_ascii);
//    share_file.close();
//}

void intermediate_passing(std::vector<Square_Site> &test_lattice, at::Tensor &force, double h = 0.01, double alpha = 0.1) {
    copy_from_ml(test_lattice, force);
    generate_t_vector(test_lattice, alpha);
    generate_c_vector(test_lattice, h);
    generate_half_spin_vector(test_lattice);
    //intermediate_output(test_lattice, first_back);
}

void copy_to_ml_2(std::vector<Square_Site> &test_lattice, at::Tensor &x) {
    for (u_long i = 0; i < test_lattice.size(); ++i) {
        x[i][0] = test_lattice[i].get_i_spin_x();
        x[i][1] = test_lattice[i].get_i_spin_y();
        x[i][2] = test_lattice[i].get_i_spin_z();
    }
}

void copy_from_ml_2(std::vector<Square_Site> &test_lattice, at::Tensor &force) {
//    arma::mat second_return_set_up(test_lattice.size(), 12);
//    second_return_set_up.load("share_file.csv", arma::csv_ascii);
    for (u_long i = 0; i < test_lattice.size(); ++i) {
//        test_lattice[i].set_spins(second_return_set_up.at(i, 0), second_return_set_up.at(i, 1), second_return_set_up.at(i, 2));
//        test_lattice[i].set_force(second_return_set_up.at(i, 3), second_return_set_up.at(i, 4), second_return_set_up.at(i, 5));
//        test_lattice[i].set_i_spin_vector(second_return_set_up.at(i, 6), second_return_set_up.at(i, 7), second_return_set_up.at(i, 8));
        test_lattice[i].set_i_force(force[i][0].item().to<double>(), force[i][1].item().to<double>(), force[i][2].item().to<double>());
    }
    // return second_return_set_up;
}

void generate_i_t_vector(std::vector<Square_Site> &test_lattice, double alpha = 0.1) {
    for (auto &site: test_lattice) {
        arma::vec B_vector(3);
        arma::vec X_vector(3);
        B_vector[0] = site.get_i_force_x(); B_vector[1] = site.get_i_force_y(); B_vector[2] = site.get_i_force_z();
        X_vector[0] = site.get_i_spin_x(); X_vector[1] = site.get_i_spin_y(); X_vector[2] = site.get_i_spin_z();
        arma::vec T_vector = B_vector + alpha * arma::cross(X_vector, B_vector);
        site.set_i_t_vector(T_vector[0], T_vector[1], T_vector[2]);
    }
}

void generate_new_spin(std::vector<Square_Site> &test_lattice, double h = 0.01) {
    for (auto &site: test_lattice) {
        arma::mat k_matrix = arma::ones(3, 3);
        arma::vec X_vector(3);
        X_vector[0] = site.get_spin_x(); X_vector[1] = site.get_spin_y(); X_vector[2] = site.get_spin_z();

        k_matrix(0, 1) = 0.5 * h * site.get_i_t_z();
        k_matrix(0, 2) = -0.5 * h * site.get_i_t_y();
        k_matrix(1, 2) = 0.5 * h * site.get_i_t_x();
        k_matrix(1, 0) = -k_matrix(0, 1);
        k_matrix(2, 0) = -k_matrix(0, 2);
        k_matrix(2, 1) = -k_matrix(1, 2);
        arma::vec c_vector = k_matrix.i() * k_matrix.t() * X_vector;
        site.set_spins(c_vector[0], c_vector[1], c_vector[2]);
    }
}

//void re_initialize(std::vector<Square_Site> &test_lattice) {
//    std::ofstream share_file("share_file.csv", std::ios::out);
//    arma::mat initial_set_up(test_lattice.size(), 12);
//    initial_set_up.zeros();
//    for (u_long i = 0; i < test_lattice.size(); ++i) {
//        initial_set_up.at(i, 0) = test_lattice[i].get_spin_x();
//        initial_set_up.at(i, 1) = test_lattice[i].get_spin_y();
//        initial_set_up.at(i, 2) = test_lattice[i].get_spin_z();
//    }
//    initial_set_up.save(share_file, arma::csv_ascii);
//    share_file.close();
//}

void final_passing(std::vector<Square_Site> &test_lattice, at::Tensor &force, double h = 0.01, double alpha = 0.1, int round = 0) {
    copy_from_ml_2(test_lattice, force);
    
    generate_i_t_vector(test_lattice, alpha);
    

    if (round % 10 == 0) {
        std::ofstream screen_shot("../data_output/snapshot_save/screenshot_" + std::to_string(round) + ".csv", std::ios::out);
        calculate_mag(test_lattice, round);
        // randomize spins
        for (u_long i = 0; i < test_lattice.size(); ++i) {
            screen_shot << test_lattice[i].get_spin_x() << ",";
            screen_shot << test_lattice[i].get_spin_y() << ",";
            screen_shot << test_lattice[i].get_spin_z() << "\n";
        }
        screen_shot.close();
    }
    generate_new_spin(test_lattice, h);
    // re-initializing start
    // re_initialize(test_lattice);

}


#endif //DE_C_PURE_ANALYSIS_H
