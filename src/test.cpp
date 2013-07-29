/** Test file for NN. */

#include <cmath>
#include <vector>
#include "ml/neural_net.h"
#include <iostream>
#include <string>

struct F { double f[3]; };

int main(int argc, char* argv[]) {
  NeuralNetwork* nn = new NeuralNetwork(3, 2, 0.1);
  Gradient* ng = nn->get_gradient_object();
  std::vector<F> items;
  std::vector<double> gold;
  F f;
  f.f[0] = 2; f.f[1] = 2; f.f[2] = 2;
  items.push_back(f);
  gold.push_back(1);
  f.f[0] = 2.5; f.f[1] = 1; f.f[2] = 0;
  items.push_back(f);
  gold.push_back(0);
  f.f[0] = 1.8; f.f[1] = 1.8; f.f[2] = 2.5;
  items.push_back(f);
  gold.push_back(1);
  f.f[0] = 1.5; f.f[1] = 1.5; f.f[2] = 1.1;
  items.push_back(f);
  gold.push_back(0);
  f.f[0] = 0.9; f.f[1] = 2; f.f[2] = 2;
  items.push_back(f);
  gold.push_back(0);
  f.f[0] = 2.3; f.f[1] = 1.7; f.f[2] = 1.6;
  items.push_back(f);
  gold.push_back(1);
  f.f[0] = 4; f.f[1] = -3; f.f[2] = 0;
  items.push_back(f);
  gold.push_back(0);
  f.f[0] = 3.1; f.f[1] = 3.1; f.f[2] = 3.1;
  items.push_back(f);
  gold.push_back(0);

  for (size_t iter = 0; iter < 100; iter++) {
    std::cout << "ITERATION " << iter << std::endl;
    std::cout << *nn << std::endl;
    double mse = 0;
    for (size_t item = 0; item < items.size(); item++) {
      double score = nn->score(items[item].f);
      double error = 2 * (score - gold[item]);
      mse += pow(score - gold[item], 2);
      std::cout << "Item: " << item << ", score: " << score << ", error: "
                << error << std::endl;
      ng->update(items[item].f, score, error);
    }
    ng->update_parent();
    ng->reset();
    std::cout << "MSE: " << mse / items.size() << std::endl;
  }
  std::cout << "Done." << std::endl;

  delete nn;
}
