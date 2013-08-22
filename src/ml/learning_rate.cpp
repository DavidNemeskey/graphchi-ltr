#include "ml/learning_rate.h"

#include <cstdio>     // sscanf
#include <algorithm>  // copy_if, for_each
#include <iterator>   // back_inserter
#include <sstream>    // istringstream

#include <iostream>

LearningRate::LearningRate(double learning_rate_)
  : learning_rate(learning_rate_) {}

ConstantLearningRate::ConstantLearningRate(double learning_rate_)
    throw (std::invalid_argument) : LearningRate(learning_rate_) {
  if (learning_rate <= 0) {
    throw (std::invalid_argument("learning_rate must be greater than 0"));
  }
}

ConstantLearningRate* ConstantLearningRate::clone() {
  return new ConstantLearningRate(*this);
}

LinearLearningRate::LinearLearningRate(double starting_value_, double decrease_,
                                       double ending_value_)
    throw (std::invalid_argument) : LearningRate(starting_value_),
                                    start_point(starting_value_),
                                    decrease(decrease_),
                                    end_point(ending_value_) {
  if (starting_value_ <= 0) {
    throw (std::invalid_argument("starting_value must be greater than 0"));
  }
  if (ending_value_ < 0) {
    throw (std::invalid_argument("ending_value must not be less than 0"));
  }
  if (ending_value_ >= starting_value_) {
    throw (std::invalid_argument(
          "ending_value must be less than starting_value"));
  }
}

LinearLearningRate* LinearLearningRate::clone() {
  return new LinearLearningRate(*this);
}

bool LinearLearningRate::advance() {
  learning_rate -= decrease;
  if (learning_rate < end_point) {
    learning_rate = end_point;
    return false;
  } else {
    return true;
  }
}

void LinearLearningRate::reset() {
  learning_rate = start_point;
}

CompositeLearningRate::CompositeLearningRate(std::vector<LearningRate*> parts_)
    throw (std::invalid_argument) : index(0) {
  std::copy_if(parts_.begin(), parts_.end(), std::back_inserter(parts),
               [] (LearningRate* p) { return p != NULL; });
  if (parts.size() == 0) {
    throw (std::invalid_argument("no valid learning rate functions specified"));
  }
  learning_rate = parts[index]->get();
}

CompositeLearningRate::CompositeLearningRate(CompositeLearningRate& orig)
    : LearningRate(orig), index(orig.index) {
  parts.resize(orig.parts.size());
  for (size_t i = 0; i < parts.size(); i++) {
    parts[i] = orig.parts[i]->clone();
  }
}

CompositeLearningRate* CompositeLearningRate::clone() {
  return new CompositeLearningRate(*this);
}

bool CompositeLearningRate::advance() {
  if (index >= parts.size()) {
    return false;
  } else {
    bool ret = parts[index]->advance();
    learning_rate = parts[index]->get();
    if (!ret) {
      index++;
    }
    return true;
  }
}

void CompositeLearningRate::reset() {
  std::for_each(parts.begin(), parts.end(), std::mem_fun(&LearningRate::reset));
  index = 0;
  learning_rate = parts[index]->get();
}

LearningRate* create_learning_rate_function(const std::string& reflection_name)
    throw (std::invalid_argument) {
  std::istringstream ss(reflection_name);
  std::string name;
  std::getline(ss, name, ':');
  if (name == "LinearLearningRate" || name == "linear") {
    std::string params;
    std::getline(ss, params, '\n');
    double start = 0;
    double step = 0;
    double end = 0;
    sscanf(params.c_str(), "%lf:%lf:%lf", &start, &step, &end);
    return new LinearLearningRate(start, step, end);
  } else if (name == "ConstantLearningRate" || name == "constant") {
    double c = 0;
    ss >> c;
    return new ConstantLearningRate(c);
  } else if (name == "CompositeLearningRate" || name == "composite") {
    std::vector<LearningRate*> parts;
    while (ss.good()) {
      std::getline(ss, name, ';');
      if (!ss.fail() && !ss.bad()) {
        parts.push_back(create_learning_rate_function(name));
      }
    }
    return new CompositeLearningRate(parts);
  } else {
    return NULL;
  }
}

//int main(int argc, char* argv[]) {
//  std::vector<LearningRate*> v;
////  v.push_back(new LinearLearningRate(0.9, 0.1, 0.15));
//  //v.push_back(create_learning_rate_function("linear:0.9:0.1:0.15"));
////  v.push_back(new LinearLearningRate(0.9, 0.2, 0.15));
////  v.push_back(create_learning_rate_function("constant:0.5"));
//  //v.push_back(new ConstantLearningRate(0.5));
////  CompositeLearningRate clr(v);
//  //LearningRate* clr = create_learning_rate_function("composite:linear:0.9:0.1:0.15;constant:0.5");
//  LearningRate* clr = create_learning_rate_function("composite:linear:0.9:0.1:0.15;linear:0.9:0.1:0.15");
//  for (size_t i = 0; i < 15; i++) {
//    std::cout << "Learning rate " << clr->get() << std::endl;
//    if (!clr->advance()) break;
//  }
//  clr->reset();
//  std::cout << std::endl;
//  for (size_t i = 0; i < 15; i++) {
//    std::cout << "Learning rate " << clr->get() << std::endl;
//    if (!clr->advance()) break;
//  }
//}

