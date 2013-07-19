/**
 * @file
 * @author  David Nemeskey
 * @version 0.1
 *
 * @section LICENSE
 *
 * Copyright [2013] [Carnegie Mellon University]
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * This is the entry point for the Learning to Rank toolkit. The user can
 * specify what input dataset he wants to use and what algorithm, and the
 * control is then forwarded to the selected algorithm.
 */
#include <string>

#include "ltr_common.hpp"
#include "input_formats.hpp"
#include "ranknet.hpp"
#include "evaluation_measures.hpp"
#include "linear_regression.hpp"
//#include "neural_net.hpp"

using namespace graphchi;

/**
 * Reads a dataset.
 * @return the number of shards.
 */
int read_data(std::string file_name, std::string file_type, size_t& dimensions) {
  if (file_type == "csv") {
    int qid_index = get_option_int("qid", 0);
    int doc_index = get_option_int("doc", 1);
    int rel_index = get_option_int("rel", -1);
    return read_csv(file_name, dimensions, qid_index, doc_index, rel_index);
  } else if (file_type == "letor") {
    return read_letor(file_name, dimensions);
  } else {
    return 0;
  } 
}

/** Instantiates the selected algorithm. */
LtrAlgorithm* get_algorithm(std::string name, MlModel* model,
                            EvaluationMeasure* eval) {
  if (name == "ranknet") {
    return new RankNet(model, eval);
  } else {
    return NULL;
  }
}

/** Instantiates the ML model. */
MlModel* get_ml_model(std::string name, size_t dimensions) {
    if (name == "linreg") {
        return new LinearRegression(dimensions);
    } else if (name == "nn") {
        return new NeuralNetwork(dimensions, /* TODO*/ 8);
    } else {
        return NULL;
    }
}

/**
 * Instantiates the evaluator object.
 * @param[in] cutoff the "at" in "nDCG@20".
 */
EvaluationMeasure* get_evaluation_measure(std::string name, int cutoff) {
  if (name == "ndcg") {
    return new NdcgEvaluator(cutoff);
  } else {
    return NULL;
  }
}

int main(int argc, const char ** argv) {
//  print_copyright();
  NeuralNetwork(50, 20);

  /* GraphChi initialization will read the command line 
     arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Parameters */
  std::string train_data = get_option_string("train_data");  // TODO: not needed (save/load model)
  std::string eval_data = get_option_string("eval_data", "");
  std::string test_data = get_option_string("test_data", "");
  int niters            = get_option_int("niters", 10);
  int cutoff            = get_option_int("cutoff", 20);
  int learning_rate     = get_option_int("lrate", 20);
  // TODO: make it overridable by --D?
  size_t dimensions     = 0;
  bool scheduler        = false;  // No scheduler is needed
  std::string reader       = get_option_string("reader");
  std::string error_metric = get_option_string("error", "ndcg");
  std::string model_name     = get_option_string("mlmodel", "linreg");
  std::string algorithm_name = get_option_string("algorithm", "ranknet");

  /* Read the data file. */
  int train_nshards = read_data(train_data, reader, dimensions);
  if (train_nshards == 0) {
    logstream(LOG_FATAL) << "Reader " << reader << " is not implemented. " <<
                            "Select one of csv, letor." << std::endl;
  }

  /* Instantiate the algorithm. */
  MlModel* model = get_ml_model(model_name, dimensions);
  if (model == NULL) {
    logstream(LOG_FATAL) << "Model " << model_name <<
                            " is not implemented; select one of " <<
                            "linreg, nn." << std::endl;
  }
  EvaluationMeasure* eval = get_evaluation_measure(error_metric, cutoff);
  if (eval == NULL) {
    logstream(LOG_FATAL) << "Evaluation metric " << error_metric <<
                            " is not implemented; select one of " <<
                            "ndcg, err, map." << std::endl;
  }
  LtrAlgorithm* algorithm = get_algorithm(algorithm_name, model, eval);
  if (algorithm == NULL) {
    logstream(LOG_FATAL) << "Algorithm " << algorithm_name <<
                            " is not implemented; select one of " <<
                            "ranknet, lambdarank, lambdamart." << std::endl;
  }

  /* Training. */
  metrics m_train("ltr_train");
  graphchi_engine<TypeVertex, FeatureEdge> engine(
      train_data, train_nshards, scheduler, m_train); 
  engine.run(*algorithm, niters);
  metrics_report(m_train);

  /* Validation. */
  if (eval_data != "") {
    int eval_nshards = read_data(eval_data, reader, dimensions);
    if (eval_nshards == 0) {
      logstream(LOG_FATAL) << "Reader " << reader << " is not implemented. " <<
                              "Select one of csv, letor." << std::endl;
    }
    algorithm->set_phase(VALIDATION);
    metrics m_eval("ltr_eval");
    graphchi_engine<TypeVertex, FeatureEdge> engine(
        eval_data, eval_nshards, scheduler, m_eval); 
    engine.run(*algorithm, niters);
    metrics_report(m_eval);
  }

  /* Testing. */
  if (test_data != "") {
    int test_nshards = read_data(test_data, reader, dimensions);
    if (test_nshards == 0) {
      logstream(LOG_FATAL) << "Reader " << reader << " is not implemented. " <<
                              "Select one of csv, letor." << std::endl;
    }
    algorithm->set_phase(VALIDATION);
    metrics m_test("ltr_test");
    graphchi_engine<TypeVertex, FeatureEdge> engine(
        test_data, test_nshards, scheduler, m_test); 
    engine.run(*algorithm, niters);
    metrics_report(m_test);
  }

  return 0;
}

/*
 * Problems:
 * - MlModel: must have a save/load function
 */

