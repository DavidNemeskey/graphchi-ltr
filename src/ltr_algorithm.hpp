#ifndef DEF_LTR_ALGORITHM_H
#define DEF_LTR_ALGORITHM_H
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
 * The LtrAlgorithm base class.
 */

#include <iostream>
#include <iomanip>
#include <map>
#include <vector>

#include "ltr_common.hpp"
//#include "util/pthread_tools.hpp"  // mutex
#include "ml_algorithm.hpp"
#include "evaluation_measures.hpp"

/** The three phases of the LTR algorithm. */
enum LtrRunningPhase {
  TRAINING,
  VALIDATION,
  TESTING
};

class LtrAlgorithm : public GraphChiProgram<TypeVertex, FeatureEdge> {
public:
  /**
   * @param[in] number_of_queries the number of query nodes.
   * @param[in] model the ML model; deleted together with this object.
   * @param[in] phase which phase of the algorithm to run.
   */
  LtrAlgorithm(MlModel* model, EvaluationMeasure* eval,
               LtrRunningPhase phase=TRAINING)
      : model(model), eval(eval), phase(phase) {
  }

  ~LtrAlgorithm() {
    delete model;
    delete eval;
    for (std::vector<MlModel*>::iterator it = parallel_models.begin();
         it != parallel_models.end(); ++it) {
      delete *it;
    }
  }

  /**
   * Changes the phase -- if after learning, validation or testing is also
   * needed.
   */
  void set_phase(LtrRunningPhase phase) {
    this->phase = phase;
  }

  /****************************** GraphChi stuff ******************************/

  /**
   * Called after an iteration has finished. Aggregates the evaluation measure.
   * Also creates a number of copies of the ML model equal to the number of
   * execution threads, so that the model update can be parallel.
   */
  void before_iteration(int iteration, graphchi_context &ginfo) {
    if (phase == TRAINING || phase == VALIDATION) {
      eval->before_iteration(iteration, ginfo);
    }
    for (int i = 0; i < ginfo.execthreads; i++) {
      if (iteration == 0) {
        parallel_models.push_back(model->clone());
      } else {
        *parallel_models[i] = *model;
      }
    }
    std::cout << std::setprecision(10);
    std::cout << "LINREG_UPDATE BEFORE ";
    LinearRegression* lr_model = (LinearRegression*)model;
    std::copy(lr_model->weights.begin(), lr_model->weights.end(),
              std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl;
  }

  /**
   * This method runs only for the query nodes. Its actual function is divided
   * into several methods, as not all is needed in each phase.
   */
  void update(graphchi_vertex<TypeVertex, FeatureEdge> &v,
              graphchi_context &ginfo) {
    // TODO Use a scheduler instead of this?
    if (v.get_data().type == QUERY) {  // Only queries have outedges (TODO: ???)
      score_documents(v, ginfo);
      if (phase == TRAINING) {
        update_weights(v, parallel_models[omp_get_thread_num()]);
      }
      if (phase == TRAINING || phase == VALIDATION) {
        evaluate_model(v, ginfo);
      }
    }
  }

  /**
   * Called after an iteration has finished. Aggregates the model updates and
   * the evaluation measure.
   */
  void after_iteration(int iteration, graphchi_context &ginfo) {
    // TODO: to separate class?

//    std::cout << "LINREG_UPDATES:" << std::endl;
    /* Compute the delta. */
    for (int i = 0; i < ginfo.execthreads; i++) {
      *(parallel_models[i]) -= *model;
    }
    /* Add the delta. */
    for (int i = 0; i < ginfo.execthreads; i++) {
      (*model) += *parallel_models[i];
    }
//    std::cout << "LINREG_UPDATE AFTER ";
//    LinearRegression* lr_model = (LinearRegression*)model;
//    std::copy(lr_model->weights.begin(), lr_model->weights.end(),
//              std::ostream_iterator<double>(std::cout, " "));
//    std::cout << std::endl;

    if (phase == TRAINING || phase == VALIDATION) {
      eval->after_iteration(iteration, ginfo);

      // Debugging stuff; remove if not needed anymore.
//      std::cout << "WEIGHTS: ";
//      std::copy(((LinearRegression*)model)->weights.begin(), ((LinearRegression*)model)->weights.end(), std::ostream_iterator<double>(std::cout, " "));
//      std::cout << std::endl;
      std::cout << "NDCG: ";
      for (std::map<vid_t, double>::const_iterator it = eval->eval.begin();
          it != eval->eval.end(); ++it) {
        std::cout << it->second << " ";
      }
//      std::copy(eval->eval.begin(), eval->eval.end(), std::ostream_iterator<double>(std::cout, " "));
      std::cout << ", avg: " << eval->avg_eval << std::endl << std::endl;
    }
  }

protected:
  /** Scores all documents for the query. The first step in update(). */
  void score_documents(graphchi_vertex<TypeVertex, FeatureEdge> &query,
                       graphchi_context &ginfo) {
    for (int doc = 0; doc < query.num_outedges(); doc++) {
      //DYN model->score(*(v.outedge(e)->get_vector()));
      FeatureEdge fe = query.outedge(doc)->get_data();
      fe.score = model->score(fe.features);
      query.outedge(doc)->set_data(fe);
    }
  }

  /**
   * Updates the weights of the model. The second step in update().
   *
   * The @p umodel is one of @c parallel_models that corresponds to the current
   * execthread.
   */
  virtual void update_weights(graphchi_vertex<TypeVertex, FeatureEdge> &query,
                              MlModel* umodel)=0;


  /** Evaluates the model. The third step in update(). */
  void evaluate_model(graphchi_vertex<TypeVertex, FeatureEdge> &query,
                      graphchi_context &ginfo) {
    eval->update(query, ginfo);
  }

  /** Returns the score on a query-document edge. */
  inline double get_score(graphchi_edge<EdgeDataType>* edge) {
    //DYN FeatureEdge* i_vect = edge->get_vector();
    //DYN return i_vect->get(i_vect->size() - 1);
    return edge->get_data().score;
  }

  /** Returns the relevance of a query-document pair. */
  inline int get_relevance(graphchi_edge<EdgeDataType>* edge) {
    //DYN FeatureEdge* i_vect = edge->get_vector();
    //DYN return i_vect->get(i_vect->size() - 2);
    return edge->get_data().relevance;
  }

protected:
  /** The model that backs up RankNet. */
  MlModel* model;
  /** The evaluation measure. */
  EvaluationMeasure* eval;
  /** Which phase to run? */
  LtrRunningPhase phase;
  /**
   * Model pool used by the execthreads. The updates modify these models instead
   * of the central one.
   * @see model
   */
  std::vector<MlModel*> parallel_models;
};

#endif
