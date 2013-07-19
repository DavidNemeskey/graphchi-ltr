#ifndef DEF_EVALUATION_MEASURES_H
#define DEF_EVALUATION_MEASURES_H
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
 * Contains the evaluation measures used in information retrieval.
 */

#include <vector>
#include <algorithm>  // std::min
#include <numeric>    // std::accumulate

#include "ltr_common.hpp"

class EvaluationMeasure : public GraphChiProgram<TypeVertex, FeatureEdge> {
public:
  EvaluationMeasure(int cutoff)
    : cutoff(cutoff) {}

  /**
   * Clears the eval object before each iteration. This might not be the fastest
   * way to do this, but at this point I don't care.
   */
  void before_iteration(int iteration, graphchi_context &ginfo) {
    if (iteration == 0) {
      eval.clear();
    }
  }

  void update(graphchi_vertex<TypeVertex, FeatureEdge> &vertex,
              graphchi_context &gcontext) {
    if (vertex.get_data().type == DOCUMENT) {
      return;
    }

    if (gcontext.iteration == 0) {
      // TODO: mutex!
      eval[vertex.id()] = 0;
    }
    compute_measure(vertex, gcontext);
  }

  /** Aggregates the evaluations. */
  void after_iteration(int iteration, graphchi_context &ginfo) {
    avg_eval = 0;
    for (std::map<vid_t, double>::const_iterator it = eval.begin();
         it != eval.end(); ++it) {
      avg_eval += it->second;
    }
    avg_eval /= eval.size();
//    avg_eval = std::accumulate(eval.begin(), eval.end(), 0.0) / eval.size();
    std::cout << "EVAL ITERATION: " << iteration << std::endl;
  }

protected:
  /**
   * This is the method that does the actual computation and what subclasses
   * must implement this. Since this the evaluators run after an iteration of
   * the learning algorithm, the scores on the edges should be valid.
   *
   * When this method is called for the <tt>n</tt>'s query, <tt>eval</tt>'s size
   * is guaranteed to be @c n. Implementations of this method should write the
   * computed measure into the last element.
   */
  virtual void compute_measure(graphchi_vertex<TypeVertex, FeatureEdge> &v,
                               graphchi_context &gcontext)=0;

  /**
   * Creates a heap from the @c cutoff best edges from vertex @p v,
   * according to comp.
   */
  std::vector<FeatureEdge> get_best(
            graphchi_vertex<TypeVertex, FeatureEdge> &v,
            bool (*comp)(FeatureEdge&, FeatureEdge&))
  {
    int heap_size = std::min(cutoff, v.num_edges());
    //DYN std::vector<FeatureEdge*> best(heap_size);
    std::vector<FeatureEdge> best(heap_size);
    for (int e = 0; e < heap_size; e++) {
      //DYN best[e] = v.edge(e)->get_vector();
      best[e] = v.edge(e)->get_data();
    }
    std::make_heap(best.begin(), best.end(), comp);
    if (heap_size < v.num_edges()) {
      std::pop_heap(best.begin(), best.end(), comp);
      for (int e = heap_size; e < v.num_edges(); e++) {
        //DYN FeatureEdge* vect = v.edge(e)->get_vector();
        FeatureEdge vect = v.edge(e)->get_data();
        if (EvaluationMeasure::rel_comp(vect, best[best.size() - 1])) {
          best[best.size() - 1] = vect;
          std::push_heap(best.begin(), best.end(), comp);
          std::pop_heap(best.begin(), best.end(), comp);
        }
      }
    }
    std::sort_heap(best.begin(), best.end(), comp);

    return best;
  }

  /** For inverse heap sorting by score. Needed by the implementations. */
  static bool score_comp(FeatureEdge& e1, FeatureEdge& e2) {
    //DYN return e1->get(e1->size() - 1) > e2->get(e2->size() - 1);
    return e1.score > e2.score;
  }

  /** For inverse heap sorting by relevance. Needed by the implementations. */
  static bool rel_comp(FeatureEdge& e1, FeatureEdge& e2) {
    //DYN return e1->get(e1->size() - 2) > e2->get(e2->size() - 2);
    return e1.relevance > e2.relevance;
  }

public:
  /**
   * The values of the evaluation measure for each query. Cannot be a vector
   * because the order the vertices are traversed is not constant.
   */
//  std::vector<double> eval;
  std::map<vid_t, double> eval;
  /** The average of the former; computed in after_iteration(). */
  double avg_eval;

protected:
  /** The "at" in "nDCG@20". */
  int cutoff;
};

/** The nDCG measure. */
class NdcgEvaluator : public EvaluationMeasure {
public:
  NdcgEvaluator(int cutoff) : EvaluationMeasure(cutoff) {}

  /**
   * Computes the DCG. If @p comp is @c score_comp, the method computes DCG; if
   * @c rel_comp is used, it computes IDCG. 
   */
  double compute_dcg(graphchi_vertex<TypeVertex, FeatureEdge> &v,
                     bool (*comp)(FeatureEdge&, FeatureEdge&)) {
    std::vector<FeatureEdge> best = get_best(v, comp);
    std::cout << "RANKING for query " << v.get_data().id << ": ";

    double dcg = 0;
    for (size_t i = 0; i < best.size(); i++) {
      std::cout << best[i].doc << "(" << best[i].relevance << "), ";
      //DYN dcg += (pow(2, best[i]->get(best[i]->size() - 2)) - 1) /
      dcg += (pow(2, best[i].relevance) - 1) /
             (log(i + 2) / log(2));
    }
    std::cout << std::endl;

    return dcg;
  }

  void compute_measure(graphchi_vertex<TypeVertex, FeatureEdge> &v,
                                 graphchi_context &gcontext) {
    /** Compute the DCG. */
    if (gcontext.iteration == 0) {
      // TODO: mutex!
      idcgs[v.id()] = compute_dcg(v, EvaluationMeasure::rel_comp);
      std::cout << "IDCG[" << v.id() << "] = " << idcgs[v.id()] << std::endl;
    }
    double dcg = compute_dcg(v, EvaluationMeasure::score_comp);
    eval[v.id()] = dcg / idcgs[v.id()];
    std::cout << "NDCG[" << v.get_data().id << "] = " << dcg << " / " << idcgs[v.id()]
              << " = " << eval[v.id()] << std::endl;
  }

private:
  /** Stores the DCGs for the query nodes. Required to compute nDCG. */
//  std::vector<double> idcgs;
  std::map<vid_t, double> idcgs;
};

///**
// * Just for convenience: the evaluation measure object and the engine in the
// * same object.
// */
//struct EvaluationTuple {
//  EvaluationTuple(EvaluationMeasure* eval, int nshards) {
//    metrics m("validation engine");
//    graphchi_engine<TypeVertex, FeatureEdge>* engine =
//        new graphchi_engine<TypeVertex, FeatureEdge>(nshards, false, m);
//  }
//};

#endif
