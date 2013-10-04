#ifndef DEF_LTR_COMMON_H
#define DEF_LTR_COMMON_H
/**
 * @file
 * @author  David Nemeskey
 * @version 0.1
 *
 * @section LICENSE
 *
 * Copyright [2013] [MTA SZTAKI]
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
 * Contains definitions used by the whole LTR toolkit.
 */
#define DYNAMICEDATA 1

#ifndef DOC_ID_LENGTH
#define DOC_ID_LENGTH 10
#endif

//#include <limits>

#include <cstdio>

#include "graphchi_basic_includes.hpp"
#include "api/dynamicdata/chivector.hpp"

#include "object.h"

using namespace graphchi;

/** Vertex type that stores a single double. */
typedef enum { QUERY, DOCUMENT } VertexType;

/**
 * Vertex type that stores the type of the node (query or document) and the
 * real ID.
 */
struct TypeVertex {
  /** The (query or document) id. */
  char       id[DOC_ID_LENGTH];
  /** The vertex type. */
  VertexType type;

  TypeVertex(const char* id, VertexType type) : type(type) {
    snprintf(this->id, DOC_ID_LENGTH, "%s", id);
  }

  TypeVertex(size_t id, VertexType type) : type(type) {
    snprintf(this->id, DOC_ID_LENGTH, "%zu", id);
  }

  /** Default, do not use. */
  TypeVertex() {}
};

/** Header for the edges: stores relevance and score. */
struct EHeader : public Object {
  int relevance;
  vid_t doc;    // DEBUG only
  double score;
  
  EHeader() {}
  EHeader(int relevance_, vid_t doc_, double score_=0)
    : relevance(relevance_), doc(doc_), score(score_) {}
};

typedef chivector<double, EHeader> FeatureEdge;

///**
// * The edge data type. A chivector that stores the features, the relevance
// * data as the next-to-last element and (a placeholder for) the score given to
// * the document by the ranker (as an EHeader).
// */
//struct FeatureEdge : public chivector<double, EHeader>, Object {
//  FeatureEdge() {}
//  FeatureEdge(uint16_t cap, EHeader hdr, double* dataptr=NULL)
//    : chivector<double, EHeader>(0, cap, hdr, dataptr) {}
//  FeatureEdge(uint16_t sz, uint16_t cap, EHeader hdr, double* dataptr=NULL)
//    : chivector<double, EHeader>(sz, cap, hdr, dataptr) {}
//  FeatureEdge(size_t cap, EHeader hdr, double* dataptr=NULL)
//    : chivector<double, EHeader>(0, (uint16_t)cap, hdr, dataptr) {}
//
//  /**
//   * Returns the pointer to the inner data so that it can be encapsulated in an
//   * Eigen vector.
//   */
//  double* const& get_data() const {
//    return data;
//  }
//
//  std::string str() const {
//    std::ostringstream ss;
//    ss << "FeatureEdge (dim: " << NUM_FEATURES << ", rel: " << hdr.relevance
//       << ", score: " << hdr.score << "):";
//    for (size_t i = 0; i < NUM_FEATURES; i++) {
//      ss << " " << data[i];
//    }
//    return ss.str();
//  }
//};

//const vid_t max_query_id = std::numeric_limits<vid_t>::max() / 2 - 1;

#endif
