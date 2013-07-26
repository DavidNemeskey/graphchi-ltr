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
//DYN #define DYNAMICEDATA 1
#ifndef NUM_FEATURES
#define NUM_FEATURES 10
#endif

#ifndef DOC_ID_LENGTH
#define DOC_ID_LENGTH 10
#endif

//#include <limits>

#include <cstdio>

#include "graphchi_basic_includes.hpp"
#include "api/dynamicdata/chivector.hpp"

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

/**
 * The edge data type. A chivector that stores the features, the relevance
 * data as the next-to-last element and (a placeholder for) the score given to
 * the document by the ranker.
 */
//DYN typedef chivector<double> FeatureEdge;
typedef struct {
  vid_t doc;    // DEBUG only
  int relevance;
  double score;
  double features[NUM_FEATURES];
} FeatureEdge;

//const vid_t max_query_id = std::numeric_limits<vid_t>::max() / 2 - 1;

#endif
