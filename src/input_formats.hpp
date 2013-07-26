/**
 * @file
 * @author  David Nemeskey
 * @version 1.0
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
 * Contains readers for the usual input formats in LTR: the LETOR dataset, csv,
 * and the dual csv (one for queries, one for query-document pairs) format.
 */
/* Use dynamic edge data (i.e. chivector). */

#include <map>
#include <string>

#include "ltr_common.hpp"
#include "input_readers.h"
#include "graphchi_basic_includes.hpp"

using namespace graphchi;

/**
 * Reads the data from a csv file. The document and query ids can be strings;
 * all features are converted to doubles.
 *
 * @param file_name the name of the file.
 * @param[out] dimensions the number of features is written to this parameter.
 * @param qid_col the column of the query id. 0 by default.
 * @param doc_col the column of the document id. 1 by default.
 * @param rel_col the column with the relevance level; by default, the last one.
 * @param has_header if true, the first row is disregarded.
 * @return the number of shards.
 * @todo compute number of queries instead of expecting it as parameter.
 */
int read_csv(const std::string& file_name, size_t& dimensions,
             int qid_col=0, int doc_col=1, int rel_col=-1,
             bool has_header=true) {
  /* Create sharder object */
  int nshards;
  sharder<FeatureEdge> sharderobj(file_name);
  sharderobj.start_preprocessing();

  CsvReader reader(file_name, qid_col, doc_col, rel_col, has_header);
  std::string qid;
  std::string doc;
  int relevance;
  std::vector<double> features;
  while (reader.read_line(qid, doc, relevance, features)) {
    FeatureEdge edge_data;
    // TODO: ids might be non-consecutive, use something to handle this
    vid_t qid_i = (vid_t)strtoul(qid.c_str(), NULL, 10);
    vid_t doc_i = (vid_t)strtoul(doc.c_str(), NULL, 10);
    // DEBUG only
    edge_data.doc = doc_i;
    // TODO: not really efficient, not safe
    //DYN if (rel_col != (int)features.size()) {
      //DYN features.push_back(relevance_level);
    //DYN }
    edge_data.relevance = relevance;
    // Placeholder for the score given to the document by the model
    //DYN features.push_back(0);
    //DYN FeatureEdge edge_data(features.size(), features.size(), &features[0]);
    if (features.size() > NUM_FEATURES) {
      sharderobj.end_preprocessing();
      logstream(LOG_FATAL) << "The number of features (" << features.size() <<
                              ") is too high; recompile with -DNUM_FEATURES " <<
                              "set to the appropriate number." << std::endl;
    } else {
      for (size_t i = 0; i < features.size(); i++) {
        edge_data.features[i] = features[i];
      }
    }
    sharderobj.preprocessing_add_edge(
        qid_i, doc_i, edge_data);
//        qid_i, doc_i + number_of_queries, edge_data);
  }
  /* Save the number of features. */
  dimensions = features.size();

  sharderobj.end_preprocessing();

  logstream(LOG_INFO) << "Now creating shards." << std::endl;

  /*
   * Shard with a specified number of shards, or determine automatically if not
   * defined
   */
  nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));

  return nshards;
}

/**
 * Reads the LETOR format.
 *
 * Document ids start from <tt>max_query_id + 1</tt>. It may not be the best
 * solution, but there you have it.
 *
 * @param file_name the name of the file.
 * @param number_of_queries still need this shit. :(
 * @param[out] dimensionsa the number of features is written to this parameter.
 */
int read_letor(const std::string& file_name, size_t& dimensions) {
  int nshards;
  sharder<FeatureEdge> sharderobj(file_name);
  sharderobj.start_preprocessing();

  LetorReader reader(file_name);
  if (NUM_FEATURES < LetorReader::VECTOR_LENGTH) {
    sharderobj.end_preprocessing();
    logstream(LOG_FATAL) << "The number of features (" <<
                            LetorReader::VECTOR_LENGTH <<
                            ") is too high; recompile with -DNUM_FEATURES " <<
                            "set to the appropriate number." << std::endl;
  }
  dimensions = LetorReader::VECTOR_LENGTH;

  /* For read_line. */
  std::string qid;
  int relevance;
  std::vector<double> features;

  vid_t curr_node = 0;  // The current node
  size_t line     = 0;  // The number of the current line
  std::map<std::string, vid_t> qids;
  TypeVertex vertex_data;

  /* The vertex data file. */
  std::string filename = filename_vertex_data<TypeVertex>(file_name);
  FILE* f = fopen(filename.c_str(), "w");

  while (reader.read_line(qid, qid, relevance, features)) {
    vid_t qid_i;
    vid_t doc_i;

    FeatureEdge edge_data;
    if (qids.find(qid) == qids.end()) {
      /* Write the vertex data. */
      vertex_data = TypeVertex(qid.c_str(), QUERY);
      fwrite(&vertex_data, sizeof(TypeVertex), 1, f);
      qids[qid] = curr_node++;
    }
    qid_i = qids[qid];
    doc_i = curr_node++;
    /* Write the vertex data. */
    vertex_data = TypeVertex(line++, DOCUMENT);
    fwrite(&vertex_data, sizeof(TypeVertex), 1, f);

    // DEBUG only
    edge_data.doc = doc_i;
    edge_data.relevance = relevance;
    for (size_t i = 0; i < features.size(); i++) {
      edge_data.features[i] = features[i];
    }
    sharderobj.preprocessing_add_edge(qid_i, doc_i, edge_data);
  }

  fclose(f);

  sharderobj.end_preprocessing();

  /*
   * Shard with a specified number of shards, or determine automatically if not
   * defined
   */
  nshards = sharderobj.execute_sharding(get_option_string("nshards", "auto"));

  return nshards;
}

