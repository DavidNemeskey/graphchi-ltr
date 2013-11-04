#pragma once
/**
 * @file
 * @author  David Nemeskey
 * @version 1.0
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
 * Contains reader objects for the usual input formats in LTR: the LETOR
 * dataset, csv, and the dual csv (one for queries, one for query-document
 * pairs) formats.
 */
#include <cstdlib>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

class InputFileReader {
public:
  InputFileReader(std::string file_name);

  ~InputFileReader();

  /**
   * Reads a line; returns the query and doc ids and the features in it.
   * @param[out] qid the query id.
   * @param[out] doc the document id.
   * @param[out] rel the relevance level.
   * @param[out] features the features converted to @c double.
   */
  virtual bool read_line(std::string& qid, std::string& doc, int& rel,
                         std::vector<double>& features)=0;

  /**
   * Returns the number of features.
   *
   * @todo Get rid of this once the static-dynamic API becomes stable.
   */
  virtual size_t num_features()=0;

protected:
  /**
   * Opens file @p name with error checking.
   * @param[in] name the name of the file.
   * @param[out] ifs the @c ifstream "returned".
   */
  void open_file(std::string name, std::ifstream& ifs);  // bool optional = false

  std::ifstream ifs;
};

/** Reads a CSV file. */
class CsvReader : public InputFileReader {
public:
  /**
   * @param rel_col if not applicable (test data), just specify any of the
   *                columns
   */
  CsvReader(const std::string& file_name, int qid_col=0, int doc_col=1,
            int rel_col=-1, bool has_header=true);

  bool read_line(std::string& qid, std::string& doc, int& rel,
                 std::vector<double>& features);

  /** Not implemented. */
  inline size_t num_features() { return 0; }

private:
  /**
   * Same as read_line(), but used used for the first line if any of
   * @c qid_col, @c doc_col, or @c rel_col is negative: it computes the
   * corresponding positive indices.
   */
  bool read_first_line(std::string& qid, std::string& doc, int& rel,
                       std::vector<double>& features);

  /** The index of the query id column. */
  int qid_col;
  /** The index of the document id column. */
  int doc_col;
  /** The index of the relevance level column. */
  int rel_col;
  /** If the first line is the headers. */
  bool has_header;

  /** Lines are read into this. */
  std::string line;
  /** Object for field separation. */
  std::stringstream ss;
  /** First line? */
  bool first_line;
};

class LetorReader : public InputFileReader {
public:
  LetorReader(const std::string& file_name);

  /** @p doc is not modified by this method, as it is not part of the data. */
  bool read_line(std::string& qid, std::string& doc, int& rel,
                 std::vector<double>& features);

  inline size_t num_features() { return VECTOR_LENGTH; }

  /** The number of features. */
  static size_t VECTOR_LENGTH;
private:
  /** Lines are read into this. */
  std::string line;
  /** Object for field separation. */
  std::stringstream ss;
};

class YahooChallengeReader : public InputFileReader {
public:
  YahooChallengeReader(const std::string& file_name);

  /** @p doc is not modified by this method, as it is not part of the data. */
  bool read_line(std::string& qid, std::string& doc, int& rel,
                 std::vector<double>& features);

  inline size_t num_features() { return vector_length; }

private:
  /** Reads the first line and sets vector_length. */
  bool read_first_line(std::string& qid, std::string& doc, int& rel,
                       std::vector<double>& features);

  /** The number of features. */
  size_t vector_length;
  /** Lines are read into this. */
  std::string line;
  /** Object for field separation. */
  std::stringstream ss;
};

