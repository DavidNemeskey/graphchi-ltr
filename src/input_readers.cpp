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

#include <iostream>
#include "input_readers.h"
#include <set>
//#include "graphchi_basic_includes.hpp"

InputFileReader::InputFileReader(std::string file_name) {
  open_file(file_name, ifs);
}

InputFileReader::~InputFileReader() {
  ifs.close();
}

void InputFileReader::open_file(std::string name, std::ifstream& ifs) {
  ifs.open(name.c_str());
  if (!ifs) {
    perror("open_file failed");
//    logstream(LOG_FATAL) << "Failed to open file " << name << std::endl;
  }
}

CsvReader::CsvReader(const std::string& file_name, int qid_col, int doc_col,
                     int rel_col, bool has_header)
    : InputFileReader(file_name), qid_col(qid_col), doc_col(doc_col),
      rel_col(rel_col), has_header(has_header), first_line(true) {
  if (has_header) {
    std::getline(ifs, line);
  }
}

bool CsvReader::read_line(std::string& qid, std::string& doc, int& rel,
                          std::vector<double>& features) {
  if (first_line && (qid_col < 0 || doc_col < 0 || rel_col < 0)) {
    first_line = false;
    return read_first_line(qid, doc, rel, features);
  } else {
    std::getline(ifs, line);
    if (!ifs) return false;

    ss.clear();
    ss.str(line);
    std::string token;
    features.clear();
    for (int i = 0; ; i++) {
      std::getline(ss, token, ',');
      if (!ss) break;
      if (i == qid_col) {
        qid = token;
      } else if (i == doc_col) {
        doc = token;
      } else if (i != rel_col) {
        features.push_back(atof(token.c_str()));
      }
      /* rel can be anything (e.g. if there is no such column, but we need to
       * specify it anyway). */
      if (i == rel_col) {
        rel = atoi(token.c_str());
      }
    }
    return true;
  }
}

bool CsvReader::read_first_line(std::string& qid, std::string& doc, int& rel,
                                std::vector<double>& features) {
  std::getline(ifs, line);
  if (!ifs) return false;

  ss.clear();
  ss.str(line);
  std::string token;
  features.clear();

  std::vector<std::string> string_fields;
  while (true) {
    std::getline(ss, token, ',');
    if (!ss) break;
    string_fields.push_back(token);
  }

  while (qid_col < 0) {
    qid_col += string_fields.size();
  }
  while (doc_col < 0) {
    doc_col += string_fields.size();
  }
  while (rel_col < 0) {
    rel_col += string_fields.size();
  }
  qid = string_fields[qid_col];
  doc = string_fields[doc_col];
  rel = atoi(string_fields[rel_col].c_str());
  std::set<int> to_remove;
  to_remove.insert(qid_col);
  to_remove.insert(doc_col);
  to_remove.insert(rel_col);
  for (std::set<int>::const_reverse_iterator it = to_remove.rbegin();
                                             it != to_remove.rend(); ++it) {
    string_fields.erase(string_fields.begin() + *it);
  }

  for (std::vector<std::string>::const_iterator it = string_fields.begin();
      it != string_fields.end(); ++it) {
    features.push_back(atof(it->c_str()));
  }
  return true;
}

LetorReader::LetorReader(const std::string& file_name)
  : InputFileReader(file_name) {}

size_t LetorReader::VECTOR_LENGTH = 136;

bool LetorReader::read_line(std::string& qid, std::string& doc, int& rel,
                            std::vector<double>& features) {
  std::getline(ifs, line);
  if (!ifs) return false;

  ss.clear();
  ss.str(line);
  std::string token;
  features.resize(VECTOR_LENGTH);
  for (int i = 0; ; i++) {
    std::getline(ss, token, ' ');
    if (!ss) break;
    if (i == 0) {
      rel = atoi(token.c_str());
    } else {
      size_t pos = token.find(":");
      std::string pos_str = token.substr(0, pos);
      std::string val_str = token.substr(pos + 1);
      if (pos_str == "qid") {
        qid = val_str;
      } else {
        features[atoi(pos_str.c_str())] = atof(val_str.c_str());
      }
    }
  }
  return true;
}

YahooChallengeReader::YahooChallengeReader(const std::string& file_name)
  : InputFileReader(file_name), vector_length(0) {}

bool YahooChallengeReader::read_line(std::string& qid, std::string& doc,
                                     int& rel, std::vector<double>& features) {
  if (vector_length == 0) {
    return read_first_line(qid, doc, rel, features);
  } else {
    std::getline(ifs, line);
    if (!ifs) return false;

    ss.clear();
    ss.str(line);
    std::string token;
    features.resize(vector_length);
    for (size_t i = 0; ; i++) {
      std::getline(ss, token, ',');
      if (!ss) break;
      if (i == 0) {
        qid = token;
      } else if (i == vector_length + 1) {
        rel = atoi(token.c_str());
      } else {
        features[i - 1] = atof(token.c_str());
      }
    }
    return true;
  }
}

bool YahooChallengeReader::read_first_line(std::string& qid, std::string& doc,
                                           int& rel,
                                           std::vector<double>& features) {
  std::getline(ifs, line);
  if (!ifs) return false;

  ss.clear();
  ss.str(line);
  std::string token;
  features.resize(0);
  for (size_t i = 0; ; i++) {
    std::getline(ss, token, ',');
    if (!ss) break;
    if (i == 0) {
      qid = token;
    } else {
      features.push_back(atof(token.c_str()));
    }
  }
  if (features.size() == 0) return false;
  vector_length = features.size() - 1;
  rel = static_cast<int>(features.back());
  features.pop_back();
  return true;
}

