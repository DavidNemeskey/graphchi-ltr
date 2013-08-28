#ifndef DEF_OBJECT_H
#define DEF_OBJECT_H
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
 * Contains useful methods for all (most) objects -- see Java's Object. Also
 * defines UnsupportedOperationException.
 */

#include <stdexcept>   // runtime_error
#include <string>

struct UnsupportedOperationException : public std::runtime_error {
  UnsupportedOperationException() : std::runtime_error("Not Implemented.") {}
};


struct Object {
  /**
   * Returns a clone of the current object.
   *
   * This default implementation throws UnsupportedOperationException.
   *
   * Subclasses should override this method to call their copy constructor, or
   * perform any other cloning strategy (deep copy, return this, etc). Not only
   * that, they should declare the function with a covariant return value.
   */
  inline virtual Object* clone() {
    throw UnsupportedOperationException();
  }

  /**
   * Returns the textual representation of the current Object. This default
   * implementation returns the string @c Object.
   */
  inline virtual std::string str() const { return "Object"; }
};

#endif
