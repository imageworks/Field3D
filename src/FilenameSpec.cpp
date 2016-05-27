//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2016 Sony Pictures Imageworks Inc.,
 *                    Pixar Animation Studios
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the
 * distribution.  Neither the name of Sony Pictures Imageworks nor the
 * names of its contributors may be used to endorse or promote
 * products derived from this software without specific prior written
 * permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//----------------------------------------------------------------------------//

/*! \file FilenameSpec.cpp
  Contains filename spec implementations
*/

//----------------------------------------------------------------------------//

// Header include
#include "FilenameSpec.h"

// Project includes
#include "PatternMatch.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// FilenameSpec
//----------------------------------------------------------------------------//

FilenameSpec::FilenameSpec(const std::string &base)
  : m_index(std::string::npos)
{
  // Check for specifiers
  if (base.find_first_of(":") != std::string::npos) {
    std::vector<std::string> parts = split(base, ":");
    if (parts.size() == 2) {
      // File:Attribute
      m_filename  = parts[0];
      m_name      = "*";
      m_attribute = parts[1];
    } else if (parts.size() == 3) {
      // File:Name:Attribute
      m_filename  = parts[0];
      m_name      = parts[1];
      m_attribute = parts[2];
    } else if (parts.size() == 4) {
      // File:Name:Attribute:Index
      m_filename  = parts[0];
      m_name      = parts[1];
      m_attribute = parts[2];
      try {
        m_index = boost::lexical_cast<int>(parts[3]);
      }
      catch (...) {
        std::cout << "WARNING: Ill-formed index spec in FilenameSpec() : " 
                  << base << std::endl;
      }
    } else {
      std::cout << "WARNING: Ill-formed spec in FilenameSpec() : " 
                << base << std::endl;
    }
  } else {
    m_filename = base;
    m_name = "*";
    m_attribute = "*";
  }
  // Update pattern
}
  
//----------------------------------------------------------------------------//

FilenameSpec::FilenameSpec(const std::string &filename,
                           const std::string &name,
                           const size_t index,
                           const std::string &attribute)
  : m_filename(filename),
    m_name(name),
    m_attribute(attribute),
    m_index(index)
{

}

//----------------------------------------------------------------------------//

const std::string& 
FilenameSpec::filename() const
{ 
  return m_filename; 
}

//----------------------------------------------------------------------------//

std::string 
FilenameSpec::pattern() const
{ 
  if (m_name.size() > 0) {
    if (m_attribute.size() > 0) {
      return m_name + ":" + m_attribute; 
    } else {
      return m_name + ":*"; 
    }
  } else {
    if (m_attribute.size() > 0) {
      return m_attribute;
    } else {
      return "";
    }
  }
}

//----------------------------------------------------------------------------//

const std::string& 
FilenameSpec::name() const
{ 
  return m_name; 
}

//----------------------------------------------------------------------------//

void
FilenameSpec::setName(const std::string &name)
{
  m_index = std::string::npos;
  m_name = name;
}

//----------------------------------------------------------------------------//

const std::string& 
FilenameSpec::attribute() const
{ 
  return m_attribute; 
}

//----------------------------------------------------------------------------//

void
FilenameSpec::setAttribute(const std::string &attribute)
{
  m_index = std::string::npos;
  m_attribute = attribute;
}

//----------------------------------------------------------------------------//

size_t 
FilenameSpec::index() const
{ 
  return m_index; 
}

//----------------------------------------------------------------------------//

bool 
FilenameSpec::isUnique() const
{ 
  return m_index != std::string::npos; 
}

//----------------------------------------------------------------------------//

std::ostream&
operator << (std::ostream &os, const FilenameSpec &spec)
{
  os << "{ file: " << spec.filename() 
     << ", name: " << spec.name() 
     << ", attr: " << spec.attribute() 
     << ", index: ";
  if (spec.isUnique()) {
    os << spec.index();
  } else {
    os << "none";
  }
  os << " }";
  return os;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
