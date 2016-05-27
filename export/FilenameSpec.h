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

/*! \file FilenameSpec.h
  \brief Contains FilenameSpec class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_FilenameSpec_H_
#define _INCLUDED_Field3D_FilenameSpec_H_

//----------------------------------------------------------------------------//

#include <iostream>
#include <string>
#include <vector>

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//------------------------------------------------------------------------------
// FilenameSpec
//------------------------------------------------------------------------------

/*! The FilenameSpec class wraps up the specification of a field in a file:
  
     file.f3d                  All fields in a file
     file.f3d:levelset         Fields with attribute='levelset'
     file.f3d:woody:levelset   Fields with name='woody', attribute='levelset'
     file.f3d:woody:levelset:1 Unique field with name, attribute, index
     file.f3d:woody:*          Specified name, wildcard attributes

 */

//------------------------------------------------------------------------------

class FilenameSpec
{
public:

  // Constants ---

  static const char k_separator;

  // Typedefs ---

  typedef std::vector<FilenameSpec> Vec;

  // Ctors ---

  //! Parses filename and possibly name/attribute/index from the base filename
  FilenameSpec(const std::string &base);
  FilenameSpec(const std::string &filename,
               const std::string &name,
               const size_t index,
               const std::string &attribute);

  // Accessors ---

  //! Returns the filename part of the spec
  const std::string& filename() const;
  //! Returns the pattern that matches the original base name.
  //! \note This is only relevant if hasIndex() is false
  std::string pattern() const;
  //! Returns the field name part of the spec
  const std::string& name() const;
  //! Resets the name
  //! \note Breaks indices, so spec is no longer unique after
  void setName(const std::string &name);
  //! Returns the field attribute part of the spec
  const std::string& attribute() const;
  //! Resets the attribute
  void setAttribute(const std::string &attribute);
  //! \note Breaks indices, so spec is no longer unique after
  //! Returns the index part of the spec
  size_t index() const;
  //! Whether the spec is unique (i.e., contains an explicit index)
  bool isUnique() const;

private:

  // Data members ---

  std::string m_filename;
  std::string m_name;
  std::string m_attribute;
  size_t      m_index;

};

//----------------------------------------------------------------------------//
// iostream -- FilenameSpec
//----------------------------------------------------------------------------//

std::ostream&
operator << (std::ostream &os, const FilenameSpec &spec);

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard

