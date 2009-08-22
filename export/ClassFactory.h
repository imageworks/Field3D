//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks
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

/*! \file ClassFactory.h
  \brief Contains the ClassFactory class for registering Field3D classes.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_ClassFactory_H_
#define _INCLUDED_Field3D_ClassFactory_H_

#include <map>
#include <vector>

#include "Field.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// ClassFactory
//----------------------------------------------------------------------------//

/*! \class ClassFactory
  \ingroup field_int
*/
  
//----------------------------------------------------------------------------//
  
class ClassFactory
{
    
public:
    
  // Typedefs ------------------------------------------------------------------

  typedef FieldRes::Ptr (*CreateFieldFnPtr) ();

  // Ctors, dtor ---------------------------------------------------------------

  //! Standard constructor
  ClassFactory();  

  // Main methods --------------------------------------------------------------

  //! Registers a class with the class pool. 
  //! \param createFunc Pointer to creation function
  void registerFieldClass(CreateFieldFnPtr createFunc);

  //! Instances an object by name
  FieldRes::Ptr createField(const std::string &className) const;

  //! Access point for the singleton instance.
  static ClassFactory& singleton();

private:
      
  // Typedefs ------------------------------------------------------------------

  typedef std::vector<std::string> SimpleNameVec;
  typedef std::map<std::string, CreateFieldFnPtr> FieldNameToFuncMap;

  // Data members --------------------------------------------------------------

  //! Map of create functions for Fields.  The key is the class name.
  FieldNameToFuncMap m_fields;
  //! 
  SimpleNameVec m_fieldSimpleNames;

  //! Pointer to static instance
  static ClassFactory *ms_instance;

};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
