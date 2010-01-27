//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks Inc
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
#include "FieldIO.h"
#include "FieldMappingIO.h"

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
  typedef FieldIO::Ptr (*CreateFieldIOFnPtr) ();
  typedef FieldMapping::Ptr (*CreateFieldMappingFnPtr) ();
  typedef FieldMappingIO::Ptr (*CreateFieldMappingIOFnPtr) ();

  // Ctors, dtor ---------------------------------------------------------------

  //! Standard constructor
  ClassFactory();  

  // Main methods --------------------------------------------------------------

  //! \name Field class 
  //! \{

  //! Registers a class with the class pool. 
  //! \param createFunc Pointer to creation function
  void registerField(CreateFieldFnPtr createFunc);

  //! Instances an object by name
  FieldRes::Ptr createField(const std::string &className) const;

  //! Registers an IO class with the class pool. 
  //! \param createFunc Pointer to creation function
  void registerFieldIO(CreateFieldIOFnPtr createFunc);

  //! Instances an IO object by name
  FieldIO::Ptr createFieldIO(const std::string &className) const;

  //! }

  //! \name FieldMapping class 
  //! \{

  //! Registers a class with the class pool. 
  //! \param createFunc Pointer to creation function
  void registerFieldMapping(CreateFieldMappingFnPtr createFunc);

  //! Instances an object by name
  FieldMapping::Ptr createFieldMapping(const std::string &className) const;

  //! Registers an IO class with the class pool. 
  //! \param createFunc Pointer to creation function
  void registerFieldMappingIO(CreateFieldMappingIOFnPtr createFunc);

  //! Instances an IO object by name
  FieldMappingIO::Ptr createFieldMappingIO(const std::string &className) const;

  //! }


  //! Access point for the singleton instance.
  static ClassFactory& singleton();

private:
      
  // Typedefs ------------------------------------------------------------------

  typedef std::vector<std::string> NameVec;
  typedef std::map<std::string, CreateFieldFnPtr> FieldFuncMap;
  typedef std::map<std::string, CreateFieldIOFnPtr> FieldIOFuncMap;
  typedef std::map<std::string, CreateFieldMappingFnPtr> FieldMappingFuncMap;
  typedef std::map<std::string, CreateFieldMappingIOFnPtr> FieldMappingIOFuncMap;

  // Data members --------------------------------------------------------------

  //! Map of create functions for Fields.  The key is the class name.
  FieldFuncMap m_fields;
  //! 
  NameVec m_fieldNames;

  //! Map of create functions for FieldIO classes.  The key is the class name.
  FieldIOFuncMap m_fieldIOs;
  //! 
  NameVec m_fieldIONames;

  //! Map of create functions for FieldMappings.  The key is the class name.
  FieldMappingFuncMap m_mappings;
  //! 
  NameVec m_fieldMappingNames;


  //! Map of create functions for FieldMapping IO classes.  The key is the class name.
  FieldMappingIOFuncMap m_mappingIOs;
  //! 
  NameVec m_fieldMappingIONames;


  //! Pointer to static instance
  static ClassFactory *ms_instance;

};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
