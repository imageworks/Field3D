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

/*! \file FieldMappingFactory.h
  \ingroup field
  \brief Contains the FieldMappingFactory class
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_FieldMappingFactory_H_
#define _INCLUDED_Field3D_FieldMappingFactory_H_

#include <hdf5.h>

#include "FieldMapping.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// FieldMappingFactory
//----------------------------------------------------------------------------//

/*! \class FieldMappingFactory
  \ingroup file_int
  Responsible for writing and reading FieldMapping object to and from Hdf5 
  files.
  \todo Merge this into ClassFactory
  \note This is a singleton object
*/

class FieldMappingFactory
{
public:
  
  // Main methods --------------------------------------------------------------

  //! Writes the mapping to the specified Hdf5 location
  bool write(hid_t loc, FieldMapping::Ptr mapping);

  //! Reads and constructs a mapping object based on the given Hdf5 group
  FieldMapping::Ptr read(hid_t loc);

  //! Returns the singleton instance. Don't worry about the long name,
  //! there's a macro to help out
  static FieldMappingFactory& theFieldMappingFactoryInstance();

private:

  // Specialied methods for writing each mapping type --------------------------
  
  //! Writes a NullFieldMapping
  bool writeNullMapping(hid_t mappingGroup, NullFieldMapping::Ptr nm);
  //! Reads a NullFieldMapping
  NullFieldMapping::Ptr readNullMapping(hid_t mappingGroup);

  //! Writes a MatrixFieldMapping
  bool writeMatrixMapping(hid_t mappingGroup, MatrixFieldMapping::Ptr nm);
  //! Reads a MatrixFieldMapping
  MatrixFieldMapping::Ptr readMatrixMapping(hid_t mappingGroup);

  //! Private to prevent instantiation
  FieldMappingFactory() 
  { }

  //! Singleton instance
  static FieldMappingFactory* ms_theFieldMappingFactory;
};

//----------------------------------------------------------------------------//

//! Convenience macro
#define theFieldMappingFactory \
  (FieldMappingFactory::theFieldMappingFactoryInstance())

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
