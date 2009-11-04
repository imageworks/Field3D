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

/*! \file FieldMappingIO.h
  \ingroup field
  \brief Contains the FieldMappingIO base class and the NullFieldMappingIO and
  MatrixFieldMappingIO subclasses.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_FieldMappingIO_H_
#define _INCLUDED_Field3D_FieldMappingIO_H_

//----------------------------------------------------------------------------//

#include <hdf5.h>

#include "FieldMapping.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//

class FieldMappingIO : public RefBase
{

public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::intrusive_ptr<FieldMappingIO> Ptr;

  // Ctors, dtor ---------------------------------------------------------------

  //! Ctor
  FieldMappingIO() : RefBase() {}
  
  //! Dtor
  virtual ~FieldMappingIO() {}

  // Methods to be implemented by subclasses -----------------------------------

  //! Read the field at the given hdf5 group
  //! \returns Pointer to the created field, or a null pointer if the field
  //! couldn't be read.
  virtual FieldMapping::Ptr read(hid_t mappingGroup) = 0;

  //! Write the field to the given mapping group
  //! \returns Whether the operation was successful
  virtual bool write(hid_t mappingGroup, FieldMapping::Ptr mapping) = 0;

  //! Returns the class name. This is used when registering the class in the
  //! ClassFactory.
  virtual std::string className() const = 0;

 private:


};

//----------------------------------------------------------------------------//
// NullFieldMappingIO
//----------------------------------------------------------------------------//

/*! \class NullFieldMappingIO
  \ingroup file_int
  Handles IO for a NullFieldMapping object
*/

//----------------------------------------------------------------------------//

class NullFieldMappingIO : public FieldMappingIO 
{

public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<NullFieldMappingIO> Ptr;

  // Constructors --------------------------------------------------------------

  //! Ctor
  NullFieldMappingIO() 
   : FieldMappingIO()
  { }

  static FieldMappingIO::Ptr create()
  { return Ptr(new NullFieldMappingIO); }

  // From FieldMappingIO -------------------------------------------------------

  //! Reads the field mapping and tries to create a NullFieldMapping
  //! object from it.
  //! \returns Null if no object was read
  virtual FieldMapping::Ptr read(hid_t mappingGroup);

  //! Writes the given field mapping to disk.
  //! \return true if successful, otherwise false
  virtual bool write(hid_t mappingGroup, FieldMapping::Ptr mapping);

  //! Returns the class name
  virtual std::string className() const;

private:

};

//----------------------------------------------------------------------------//
// MatrixFieldMappingIO
//----------------------------------------------------------------------------//

/*! \class MatrixFieldMappingIO
  \ingroup file_int
  Handles IO for a MatrixFieldMapping object
*/

//----------------------------------------------------------------------------//

class MatrixFieldMappingIO : public FieldMappingIO 
{

public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<MatrixFieldMappingIO> Ptr;

  // Constructors --------------------------------------------------------------

  //! Ctor
  MatrixFieldMappingIO() 
   : FieldMappingIO()
  { }

  static FieldMappingIO::Ptr create()
  { return Ptr(new MatrixFieldMappingIO); }

  // From FieldMappingIO -------------------------------------------------------

  //! Reads the field mapping and tries to create a MatrixFieldMapping
  //! object from it.
  //! \returns Matrix if no object was read
  virtual FieldMapping::Ptr read(hid_t mappingGroup);

  //! Writes the given field mapping to disk.
  //! \return true if successful, otherwise false
  virtual bool write(hid_t mappingGroup, FieldMapping::Ptr mapping);

  //! Returns the class name
  virtual std::string className() const;

private:

};

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
