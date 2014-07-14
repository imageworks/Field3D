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

/*! \file SparseFieldIO.h
  \brief Contains the SparseFieldIO class.
  
  \todo Use boost::addressof instead of & operator
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_SparseFieldIO_H_
#define _INCLUDED_Field3D_SparseFieldIO_H_

//----------------------------------------------------------------------------//

#include <string>
#include <cmath>

#include <hdf5.h>

#include "OgIO.h"
#include "OgSparseDataReader.h"
#include "SparseDataReader.h"
#include "SparseField.h"
#include "SparseFile.h"
#include "FieldIO.h"
#include "Field3DFile.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// SparseFieldIO
//----------------------------------------------------------------------------//

/*! \class SparseFieldIO
  \ingroup file_int
   Defines the IO for a SparseField object
 */

//----------------------------------------------------------------------------//

class SparseFieldIO : public FieldIO 
{

public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<SparseFieldIO> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef SparseFieldIO class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  const char *staticClassType() const
  {
    return "SparseFieldIO";
  }
  
  // Constructors --------------------------------------------------------------

  //! Ctor
  SparseFieldIO() 
   : FieldIO()
  { }

  //! Dtor
  virtual ~SparseFieldIO() 
  { /* Empty */ }


  static FieldIO::Ptr create()
  { return Ptr(new SparseFieldIO); }

  // From FieldIO --------------------------------------------------------------

  //! Reads the field at the given location and tries to create a SparseField
  //! object from it.
  //! \returns Null if no object was read
  virtual FieldBase::Ptr read(hid_t layerGroup, const std::string &filename, 
                              const std::string &layerPath,
                              DataTypeEnum typeEnum);

  //! Reads the field at the given location and tries to create a SparseField
  //! object from it.
  //! \returns Null if no object was read
  virtual FieldBase::Ptr read(const OgIGroup &layerGroup, 
                              const std::string &filename, 
                              const std::string &layerPath,
                              OgDataType typeEnum);

  //! Writes the given field to disk. 
  //! \return true if successful, otherwise false
  virtual bool write(hid_t layerGroup, FieldBase::Ptr field);

  //! Writes the given field to disk. 
  //! \return true if successful, otherwise false
  virtual bool write(OgOGroup &layerGroup, FieldBase::Ptr field);

  //! Returns the class name
  virtual std::string className() const
  { return "SparseField"; }

private:

  // Internal methods ----------------------------------------------------------

  //! This call writes all the attributes and sets up the data space.
  template <class Data_T>
  bool writeInternal(hid_t layerGroup, typename SparseField<Data_T>::Ptr field);

  //! This call writes all the attributes and sets up the data space.
  template <class Data_T>
  bool writeInternal(OgOGroup &layerGroup, 
                     typename SparseField<Data_T>::Ptr field);

  //! Reads the data that is dependent on the data type on disk
  template <class Data_T>
  bool readData(hid_t location, 
                int numBlocks, 
                const std::string &filename, 
                const std::string &layerPath, 
                typename SparseField<Data_T>::Ptr result);

  template <class Data_T>
  typename SparseField<Data_T>::Ptr
  readData(const OgIGroup &location, const Box3i &extents, 
           const Box3i &dataWindow, const size_t blockOrder, 
           const size_t numBlocks, const std::string &filename, 
           const std::string &layerPath);

  // Strings -------------------------------------------------------------------

  static const int         k_versionNumber;
  static const std::string k_versionAttrName;
  static const std::string k_extentsStr;
  static const std::string k_extentsMinStr;
  static const std::string k_extentsMaxStr;
  static const std::string k_dataWindowStr;
  static const std::string k_dataWindowMinStr;
  static const std::string k_dataWindowMaxStr;
  static const std::string k_componentsStr;
  static const std::string k_blockOrderStr;
  static const std::string k_numBlocksStr;
  static const std::string k_blockResStr;
  static const std::string k_bitsPerComponentStr;
  static const std::string k_numOccupiedBlocksStr;
  static const std::string k_dataStr;
  static const std::string k_isCompressed;
  
  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef FieldIO base;  
};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
