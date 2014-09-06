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

/*! \file DenseFieldIO.h
  \brief Contains the DenseFieldIO class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_DenseFieldIO_H_
#define _INCLUDED_Field3D_DenseFieldIO_H_

//----------------------------------------------------------------------------//

#include <string>

#include <boost/intrusive_ptr.hpp>

#include <hdf5.h>

#include "DenseField.h"
#include "Exception.h"
#include "FieldIO.h"
#include "Field3DFile.h"
#include "Hdf5Util.h"
#include "OgIGroup.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// DenseFieldIO
//----------------------------------------------------------------------------//

/*! \class DenseFieldIO
  \ingroup file_int
  Handles IO for a DenseField object
*/

//----------------------------------------------------------------------------//

class DenseFieldIO : public FieldIO 
{

public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<DenseFieldIO> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef DenseFieldIO class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  static const char* staticClassType()
  {
    return "DenseFieldIO";
  }

  // Constructors --------------------------------------------------------------

  //! Ctor
  DenseFieldIO() 
   : FieldIO()
  { }

  //! Dtor
  virtual ~DenseFieldIO() 
  { /* Empty */ }

  static FieldIO::Ptr create()
  { return Ptr(new DenseFieldIO); }

  // From FieldIO --------------------------------------------------------------

  //! Reads the field at the given location and tries to create a DenseField
  //! object from it. Calls out to readData() for template-specific work.
  //! \returns Null if no object was read
  virtual FieldBase::Ptr read(hid_t layerGroup, const std::string &filename, 
                              const std::string &layerPath,
                              DataTypeEnum typeEnum);

  //! Reads the field at the given location and tries to create a DenseField
  //! object from it. Calls out to readData() for template-specific work.
  //! \returns Null if no object was read
  virtual FieldBase::Ptr read(const OgIGroup &layerGroup, 
                              const std::string &filename,
                              const std::string &layerPath,
                              OgDataType typeEnum);

  //! Writes the given field to disk. This function calls out to writeInternal
  //! once the template type has been determined.
  //! \return true if successful, otherwise false
  virtual bool write(hid_t layerGroup, FieldBase::Ptr field);

  //! Writes the given field to disk. This function calls out to writeInternal
  //! once the template type has been determined.
  //! \return true if successful, otherwise false
  virtual bool write(OgOGroup &layerGroup, FieldBase::Ptr field);

  //! Returns the class name
  virtual std::string className() const
  { return "DenseField"; }
  
private:

  // Internal methods ----------------------------------------------------------

  //! This call writes all the attributes and sets up the data space.
  template <class Data_T>
  bool writeInternal(hid_t layerGroup, 
                     typename DenseField<Data_T>::Ptr field);

  //! This call writes all the attributes and sets up the data space.
  template <class Data_T>
  bool writeInternal(OgOGroup &layerGroup, 
                     typename DenseField<Data_T>::Ptr field);

  //! This call performs the actual writing of data to disk. 
  template <class Data_T>
  bool writeData(hid_t dataSet, typename DenseField<Data_T>::Ptr field,
                 Data_T dummy);

  //! This call performs the actual reading of data from disk.
  template <class Data_T>
  typename DenseField<Data_T>::Ptr 
  readData(hid_t dataSet, const Box3i &extents, const Box3i &dataW);

  //! This call performs the actual reading of data from disk.
  template <class Data_T>
  typename DenseField<Data_T>::Ptr 
  readData(const OgIGroup &layerGroup, const Box3i &extents, 
           const Box3i &dataW);

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
  static const std::string k_bitsPerComponentStr;
  static const std::string k_dataStr;

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef FieldIO base;

};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
