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

  //! Writes the given field to disk. This function calls out to writeInternal
  //! once the template type has been determined.
  //! \return true if successful, otherwise false
  virtual bool write(hid_t layerGroup, FieldBase::Ptr field);

  //! Returns the class name
  virtual std::string className() const
  { return std::string("DenseField"); }

private:

  // Internal methods ----------------------------------------------------------

  //! This call writes all the attributes and sets up the data space.
  template <class Data_T>
  bool writeInternal(hid_t layerGroup, 
                     typename DenseField<Data_T>::Ptr field);

  //! This call performs the actual writing of data to disk. 
  template <class Data_T>
  bool writeData(hid_t dataSet, typename DenseField<Data_T>::Ptr field,
                 Data_T dummy);

  //! This call performs the actual reading of data from disk.
  template <class Data_T>
  typename DenseField<Data_T>::Ptr readData(hid_t dataSet, const Box3i &extents,
                                            const Box3i &dataW);

  // Strings -------------------------------------------------------------------

  static const int         k_versionNumber;
  static const std::string k_versionAttrName;
  static const std::string k_extentsStr;
  static const std::string k_dataWindowStr;
  static const std::string k_componentsStr;
  static const std::string k_dataStr;

};

//----------------------------------------------------------------------------//
// Templated methods
//----------------------------------------------------------------------------//

//! \todo Tune the chunk size of the gzip call
template <class Data_T>
bool DenseFieldIO::writeInternal(hid_t layerGroup, 
                                 typename DenseField<Data_T>::Ptr field)
{
  using namespace Exc;
  using namespace Hdf5Util;

  const V3i& memSize = field->internalMemSize();
  int size[3];
  size[0] = memSize.x;
  size[1] = memSize.y;
  size[2] = memSize.z;

  int components = FieldTraits<Data_T>::dataDims();

  hsize_t totalSize[1];
  totalSize[0] = size[0] * size[1] * size[2] * components;

  // Make sure chunk size isn't too big.
  hsize_t preferredChunkSize = 4096 * 16;
  const hsize_t chunkSize = std::min(preferredChunkSize, totalSize[0] / 2);

  Box3i ext(field->extents()), dw(field->dataWindow());

  // Add extents attribute ---

  int extents[6] = 
    { ext.min.x, ext.min.y, ext.min.z, ext.max.x, ext.max.y, ext.max.z };

  if (!writeAttribute(layerGroup, k_extentsStr, 6, extents[0])) {
    throw WriteAttributeException("Couldn't write attribute " + k_extentsStr);
  }

  // Add data window attribute ---

  int dataWindow[6] = 
    { dw.min.x, dw.min.y, dw.min.z, dw.max.x, dw.max.y, dw.max.z };

  if (!writeAttribute(layerGroup, k_dataWindowStr, 6, dataWindow[0])) {
    throw WriteAttributeException("Couldn't write attribute " + k_dataWindowStr);
  }

  // Add components attribute ---

  if (!writeAttribute(layerGroup, k_componentsStr, 1, components)) {
    throw WriteAttributeException("Couldn't write attribute " + k_componentsStr);
  }

  // Add data to file ---

  H5ScopedScreate dataSpace(H5S_SIMPLE);

  if (dataSpace.id() < 0) {
    throw CreateDataSpaceException("Couldn't create data space in "
                                   "DenseFieldIO::writeInternal");
  }

  // Create a "simple" data structure ---

  H5Sset_extent_simple(dataSpace.id(), 1, totalSize, NULL);

  // Set up gzip property list
  bool gzipAvailable = checkHdf5Gzip();
  hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  if (gzipAvailable) {
    herr_t status = H5Pset_deflate(dcpl, 9);
    if (status < 0) {
      return false;
    }
    status = H5Pset_chunk(dcpl, 1, &chunkSize);
    if (status < 0) {
      return false;
    }    
  }
  
  H5ScopedDcreate dataSet(layerGroup, k_dataStr, 
                          DataTypeTraits<Data_T>::h5type(), 
                          dataSpace.id(), 
                          H5P_DEFAULT, dcpl, H5P_DEFAULT);

  if (dataSet.id() < 0) {
    throw CreateDataSetException("Couldn't create data set in "
                                 "DenseFieldIO::writeInternal");
  }

  // Call out to the templated function, it will figure out how to get
  // the data into the file in the appropriate fashion.
  if (!writeData<Data_T>(dataSet.id(), field, Data_T(0.0f))) {
    throw WriteLayerException("Error writing layer");
  }

  return true; 
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool DenseFieldIO::writeData(hid_t dataSet, 
                             typename DenseField<Data_T>::Ptr field,
                             Data_T dummy)
{ 
  using namespace Hdf5Util;

  hid_t err = H5Dwrite(dataSet, 
                       DataTypeTraits<Data_T>::h5type(), 
                       H5S_ALL, H5S_ALL, 
                       H5P_DEFAULT, &(*field->begin()));

  if (err < 0) {
    throw Exc::WriteLayerException("Error writing layer in "
                                   "DenseFieldIO::writeData");
  }

  return true;
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename DenseField<Data_T>::Ptr 
DenseFieldIO::readData(hid_t dataSet, const Box3i &extents, const Box3i &dataW)
{
  typename DenseField<Data_T>::Ptr field(new DenseField<Data_T>);
  field->setSize(extents, dataW);

  if (H5Dread(dataSet, DataTypeTraits<Data_T>::h5type(), 
              H5S_ALL, H5S_ALL, H5P_DEFAULT, &(*field->begin())) < 0) 
  {
    std::string typeName = "DenseField<" + 
      DataTypeTraits<Data_T>::name() + ">";
    throw Exc::Hdf5DataReadException("Couldn't read " + typeName + " data");
  } 

  return field;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
