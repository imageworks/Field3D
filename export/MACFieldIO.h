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

/*! \file MACFieldIO.h
  \brief Contains the MACFieldIO class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_MACFieldIO_H_
#define _INCLUDED_Field3D_MACFieldIO_H_

//----------------------------------------------------------------------------//

#include <string>

#include <boost/intrusive_ptr.hpp>

#include <hdf5.h>

#include "Exception.h"
#include "Field3DFile.h"
#include "FieldIO.h"
#include "Hdf5Util.h"
#include "MACField.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// MACFieldIO
//----------------------------------------------------------------------------//

/*! \class MACFieldIO
  \ingroup file_int
  Defines the IO for a MACField object
*/

//----------------------------------------------------------------------------//

class MACFieldIO : public FieldIO 
{

public:
  
  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<MACFieldIO> Ptr;

  // Constructors --------------------------------------------------------------

  //! Ctor
  MACFieldIO() 
   : FieldIO()
  { }

  //! Dtor
  virtual ~MACFieldIO() 
  { /* Empty */ }

  static FieldIO::Ptr create()
  { return Ptr(new MACFieldIO); }

  // From FieldIO --------------------------------------------------------------

  //! Reads the field at the given location and tries to create a MACField
  //! object from it.
  //! \returns Null if no object was read
  virtual FieldBase::Ptr read(hid_t layerGroup, const std::string &filename, 
                              const std::string &layerPath,
                              DataTypeEnum typeEnum);

  //! Writes the given field to disk. 
  //! \return true if successful, otherwise false
  virtual bool write(hid_t layerGroup, FieldBase::Ptr field);

  //! Returns the class name
  virtual std::string className() const
  { return std::string("MACField"); }

private:

  // Internal methods ----------------------------------------------------------

  //! This call writes all the attributes and sets up the data space.
  template <class Data_T>
  bool writeInternal(hid_t layerGroup, typename MACField<Data_T>::Ptr field);
  
  //! This call writes out the u,v,w data 
  template <class Data_T>
  bool writeData(hid_t layerGroup, typename MACField<Data_T>::Ptr field,
                 MACComponent comp);

  //! Reads the data that is dependent on the data type on disk
  template <class Data_T>
  bool readData(hid_t location, typename MACField<Data_T>::Ptr result);

  // Strings -------------------------------------------------------------------

  static const int         k_versionNumber;
  static const std::string k_versionAttrName;
  static const std::string k_extentsStr;
  static const std::string k_dataWindowStr;
  static const std::string k_componentsStr;
  static const std::string k_bitsPerComponentStr;
  static const std::string k_uDataStr;
  static const std::string k_vDataStr;
  static const std::string k_wDataStr;

};
 
//----------------------------------------------------------------------------//
// Template methods
//----------------------------------------------------------------------------//

//! \todo Tune the chunk size of the gzip call
template <class Data_T>
bool MACFieldIO::writeInternal(hid_t layerGroup, 
                                 typename MACField<Data_T>::Ptr field)
{
  using namespace Exc;
  using namespace Hdf5Util;

  int components = FieldTraits<Data_T>::dataDims();
  V3i compSize = field->getComponentSize();
  int size[3];
  size[0] = compSize.x;
  size[1] = compSize.y;
  size[2] = compSize.z;

  Box3i ext(field->extents()), dw(field->dataWindow());

  // Add extents attribute ---

  int extents[6] = 
    { ext.min.x, ext.min.y, ext.min.z, ext.max.x, ext.max.y, ext.max.z };

  if (!writeAttribute(layerGroup, k_extentsStr, 6, extents[0]))
    throw WriteAttributeException("Couldn't write attribute " + k_extentsStr);

  // Add data window attribute ---

  int dataWindow[6] = 
    { dw.min.x, dw.min.y, dw.min.z, dw.max.x, dw.max.y, dw.max.z };

  if (!writeAttribute(layerGroup, k_dataWindowStr, 6, dataWindow[0])) 
    throw WriteAttributeException("Couldn't write attribute " + k_dataWindowStr);

  // Add components attribute ---

  if (!writeAttribute(layerGroup, k_componentsStr, 1, components)) 
    throw WriteAttributeException("Couldn't write attribute " + k_componentsStr);

  // Add the bits per component attribute ---

  int bits = DataTypeTraits<Data_T>::h5bits();
  if (!writeAttribute(layerGroup, k_bitsPerComponentStr, 1, bits)) {
    throw WriteAttributeException("Couldn't write attribute " + k_bitsPerComponentStr);
    return false;    
  }

  // Add data to file ---
  if (!writeData<Data_T>(layerGroup, field, MACCompU)) {
    throw WriteMACFieldDataException("Error writing u_data");
    return false;
  }    
  if (!writeData<Data_T>(layerGroup, field, MACCompV)) {
    throw WriteMACFieldDataException("Error writing v_data");
    return false;
  }    
  if (!writeData<Data_T>(layerGroup, field, MACCompW)) {
    throw WriteMACFieldDataException("Error writing w_data");
    return false;
  }    

  return true; 
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool MACFieldIO::writeData(hid_t layerGroup, 
                           typename MACField<Data_T>::Ptr field,
                           MACComponent comp)
{
  using namespace Exc;
  using namespace Hdf5Util;

  const V3i &compSize = field->getComponentSize();

  hsize_t totalSize[1];
  std::string compStr;

  switch (comp) {
    case MACCompU:
      totalSize[0] = compSize.x;
      compStr = k_uDataStr;
        break;
    case MACCompV:
      totalSize[0] = compSize.y;
      compStr = k_vDataStr;
        break;
    case MACCompW:
      totalSize[0] = compSize.z;
      compStr = k_wDataStr;
        break;
    default:
      break;
  }    

  // Make sure chunk size isn't too big.
  hsize_t preferredChunkSize = 4096 * 16;
  const hsize_t chunkSize = std::min(preferredChunkSize, totalSize[0] / 2);

  H5ScopedScreate dataSpace(H5S_SIMPLE);

  if (dataSpace.id() < 0) 
    throw CreateDataSpaceException("Couldn't create data space in "
                                   "MACFieldIO::writeData");

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
  
  H5ScopedDcreate dataSet(layerGroup, compStr, 
                          DataTypeTraits<Data_T>::h5type(), 
                          dataSpace.id(), 
                          H5P_DEFAULT, dcpl, H5P_DEFAULT);

  if (dataSet.id() < 0) 
    throw CreateDataSetException("Couldn't create data set in "
                                 "MACFieldIO::writeData");

  hid_t err = H5Dwrite(dataSet, 
                       DataTypeTraits<Data_T>::h5type(), 
                       H5S_ALL, H5S_ALL, 
                       H5P_DEFAULT, &(*field->cbegin_comp(comp)));
  if (err < 0) 
    throw Exc::WriteLayerException("Error writing layer in "
                                   "MACFieldIO::writeData");


  return true;
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool MACFieldIO::readData(hid_t layerGroup,
                          typename MACField<Data_T>::Ptr field)
{
  using namespace std;
  using namespace Exc;
  using namespace Hdf5Util;

  hsize_t dims[1];

  // read u_data
  {

    H5ScopedDopen dataSet(layerGroup, k_uDataStr, H5P_DEFAULT);
    if (dataSet.id() < 0) 
      throw OpenDataSetException("Couldn't open data set: " + k_uDataStr);

    H5ScopedDget_space dataSpace(dataSet.id());
    H5ScopedDget_type dataType(dataSet.id());
    H5Sget_simple_extent_dims(dataSpace.id(), dims, NULL);

    if (dataSpace.id() < 0) 
      throw GetDataSpaceException("Couldn't get data space");

    if (dataType.id() < 0)
      throw GetDataTypeException("Couldn't get data type");

    if (H5Dread(dataSet, DataTypeTraits<Data_T>::h5type(), 
                H5S_ALL, H5S_ALL, H5P_DEFAULT, &(*field->begin_comp(MACCompU))) < 0) 
      {
        std::string typeName = "MACField<" + 
          DataTypeTraits<Data_T>::name() + ">";
        throw Exc::Hdf5DataReadException("Couldn't read " + typeName + " data");
      } 

  }

  // read v_data
  {

    H5ScopedDopen dataSet(layerGroup, k_vDataStr, H5P_DEFAULT);
    if (dataSet.id() < 0) 
      throw OpenDataSetException("Couldn't open data set: " + k_vDataStr);

    H5ScopedDget_space dataSpace(dataSet.id());
    H5ScopedDget_type dataType(dataSet.id());
    H5Sget_simple_extent_dims(dataSpace.id(), dims, NULL);

    if (dataSpace.id() < 0) 
      throw GetDataSpaceException("Couldn't get data space");

    if (dataType.id() < 0)
      throw GetDataTypeException("Couldn't get data type");


    if (H5Dread(dataSet, DataTypeTraits<Data_T>::h5type(), 
                H5S_ALL, H5S_ALL, H5P_DEFAULT, &(*field->begin_comp(MACCompV))) < 0) 
      {
        std::string typeName = "MACField<" + 
          DataTypeTraits<Data_T>::name() + ">";
        throw Exc::Hdf5DataReadException("Couldn't read " + typeName + " data");
      } 

  }

  // read w_data
  {

    H5ScopedDopen dataSet(layerGroup, k_wDataStr, H5P_DEFAULT);
    if (dataSet.id() < 0) 
      throw OpenDataSetException("Couldn't open data set: " + k_wDataStr);

    H5ScopedDget_space dataSpace(dataSet.id());
    H5ScopedDget_type dataType(dataSet.id());
    H5Sget_simple_extent_dims(dataSpace.id(), dims, NULL);

    if (dataSpace.id() < 0) 
      throw GetDataSpaceException("Couldn't get data space");

    if (dataType.id() < 0)
      throw GetDataTypeException("Couldn't get data type");


    if (H5Dread(dataSet, DataTypeTraits<Data_T>::h5type(), 
                H5S_ALL, H5S_ALL, H5P_DEFAULT, &(*field->begin_comp(MACCompW))) < 0) 
      {
        std::string typeName = "MACField<" + 
          DataTypeTraits<Data_T>::name() + ">";
        throw Exc::Hdf5DataReadException("Couldn't read " + typeName + " data");
      } 

  }

  return true;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
