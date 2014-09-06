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

/*! \file DenseFieldIO.cpp
  Containts implementation of the DenseFieldIO class
*/

//----------------------------------------------------------------------------//

#include "DenseFieldIO.h"

#include "OgIO.h"

//----------------------------------------------------------------------------//

using namespace boost;
using namespace std;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Field3D namespaces
//----------------------------------------------------------------------------//

using namespace Exc;
using namespace Hdf5Util;

//----------------------------------------------------------------------------//
// Static members
//----------------------------------------------------------------------------//

const int         DenseFieldIO::k_versionNumber(1);
const std::string DenseFieldIO::k_versionAttrName("version");
const std::string DenseFieldIO::k_extentsStr("extents");
const std::string DenseFieldIO::k_extentsMinStr("extents_min");
const std::string DenseFieldIO::k_extentsMaxStr("extents_max");
const std::string DenseFieldIO::k_dataWindowStr("data_window");
const std::string DenseFieldIO::k_dataWindowMinStr("data_window_min");
const std::string DenseFieldIO::k_dataWindowMaxStr("data_window_max");
const std::string DenseFieldIO::k_componentsStr("components");
const std::string DenseFieldIO::k_bitsPerComponentStr("bits_per_component");
const std::string DenseFieldIO::k_dataStr("data");

//----------------------------------------------------------------------------//

FieldBase::Ptr
DenseFieldIO::read(hid_t layerGroup, const std::string &/*filename*/, 
                   const std::string &/*layerPath*/,
                   DataTypeEnum typeEnum)
{
  Box3i extents, dataW;
  int components;
  hsize_t dims[1];
  
  if (layerGroup == -1)
    throw BadHdf5IdException("Bad layer group in DenseFieldIO::read");

  int version;
  if (!readAttribute(layerGroup, k_versionAttrName, 1, version))
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_versionAttrName);

  if (version != k_versionNumber)
    throw UnsupportedVersionException("DenseField version not supported: " + 
                                      lexical_cast<std::string>(version));

  if (!readAttribute(layerGroup, k_extentsStr, 6, extents.min.x)) 
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_extentsStr);

  if (!readAttribute(layerGroup, k_dataWindowStr, 6, dataW.min.x)) 
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_dataWindowStr);
  
  if (!readAttribute(layerGroup, k_componentsStr, 1, components)) 
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_componentsStr);

  H5ScopedDopen dataSet(layerGroup, k_dataStr, H5P_DEFAULT);

  if (dataSet.id() < 0) 
    throw OpenDataSetException("Couldn't open data set: " + k_dataStr);

  H5ScopedDget_space dataSpace(dataSet.id());
  H5ScopedDget_type dataType(dataSet.id());
  H5Sget_simple_extent_dims(dataSpace.id(), dims, NULL);

  if (dataSpace.id() < 0) 
    throw GetDataSpaceException("Couldn't get data space");

  if (dataType.id() < 0)
    throw GetDataTypeException("Couldn't get data type");

  // Double-check that the sizes match ---

  V3i size(dataW.size() + V3i(1));
  int calculatedTotal = size.x * size.y * size.z;
  int reportedSize = dims[0] / components;

  if (calculatedTotal != reportedSize) 
    throw FileIntegrityException("Data size doesn't match number of voxels");

  // Build a DenseField to store everything in
  FieldBase::Ptr result;

  // Read the data ---

  bool isHalf, isFloat, isDouble;
  isHalf = H5Tequal(dataType, H5T_NATIVE_SHORT);
  isFloat = H5Tequal(dataType, H5T_NATIVE_FLOAT);
  isDouble = H5Tequal(dataType, H5T_NATIVE_DOUBLE);

  if (isHalf && components == 1 && typeEnum == DataTypeHalf)
    result = readData<half>(dataSet.id(), extents, dataW);
  if (isFloat && components == 1 && typeEnum == DataTypeFloat)
    result = readData<float>(dataSet.id(), extents, dataW);
  if (isDouble && components == 1 && typeEnum == DataTypeDouble)
    result = readData<double>(dataSet.id(), extents, dataW);
  if (isHalf && components == 3 && typeEnum == DataTypeVecHalf)
    result = readData<V3h>(dataSet.id(), extents, dataW);
  if (isFloat && components == 3 && typeEnum == DataTypeVecFloat)
    result = readData<V3f>(dataSet.id(), extents, dataW);
  if (isDouble && components == 3 && typeEnum == DataTypeVecDouble)
    result = readData<V3d>(dataSet.id(), extents, dataW);

  return result;
}

//----------------------------------------------------------------------------//

FieldBase::Ptr
DenseFieldIO::read(const OgIGroup &lg, const std::string &/*filename*/, 
                   const std::string &/*layerPath*/, OgDataType typeEnum)
{
  Box3i extents, dataW;

  if (!lg.isValid()) {
    throw MissingGroupException("Invalid group in DenseFieldIO::read()");
  }

  // Check version ---

  OgIAttribute<int> versionAttr = lg.findAttribute<int>(k_versionAttrName);
  if (!versionAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_versionAttrName);
  }

  int version = versionAttr.value();
  if (version != k_versionNumber) {
    throw UnsupportedVersionException("DenseField version not supported: " + 
                                      lexical_cast<std::string>(version));
  }

  // Get extents ---

  OgIAttribute<veci32_t> extMinAttr = 
    lg.findAttribute<veci32_t>(k_extentsMinStr);
  OgIAttribute<veci32_t> extMaxAttr = 
    lg.findAttribute<veci32_t>(k_extentsMaxStr);
  if (!extMinAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_extentsMinStr);
  }
  if (!extMaxAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_extentsMaxStr);
  }

  extents.min = extMinAttr.value();
  extents.max = extMaxAttr.value();

  // Get data window ---

  OgIAttribute<veci32_t> dwMinAttr = 
    lg.findAttribute<veci32_t>(k_dataWindowMinStr);
  OgIAttribute<veci32_t> dwMaxAttr = 
    lg.findAttribute<veci32_t>(k_dataWindowMaxStr);
  if (!dwMinAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_dataWindowMinStr);
  }
  if (!dwMaxAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_dataWindowMaxStr);
  }

  dataW.min = dwMinAttr.value();
  dataW.max = dwMaxAttr.value();

  // Get num components ---

  OgIAttribute<int> numComponentsAttr = 
    lg.findAttribute<int>(k_componentsStr);
  if (!numComponentsAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_componentsStr);
  }

  // Read the data ---

  FieldBase::Ptr result;
  
  OgDataType typeOnDisk = lg.datasetType(k_dataStr);

  if (typeEnum == typeOnDisk) {
    if (typeEnum == F3DFloat16) {
      result = readData<float16_t>(lg, extents, dataW);
    } else if (typeEnum == F3DFloat32) {
      result = readData<float32_t>(lg, extents, dataW);
    } else if (typeEnum == F3DFloat64) {
      result = readData<float64_t>(lg, extents, dataW);
    } else if (typeEnum == F3DVec16) {
      result = readData<vec16_t>(lg, extents, dataW);
    } else if (typeEnum == F3DVec32) {
      result = readData<vec32_t>(lg, extents, dataW);
    } else if (typeEnum == F3DVec64) {
      result = readData<vec64_t>(lg, extents, dataW);
    } 
  }

  return result;

}

//----------------------------------------------------------------------------//

bool
DenseFieldIO::write(hid_t layerGroup, FieldBase::Ptr field)
{
  if (layerGroup == -1)
    throw BadHdf5IdException("Bad layer group in DenseFieldIO::write");

  // Add version attribute
  if (!writeAttribute(layerGroup, k_versionAttrName, 
                    1, k_versionNumber))
    throw WriteAttributeException("Couldn't write attribute " + 
                                  k_versionAttrName);

  DenseField<half>::Ptr halfField = 
    field_dynamic_cast<DenseField<half> >(field);
  DenseField<float>::Ptr floatField = 
    field_dynamic_cast<DenseField<float> >(field);
  DenseField<double>::Ptr doubleField = 
    field_dynamic_cast<DenseField<double> >(field);
  DenseField<V3h>::Ptr vecHalfField = 
    field_dynamic_cast<DenseField<V3h> >(field);
  DenseField<V3f>::Ptr vecFloatField = 
    field_dynamic_cast<DenseField<V3f> >(field);
  DenseField<V3d>::Ptr vecDoubleField = 
    field_dynamic_cast<DenseField<V3d> >(field);

  bool success = true;

  if (floatField) {
    success = writeInternal<float>(layerGroup, floatField);
  }
  else if (halfField) {
    success = writeInternal<half>(layerGroup, halfField);
  }
  else if (doubleField) {
    success = writeInternal<double>(layerGroup, doubleField);
  }
  else if (vecFloatField) {
    success = writeInternal<V3f>(layerGroup, vecFloatField);
  }
  else if (vecHalfField) {
    success = writeInternal<V3h>(layerGroup, vecHalfField);
  }
  else if (vecDoubleField) {
    success = writeInternal<V3d>(layerGroup, vecDoubleField);
  }
  else {
    throw WriteLayerException("DenseFieldIO does not support the given "
                              "DenseField template parameter");
  }

  return success;
}

//----------------------------------------------------------------------------//

bool
DenseFieldIO::write(OgOGroup &layerGroup, FieldBase::Ptr field)
{
  using namespace Exc;

  // Add version attribute
  OgOAttribute<int> version(layerGroup, k_versionAttrName, k_versionNumber);

  DenseField<half>::Ptr halfField = 
    field_dynamic_cast<DenseField<half> >(field);
  DenseField<float>::Ptr floatField = 
    field_dynamic_cast<DenseField<float> >(field);
  DenseField<double>::Ptr doubleField = 
    field_dynamic_cast<DenseField<double> >(field);
  DenseField<V3h>::Ptr vecHalfField = 
    field_dynamic_cast<DenseField<V3h> >(field);
  DenseField<V3f>::Ptr vecFloatField = 
    field_dynamic_cast<DenseField<V3f> >(field);
  DenseField<V3d>::Ptr vecDoubleField = 
    field_dynamic_cast<DenseField<V3d> >(field);

  bool success = true;

  if (floatField) {
    success = writeInternal<float>(layerGroup, floatField);
  }
  else if (halfField) {
    success = writeInternal<half>(layerGroup, halfField);
  }
  else if (doubleField) {
    success = writeInternal<double>(layerGroup, doubleField);
  }
  else if (vecFloatField) {
    success = writeInternal<V3f>(layerGroup, vecFloatField);
  }
  else if (vecHalfField) {
    success = writeInternal<V3h>(layerGroup, vecHalfField);
  }
  else if (vecDoubleField) {
    success = writeInternal<V3d>(layerGroup, vecDoubleField);
  }
  else {
    throw WriteLayerException("DenseFieldIO does not support the given "
                              "DenseField template parameter");
  }

  return success;
}

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

  // Add the bits per component attribute ---

  int bits = DataTypeTraits<Data_T>::h5bits();
  if (!writeAttribute(layerGroup, k_bitsPerComponentStr, 1, bits)) {
    Msg::print(Msg::SevWarning, "Error adding bits per component attribute.");
    return false;
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
bool DenseFieldIO::writeInternal(OgOGroup &layerGroup, 
                                 typename DenseField<Data_T>::Ptr field)
{
  using namespace Exc;

  const int    components  = FieldTraits<Data_T>::dataDims();
  const V3i&   memSize     = field->internalMemSize();
  const int    bits        = DataTypeTraits<Data_T>::h5bits();
  
  Box3i ext(field->extents()), dw(field->dataWindow());

  // Add extents attributes ---

  OgOAttribute<veci32_t> extMinAttr(layerGroup, k_extentsMinStr, ext.min);
  OgOAttribute<veci32_t> extMaxAttr(layerGroup, k_extentsMaxStr, ext.max);
  
  // Add data window attributes ---
  
  OgOAttribute<veci32_t> dwMinAttr(layerGroup, k_dataWindowMinStr, dw.min);
  OgOAttribute<veci32_t> dwMaxAttr(layerGroup, k_dataWindowMaxStr, dw.max);

  // Add components attribute ---

  OgOAttribute<int> componentsAttr(layerGroup, k_componentsStr, components);

  // Add the bits per component attribute ---

  OgOAttribute<int> bitsAttr(layerGroup, k_bitsPerComponentStr, bits);

  // Add data to file ---

  const size_t length = memSize[0] * memSize[1] * memSize[2];

  OgODataset<Data_T> data(layerGroup, k_dataStr);
  data.addData(length, &(*field->begin()));

  return true;
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool DenseFieldIO::writeData(hid_t dataSet, 
                             typename DenseField<Data_T>::Ptr field,
                             Data_T /* dummy */)
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

template <class Data_T>
typename DenseField<Data_T>::Ptr 
DenseFieldIO::readData(const OgIGroup &layerGroup, const Box3i &extents, 
                       const Box3i &dataW)
{
  typename DenseField<Data_T>::Ptr field(new DenseField<Data_T>);
  field->setSize(extents, dataW);

  // Open the dataset
  OgIDataset<Data_T> data = layerGroup.findDataset<Data_T>(k_dataStr);
  if (!data.isValid()) {
    throw Exc::ReadDataException("DenseFieldIO::readData() couldn't open "
                                 "the dataset.");
  }

  // Read the data
  if (!data.getData(0, &(*field->begin()), OGAWA_THREAD)) {
    throw Exc::ReadDataException("DenseFieldIO::readData() couldn't read "
                                 "the dataset.");
  }

  return field;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
