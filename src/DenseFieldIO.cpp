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
const std::string DenseFieldIO::k_dataWindowStr("data_window");
const std::string DenseFieldIO::k_componentsStr("components");
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

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
