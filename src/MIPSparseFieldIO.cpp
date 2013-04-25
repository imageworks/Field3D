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

/*! \file MIPSparseFieldIO.cpp
  Containts implementation of the MIPSparseFieldIO class
*/

//----------------------------------------------------------------------------//

#include "MIPSparseFieldIO.h"

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
// Static member initialization
//----------------------------------------------------------------------------//

const int         MIPSparseFieldIO::k_versionNumber(1);
const std::string MIPSparseFieldIO::k_versionAttrName("version");
const std::string MIPSparseFieldIO::k_extentsStr("extents");
const std::string MIPSparseFieldIO::k_dataWindowStr("data_window");
const std::string MIPSparseFieldIO::k_componentsStr("components");
const std::string MIPSparseFieldIO::k_bitsPerComponentStr("bits_per_component");
const std::string MIPSparseFieldIO::k_mipGroupStr("mip_levels");
const std::string MIPSparseFieldIO::k_levelGroupStr("level");
const std::string MIPSparseFieldIO::k_levelsStr("levels");

//----------------------------------------------------------------------------//
// MIPSparseFieldIO
//----------------------------------------------------------------------------//

FieldBase::Ptr
MIPSparseFieldIO::read(hid_t layerGroup, const std::string &filename, 
                      const std::string &layerPath,
                      DataTypeEnum typeEnum)
{
  Box3i extents, dataW;
  int components;
  
  if (layerGroup == -1)
    throw BadHdf5IdException("Bad layer group in MIPSparseFieldIO::read");

  int version;
  if (!readAttribute(layerGroup, k_versionAttrName, 1, version))
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_versionAttrName);

  if (version != k_versionNumber)
    throw UnsupportedVersionException("MIPSparseField version not supported: " + 
                                      lexical_cast<std::string>(version));

  if (!readAttribute(layerGroup, k_componentsStr, 1, components)) 
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_componentsStr);

  // Check data type ---

  int bits;
  if (!readAttribute(layerGroup, k_bitsPerComponentStr, 1, bits)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_bitsPerComponentStr);  

  bool isHalf = false;
  bool isFloat = false;
  bool isDouble = false;

  switch (bits) {
  case 16:
    isHalf = true;
    break;
  case 64:
    isDouble = true;
    break;
  case 32:
  default:
    isFloat = true;
  }

  // Read the data ---

  FieldBase::Ptr result;

  if (isHalf && components == 1 && typeEnum == DataTypeHalf)
    result = readInternal<half>(layerGroup, filename, layerPath, typeEnum);
  if (isFloat && components == 1 && typeEnum == DataTypeFloat)
    result = readInternal<float>(layerGroup, filename, layerPath, typeEnum);
  if (isDouble && components == 1 && typeEnum == DataTypeDouble)
    result = readInternal<double>(layerGroup, filename, layerPath, typeEnum); 
  if (isHalf && components == 3 && typeEnum == DataTypeVecHalf)
    result = readInternal<V3h>(layerGroup, filename, layerPath, typeEnum);
  if (isFloat && components == 3 && typeEnum == DataTypeVecFloat)
    result = readInternal<V3f>(layerGroup, filename, layerPath, typeEnum);
  if (isDouble && components == 3 && typeEnum == DataTypeVecDouble)
    result = readInternal<V3d>(layerGroup, filename, layerPath, typeEnum);

  return result;
}

//----------------------------------------------------------------------------//

bool
MIPSparseFieldIO::write(hid_t layerGroup, FieldBase::Ptr field)
{
  if (layerGroup == -1) {
    throw BadHdf5IdException("Bad layer group in MIPSparseFieldIO::write");
  }

  // Add version attribute
  if (!writeAttribute(layerGroup, k_versionAttrName, 
                      1, k_versionNumber)) {
    throw WriteAttributeException("Couldn't write attribute " + 
                                  k_versionAttrName);
  }

  MIPSparseField<half>::Ptr halfField = 
    field_dynamic_cast<MIPSparseField<half> >(field);
  MIPSparseField<float>::Ptr floatField = 
    field_dynamic_cast<MIPSparseField<float> >(field);
  MIPSparseField<double>::Ptr doubleField = 
    field_dynamic_cast<MIPSparseField<double> >(field);
  MIPSparseField<V3h>::Ptr vecHalfField = 
    field_dynamic_cast<MIPSparseField<V3h> >(field);
  MIPSparseField<V3f>::Ptr vecFloatField = 
    field_dynamic_cast<MIPSparseField<V3f> >(field);
  MIPSparseField<V3d>::Ptr vecDoubleField = 
    field_dynamic_cast<MIPSparseField<V3d> >(field);

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
    throw WriteLayerException("MIPSparseFieldIO does not support the given "
                              "MIPSparseField template parameter");
  }

  return success;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
