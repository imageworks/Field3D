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

/*! \file MIPFieldIO.cpp
  Containts implementation of the MIPFieldIO class
*/

//----------------------------------------------------------------------------//

#include "MIPFieldIO.h"

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

const int         MIPFieldIO::k_versionNumber      (1);
const std::string MIPFieldIO::k_versionAttrName    ("version");
const std::string MIPFieldIO::k_extentsStr         ("extents");
const std::string MIPFieldIO::k_extentsMinStr      ("extents_min");
const std::string MIPFieldIO::k_extentsMaxStr      ("extents_max");
const std::string MIPFieldIO::k_dataWindowStr      ("data_window");
const std::string MIPFieldIO::k_dataWindowMinStr   ("data_window_min");
const std::string MIPFieldIO::k_dataWindowMaxStr   ("data_window_max");
const std::string MIPFieldIO::k_componentsStr      ("components");
const std::string MIPFieldIO::k_bitsPerComponentStr("bits_per_component");
const std::string MIPFieldIO::k_mipGroupStr        ("mip_levels");
const std::string MIPFieldIO::k_levelGroupStr      ("level");
const std::string MIPFieldIO::k_levelsStr          ("levels");
const std::string MIPFieldIO::k_baseTypeStr        ("base_type");
const std::string MIPFieldIO::k_dummyDataStr       ("dummy_data");

//----------------------------------------------------------------------------//
// MIPFieldIO
//----------------------------------------------------------------------------//

FieldBase::Ptr
MIPFieldIO::read(hid_t layerGroup, const std::string &filename, 
                 const std::string &layerPath,
                 DataTypeEnum typeEnum)
{
  Box3i extents, dataW;
  int components;
  
  if (layerGroup == -1)
    throw BadHdf5IdException("Bad layer group in MIPFieldIO::read");

  int version;
  if (!readAttribute(layerGroup, k_versionAttrName, 1, version))
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_versionAttrName);

  if (version != k_versionNumber)
    throw UnsupportedVersionException("MIPField version not supported: " + 
                                      lexical_cast<std::string>(version));

  if (!readAttribute(layerGroup, k_componentsStr, 1, components)) 
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_componentsStr);

  // Check data type ---

  int bits;
  if (!readAttribute(layerGroup, k_bitsPerComponentStr, 1, bits)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_bitsPerComponentStr);  

  std::string baseType;
  if (!readAttribute(layerGroup, k_baseTypeStr, baseType)) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_baseTypeStr);
  }

  bool isHalf   = false;
  bool isFloat  = false;
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

  bool isSparse = false;
  bool isDense  = false;

  if (baseType == "SparseField") {
    isSparse = true;
  } else if (baseType == "DenseField") {
    isDense = true;
  }

  // Read the data ---

  FieldBase::Ptr result;

  if (isDense && isHalf && components == 1 && typeEnum == DataTypeHalf)
    result = readInternal<DenseField, half>(layerGroup, filename, 
                                            layerPath, typeEnum);
  if (isDense && isFloat && components == 1 && typeEnum == DataTypeFloat)
    result = readInternal<DenseField, float>(layerGroup, filename, 
                                             layerPath, typeEnum);
  if (isDense && isDouble && components == 1 && typeEnum == DataTypeDouble)
    result = readInternal<DenseField, double>(layerGroup, filename, 
                                              layerPath, typeEnum); 
  if (isDense && isHalf && components == 3 && typeEnum == DataTypeVecHalf)
    result = readInternal<DenseField, V3h>(layerGroup, filename, 
                                           layerPath, typeEnum);
  if (isDense && isFloat && components == 3 && typeEnum == DataTypeVecFloat)
    result = readInternal<DenseField, V3f>(layerGroup, filename, 
                                           layerPath, typeEnum);
  if (isDense && isDouble && components == 3 && typeEnum == DataTypeVecDouble)
    result = readInternal<DenseField, V3d>(layerGroup, filename, 
                                           layerPath, typeEnum);
  if (isSparse && isHalf && components == 1 && typeEnum == DataTypeHalf)
    result = readInternal<SparseField, half>(layerGroup, filename, 
                                            layerPath, typeEnum);
  if (isSparse && isFloat && components == 1 && typeEnum == DataTypeFloat)
    result = readInternal<SparseField, float>(layerGroup, filename, 
                                             layerPath, typeEnum);
  if (isSparse && isDouble && components == 1 && typeEnum == DataTypeDouble)
    result = readInternal<SparseField, double>(layerGroup, filename, 
                                              layerPath, typeEnum); 
  if (isSparse && isHalf && components == 3 && typeEnum == DataTypeVecHalf)
    result = readInternal<SparseField, V3h>(layerGroup, filename, 
                                           layerPath, typeEnum);
  if (isSparse && isFloat && components == 3 && typeEnum == DataTypeVecFloat)
    result = readInternal<SparseField, V3f>(layerGroup, filename, 
                                           layerPath, typeEnum);
  if (isSparse && isDouble && components == 3 && typeEnum == DataTypeVecDouble)
    result = readInternal<SparseField, V3d>(layerGroup, filename, 
                                           layerPath, typeEnum);

  return result;
}

//----------------------------------------------------------------------------//

FieldBase::Ptr
MIPFieldIO::read(const OgIGroup &layerGroup, const std::string &filename, 
                 const std::string &layerPath, OgDataType typeEnum)
{
  Box3i extents, dataW;

  if (!layerGroup.isValid()) {
    Msg::print(Msg::SevWarning, "Bad layerGroup group in "
               "MIPFieldIO::read(ogawa).");
    return FieldBase::Ptr();
  }

  // Check version ---

  OgIAttribute<int> versionAttr = 
    layerGroup.findAttribute<int>(k_versionAttrName);
  if (!versionAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_versionAttrName);
  }
  const int version = versionAttr.value();

  if (version != k_versionNumber) {
    throw UnsupportedVersionException("MIPField version not supported: " +
                                      lexical_cast<std::string>(version));
  }

  // Get num components ---

  OgIAttribute<uint8_t> numComponentsAttr = 
    layerGroup.findAttribute<uint8_t>(k_componentsStr);
  if (!numComponentsAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_componentsStr);
  }

  // Get base type ---

  OgIAttribute<std::string> baseTypeAttr = 
    layerGroup.findAttribute<std::string>(k_baseTypeStr);
  if (!baseTypeAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_baseTypeStr);
  }

  bool isSparse = false;
  bool isDense  = false;

  if (baseTypeAttr.value() == "SparseField") {
    isSparse = true;
  } else if (baseTypeAttr.value() == "DenseField") {
    isDense = true;
  }

  // Get the deta type ---

  OgDataType typeOnDisk = layerGroup.datasetType(k_dummyDataStr);

  FieldBase::Ptr result;

  if (typeEnum == typeOnDisk) {
    if (isDense) {
      if (typeEnum == F3DFloat16) {
        result = readInternal<DenseField, float16_t>(layerGroup, filename, 
                                                     layerPath, typeEnum);
      } else if (typeEnum == F3DFloat32) {
        result = readInternal<DenseField, float32_t>(layerGroup, filename, 
                                                     layerPath, typeEnum);
      } else if (typeEnum == F3DFloat64) {
        result = readInternal<DenseField, float64_t>(layerGroup, filename, 
                                                     layerPath, typeEnum);
      } else if (typeEnum == F3DVec16) {
        result = readInternal<DenseField, vec16_t>(layerGroup, filename, 
                                                   layerPath, typeEnum);
      } else if (typeEnum == F3DVec32) {
        result = readInternal<DenseField, vec32_t>(layerGroup, filename, 
                                                   layerPath, typeEnum);
      } else if (typeEnum == F3DVec64) {
        result = readInternal<DenseField, vec64_t>(layerGroup, filename, 
                                                   layerPath, typeEnum);
      }
    } else if (isSparse) {
      if (typeEnum == F3DFloat16) {
        result = readInternal<SparseField, float16_t>(layerGroup, filename, 
                                                      layerPath, typeEnum);
      } else if (typeEnum == F3DFloat32) {
        result = readInternal<SparseField, float32_t>(layerGroup, filename, 
                                                      layerPath, typeEnum);
      } else if (typeEnum == F3DFloat64) {
        result = readInternal<SparseField, float64_t>(layerGroup, filename, 
                                                      layerPath, typeEnum);
      } else if (typeEnum == F3DVec16) {
        result = readInternal<SparseField, vec16_t>(layerGroup, filename, 
                                                    layerPath, typeEnum);
      } else if (typeEnum == F3DVec32) {
        result = readInternal<SparseField, vec32_t>(layerGroup, filename, 
                                                    layerPath, typeEnum);
      } else if (typeEnum == F3DVec64) {
        result = readInternal<SparseField, vec64_t>(layerGroup, filename, 
                                                    layerPath, typeEnum);
      }
    } 
  }

  return result;
}

//----------------------------------------------------------------------------//

bool
MIPFieldIO::write(hid_t layerGroup, FieldBase::Ptr field)
{
  if (layerGroup == -1) {
    throw BadHdf5IdException("Bad layer group in MIPFieldIO::write");
  }

  // Add version attribute
  if (!writeAttribute(layerGroup, k_versionAttrName, 
                      1, k_versionNumber)) {
    throw WriteAttributeException("Couldn't write attribute " + 
                                  k_versionAttrName);
  }

  MIPField<DenseField<half> >::Ptr halfDenseField = 
    field_dynamic_cast<MIPField<DenseField<half> > >(field);
  MIPField<DenseField<float> >::Ptr floatDenseField = 
    field_dynamic_cast<MIPField<DenseField<float> > >(field);
  MIPField<DenseField<double> >::Ptr doubleDenseField = 
    field_dynamic_cast<MIPField<DenseField<double> > >(field);
  MIPField<DenseField<V3h> >::Ptr vecHalfDenseField = 
    field_dynamic_cast<MIPField<DenseField<V3h> > >(field);
  MIPField<DenseField<V3f> >::Ptr vecFloatDenseField = 
    field_dynamic_cast<MIPField<DenseField<V3f> > >(field);
  MIPField<DenseField<V3d> >::Ptr vecDoubleDenseField = 
    field_dynamic_cast<MIPField<DenseField<V3d> > >(field);
  MIPField<SparseField<half> >::Ptr halfSparseField = 
    field_dynamic_cast<MIPField<SparseField<half> > >(field);
  MIPField<SparseField<float> >::Ptr floatSparseField = 
    field_dynamic_cast<MIPField<SparseField<float> > >(field);
  MIPField<SparseField<double> >::Ptr doubleSparseField = 
    field_dynamic_cast<MIPField<SparseField<double> > >(field);
  MIPField<SparseField<V3h> >::Ptr vecHalfSparseField = 
    field_dynamic_cast<MIPField<SparseField<V3h> > >(field);
  MIPField<SparseField<V3f> >::Ptr vecFloatSparseField = 
    field_dynamic_cast<MIPField<SparseField<V3f> > >(field);
  MIPField<SparseField<V3d> >::Ptr vecDoubleSparseField = 
    field_dynamic_cast<MIPField<SparseField<V3d> > >(field);

  bool success = true;

  if (floatDenseField) {
    success = writeInternal<DenseField, float>(layerGroup, floatDenseField);
  }
  else if (halfDenseField) {
    success = writeInternal<DenseField, half>(layerGroup, halfDenseField);
  }
  else if (doubleDenseField) {
    success = writeInternal<DenseField, double>(layerGroup, doubleDenseField);
  }
  else if (vecFloatDenseField) {
    success = writeInternal<DenseField, V3f>(layerGroup, vecFloatDenseField);
  }
  else if (vecHalfDenseField) {
    success = writeInternal<DenseField, V3h>(layerGroup, vecHalfDenseField);
  }
  else if (vecDoubleDenseField) {
    success = writeInternal<DenseField, V3d>(layerGroup, vecDoubleDenseField);
  }
  else if (floatSparseField) {
    success = writeInternal<SparseField, float>(layerGroup, floatSparseField);
  }
  else if (halfSparseField) {
    success = writeInternal<SparseField, half>(layerGroup, halfSparseField);
  }
  else if (doubleSparseField) {
    success = writeInternal<SparseField, double>(layerGroup, doubleSparseField);
  }
  else if (vecFloatSparseField) {
    success = writeInternal<SparseField, V3f>(layerGroup, vecFloatSparseField);
  }
  else if (vecHalfSparseField) {
    success = writeInternal<SparseField, V3h>(layerGroup, vecHalfSparseField);
  }
  else if (vecDoubleSparseField) {
    success = writeInternal<SparseField, V3d>(layerGroup, vecDoubleSparseField);
  }
  else {
    throw WriteLayerException("MIPFieldIO does not support the given "
                              "MIPField template parameter");
  }

  return success;
}

//----------------------------------------------------------------------------//

bool
MIPFieldIO::write(OgOGroup &layerGroup, FieldBase::Ptr field)
{
  // Add version attribute
  OgOAttribute<int> version(layerGroup, k_versionAttrName, k_versionNumber);

  MIPField<DenseField<half> >::Ptr halfDenseField = 
    field_dynamic_cast<MIPField<DenseField<half> > >(field);
  MIPField<DenseField<float> >::Ptr floatDenseField = 
    field_dynamic_cast<MIPField<DenseField<float> > >(field);
  MIPField<DenseField<double> >::Ptr doubleDenseField = 
    field_dynamic_cast<MIPField<DenseField<double> > >(field);
  MIPField<DenseField<V3h> >::Ptr vecHalfDenseField = 
    field_dynamic_cast<MIPField<DenseField<V3h> > >(field);
  MIPField<DenseField<V3f> >::Ptr vecFloatDenseField = 
    field_dynamic_cast<MIPField<DenseField<V3f> > >(field);
  MIPField<DenseField<V3d> >::Ptr vecDoubleDenseField = 
    field_dynamic_cast<MIPField<DenseField<V3d> > >(field);
  MIPField<SparseField<half> >::Ptr halfSparseField = 
    field_dynamic_cast<MIPField<SparseField<half> > >(field);
  MIPField<SparseField<float> >::Ptr floatSparseField = 
    field_dynamic_cast<MIPField<SparseField<float> > >(field);
  MIPField<SparseField<double> >::Ptr doubleSparseField = 
    field_dynamic_cast<MIPField<SparseField<double> > >(field);
  MIPField<SparseField<V3h> >::Ptr vecHalfSparseField = 
    field_dynamic_cast<MIPField<SparseField<V3h> > >(field);
  MIPField<SparseField<V3f> >::Ptr vecFloatSparseField = 
    field_dynamic_cast<MIPField<SparseField<V3f> > >(field);
  MIPField<SparseField<V3d> >::Ptr vecDoubleSparseField = 
    field_dynamic_cast<MIPField<SparseField<V3d> > >(field);
  
  bool success = true;

  if (floatDenseField) {
    success = writeInternal<DenseField, float>(layerGroup, floatDenseField);
  }
  else if (halfDenseField) {
    success = writeInternal<DenseField, half>(layerGroup, halfDenseField);
  }
  else if (doubleDenseField) {
    success = writeInternal<DenseField, double>(layerGroup, doubleDenseField);
  }
  else if (vecFloatDenseField) {
    success = writeInternal<DenseField, V3f>(layerGroup, vecFloatDenseField);
  }
  else if (vecHalfDenseField) {
    success = writeInternal<DenseField, V3h>(layerGroup, vecHalfDenseField);
  }
  else if (vecDoubleDenseField) {
    success = writeInternal<DenseField, V3d>(layerGroup, vecDoubleDenseField);
  }
  else if (floatSparseField) {
    success = writeInternal<SparseField, float>(layerGroup, floatSparseField);
  }
  else if (halfSparseField) {
    success = writeInternal<SparseField, half>(layerGroup, halfSparseField);
  }
  else if (doubleSparseField) {
    success = writeInternal<SparseField, double>(layerGroup, doubleSparseField);
  }
  else if (vecFloatSparseField) {
    success = writeInternal<SparseField, V3f>(layerGroup, vecFloatSparseField);
  }
  else if (vecHalfSparseField) {
    success = writeInternal<SparseField, V3h>(layerGroup, vecHalfSparseField);
  }
  else if (vecDoubleSparseField) {
    success = writeInternal<SparseField, V3d>(layerGroup, vecDoubleSparseField);
  }
  else {
    throw WriteLayerException("MIPFieldIO does not support the given "
                              "MIPField template parameter");
  }

  return true;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
