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

/*! \file SparseFieldIO.cpp
  \brief Contains implementations of the SparseFieldIO class.
*/

//----------------------------------------------------------------------------//

#include <boost/intrusive_ptr.hpp>

#include "SparseFieldIO.h"
#include "Types.h"

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

const int         SparseFieldIO::k_versionNumber(1);
const std::string SparseFieldIO::k_versionAttrName("version");
const std::string SparseFieldIO::k_extentsStr("extents");
const std::string SparseFieldIO::k_extentsMinStr("extents_min");
const std::string SparseFieldIO::k_extentsMaxStr("extents_max");
const std::string SparseFieldIO::k_dataWindowStr("data_window");
const std::string SparseFieldIO::k_dataWindowMinStr("data_window_min");
const std::string SparseFieldIO::k_dataWindowMaxStr("data_window_max");
const std::string SparseFieldIO::k_componentsStr("components");
const std::string SparseFieldIO::k_dataStr("data");
const std::string SparseFieldIO::k_blockOrderStr("block_order");
const std::string SparseFieldIO::k_numBlocksStr("num_blocks");
const std::string SparseFieldIO::k_blockResStr("block_res");
const std::string SparseFieldIO::k_bitsPerComponentStr("bits_per_component");
const std::string SparseFieldIO::k_numOccupiedBlocksStr("num_occupied_blocks");

//----------------------------------------------------------------------------//

FieldBase::Ptr
SparseFieldIO::read(hid_t layerGroup, const std::string &filename, 
                    const std::string &layerPath,
                    DataTypeEnum typeEnum)
{
  Box3i extents, dataW;
  int components;
  int blockOrder;
  int numBlocks;
  V3i blockRes;
  
  if (layerGroup == -1) {
    Msg::print(Msg::SevWarning, "Bad layerGroup.");
    return FieldBase::Ptr();
  }

  int version;
  if (!readAttribute(layerGroup, k_versionAttrName, 1, version)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_versionAttrName);

  if (version != k_versionNumber) 
    throw UnsupportedVersionException("SparseField version not supported: " +
                                      lexical_cast<std::string>(version));

  if (!readAttribute(layerGroup, k_extentsStr, 6, extents.min.x)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_extentsStr);

  if (!readAttribute(layerGroup, k_dataWindowStr, 6, dataW.min.x)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_dataWindowStr);
  
  if (!readAttribute(layerGroup, k_componentsStr, 1, components)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_componentsStr);
  
  // Read block order
  if (!readAttribute(layerGroup, k_blockOrderStr, 1, blockOrder)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_blockOrderStr);

  // Read number of blocks total
  if (!readAttribute(layerGroup, k_numBlocksStr, 1, numBlocks)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numBlocksStr);

  // Read block resolution in each dimension
  if (!readAttribute(layerGroup, k_blockResStr, 3, blockRes.x)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_blockResStr);

  // ... Check that it matches the # reported by summing the active blocks

  int numCalculatedBlocks = blockRes.x * blockRes.y * blockRes.z;
  if (numCalculatedBlocks != numBlocks)
    throw FileIntegrityException("Incorrect block count in SparseFieldIO::read");

  // Call the appropriate read function based on the data type ---

  FieldBase::Ptr result;
  
  int occupiedBlocks;
  if (!readAttribute(layerGroup, k_numOccupiedBlocksStr, 1, occupiedBlocks)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numOccupiedBlocksStr);

  // Check the data type ---

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

  // Finally, read the data ---

  if (components == 1) {
    if (isHalf && typeEnum == DataTypeHalf) {
      SparseField<half>::Ptr field(new SparseField<half>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<half>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } else if (isFloat && typeEnum == DataTypeFloat) {
      SparseField<float>::Ptr field(new SparseField<float>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<float>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } else if (isDouble && typeEnum == DataTypeDouble) {
      SparseField<double>::Ptr field(new SparseField<double>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<double>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    }
  } else if (components == 3) {
    if (isHalf && typeEnum == DataTypeVecHalf) {
      SparseField<V3h>::Ptr field(new SparseField<V3h>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<V3h>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } else if (isFloat && typeEnum == DataTypeVecFloat) {
      SparseField<V3f>::Ptr field(new SparseField<V3f>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<V3f>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } else if (isDouble && typeEnum == DataTypeVecDouble) {
      SparseField<V3d>::Ptr field(new SparseField<V3d>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<V3d>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    }    
  }

  return result;
}

//----------------------------------------------------------------------------//

FieldBase::Ptr 
SparseFieldIO::read(const OgIGroup &layerGroup, const std::string &filename, 
                    const std::string &layerPath, OgDataType typeEnum)
{
  Box3i extents, dataW;
  int blockOrder;
  int numBlocks;
  V3i blockRes;
  
  if (!layerGroup.isValid()) {
    Msg::print(Msg::SevWarning, "Bad layerGroup.");
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
    throw UnsupportedVersionException("SparseField version not supported: " +
                                      lexical_cast<std::string>(version));
  }

  // Get extents ---

  OgIAttribute<veci32_t> extMinAttr = 
    layerGroup.findAttribute<veci32_t>(k_extentsMinStr);
  OgIAttribute<veci32_t> extMaxAttr = 
    layerGroup.findAttribute<veci32_t>(k_extentsMaxStr);
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
    layerGroup.findAttribute<veci32_t>(k_dataWindowMinStr);
  OgIAttribute<veci32_t> dwMaxAttr = 
    layerGroup.findAttribute<veci32_t>(k_dataWindowMaxStr);
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

  OgIAttribute<uint8_t> numComponentsAttr = 
    layerGroup.findAttribute<uint8_t>(k_componentsStr);
  if (!numComponentsAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_componentsStr);
  }

  // Read block order ---

  OgIAttribute<uint8_t> blockOrderAttr = 
    layerGroup.findAttribute<uint8_t>(k_blockOrderStr); 
  if (!blockOrderAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_blockOrderStr);
  }
  blockOrder = blockOrderAttr.value();

  // Read number of blocks total ---
  
  OgIAttribute<uint32_t> numBlocksAttr = 
    layerGroup.findAttribute<uint32_t>(k_numBlocksStr);
  if (!numBlocksAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numBlocksStr);
  }
  numBlocks = numBlocksAttr.value();

  // Read block resolution in each dimension ---

  OgIAttribute<veci32_t> blockResAttr = 
    layerGroup.findAttribute<veci32_t>(k_blockResStr);
  if (!blockResAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_blockResStr);
  }
  blockRes = blockResAttr.value();

  // ... Check that it matches the # reported by summing the active blocks

  int numCalculatedBlocks = blockRes.x * blockRes.y * blockRes.z;
  if (numCalculatedBlocks != numBlocks) {
    throw FileIntegrityException("Incorrect block count in "
                                 "SparseFieldIO::read()");
  }

  // Call the appropriate read function based on the data type ---

  OgIAttribute<uint32_t> occupiedBlocksAttr = 
    layerGroup.findAttribute<uint32_t>(k_numOccupiedBlocksStr);
  if (!occupiedBlocksAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numOccupiedBlocksStr);
  }
  
  // Finally, read the data ---

  FieldBase::Ptr result;
  
  OgDataType typeOnDisk = layerGroup.datasetType(k_dataStr);

  if (typeEnum == typeOnDisk) {
    if (typeEnum == F3DFloat16) {
      result = readData<float16_t>(layerGroup, extents, dataW, blockOrder,
                                   numBlocks, filename, layerPath);
    } else if (typeEnum == F3DFloat32) {
      result = readData<float32_t>(layerGroup, extents, dataW, blockOrder,
                                   numBlocks, filename, layerPath);
    } else if (typeEnum == F3DFloat64) {
      result = readData<float64_t>(layerGroup, extents, dataW, blockOrder,
                                   numBlocks, filename, layerPath);
    } else if (typeEnum == F3DVec16) {
      result = readData<vec16_t>(layerGroup, extents, dataW, blockOrder,
                                 numBlocks, filename, layerPath);
    } else if (typeEnum == F3DVec32) {
      result = readData<vec32_t>(layerGroup, extents, dataW, blockOrder,
                                 numBlocks, filename, layerPath);
    } else if (typeEnum == F3DVec64) {
      result = readData<vec64_t>(layerGroup, extents, dataW, blockOrder,
                                 numBlocks, filename, layerPath);
    } 
  }

#if 0
  if (components == 1) {
    if (isHalf && typeEnum == DataTypeHalf) {
      SparseField<half>::Ptr field(new SparseField<half>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<half>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } 
  } 
#endif

  return result;
}

//----------------------------------------------------------------------------//

bool
SparseFieldIO::write(hid_t layerGroup, FieldBase::Ptr field)
{
  if (layerGroup == -1) {
    Msg::print(Msg::SevWarning, "Bad layerGroup.");
    return false;
  }

  // Add version attribute
  if (!writeAttribute(layerGroup, k_versionAttrName, 
                    1, k_versionNumber)) {
    Msg::print(Msg::SevWarning, "Error adding version attribute.");
    return false;
  }

  SparseField<half>::Ptr halfField = 
    field_dynamic_cast<SparseField<half> >(field);
  SparseField<float>::Ptr floatField = 
    field_dynamic_cast<SparseField<float> >(field);
  SparseField<double>::Ptr doubleField = 
    field_dynamic_cast<SparseField<double> >(field);
  SparseField<V3h>::Ptr vecHalfField = 
    field_dynamic_cast<SparseField<V3h> >(field);
  SparseField<V3f>::Ptr vecFloatField = 
    field_dynamic_cast<SparseField<V3f> >(field);
  SparseField<V3d>::Ptr vecDoubleField = 
    field_dynamic_cast<SparseField<V3d> >(field);

  bool success = true;
  if (halfField) {
    success = writeInternal<half>(layerGroup, halfField);
  } else if (floatField) {
    success = writeInternal<float>(layerGroup, floatField);
  } else if (doubleField) {
    success = writeInternal<double>(layerGroup, doubleField);
  } else if (vecHalfField) {
    success = writeInternal<V3h>(layerGroup, vecHalfField);
  } else if (vecFloatField) {
    success = writeInternal<V3f>(layerGroup, vecFloatField);
  } else if (vecDoubleField) {
    success = writeInternal<V3d>(layerGroup, vecDoubleField);
  } else {
    throw WriteLayerException("SparseFieldIO::write does not support the given "
                              "SparseField template parameter");
  }

  return success;
}

//----------------------------------------------------------------------------//

bool
SparseFieldIO::write(OgOGroup &layerGroup, FieldBase::Ptr field)
{
  using namespace Exc;

  // Add version attribute
  OgOAttribute<int> version(layerGroup, k_versionAttrName, k_versionNumber);

  SparseField<half>::Ptr halfField = 
    field_dynamic_cast<SparseField<half> >(field);
  SparseField<float>::Ptr floatField = 
    field_dynamic_cast<SparseField<float> >(field);
  SparseField<double>::Ptr doubleField = 
    field_dynamic_cast<SparseField<double> >(field);
  SparseField<V3h>::Ptr vecHalfField = 
    field_dynamic_cast<SparseField<V3h> >(field);
  SparseField<V3f>::Ptr vecFloatField = 
    field_dynamic_cast<SparseField<V3f> >(field);
  SparseField<V3d>::Ptr vecDoubleField = 
    field_dynamic_cast<SparseField<V3d> >(field);

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
    throw WriteLayerException("SparseFieldIO does not support the given "
                              "SparseField template parameter");
  }

  return success;
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename SparseField<Data_T>::Ptr
SparseFieldIO::readData(const OgIGroup &location, const Box3i &extents, 
                        const Box3i &dataW, const size_t blockOrder, 
                        const size_t numBlocks, const std::string &filename, 
                        const std::string &layerPath)
{
  using namespace std;
  using namespace Exc;
  using namespace Sparse;

  typename SparseField<Data_T>::Ptr result(new SparseField<Data_T>);
  result->setSize(extents, dataW);
  result->setBlockOrder(blockOrder);

  const bool   dynamicLoading = SparseFileManager::singleton().doLimitMemUse();
  const int    components     = FieldTraits<Data_T>::dataDims();
  const size_t numVoxels      = (1 << (result->m_blockOrder * 3));
  const int    valuesPerBlock = (1 << (result->m_blockOrder * 3)) * components;
  
  // Read the number of occupied blocks ---

  OgIAttribute<uint32_t> occupiedBlocksAttr = 
    location.findAttribute<uint32_t>(k_numOccupiedBlocksStr);
  if (!occupiedBlocksAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numOccupiedBlocksStr);
  }
  const int occupiedBlocks = occupiedBlocksAttr.value();

  // Set up the dynamic read info ---

  if (dynamicLoading) {
    // Set up the field reference
    //! \todo The valuesPerBlock is wrong. Fix
    result->addReference(filename, layerPath, valuesPerBlock, occupiedBlocks);
  }

  // Read the block info data sets ---

  SparseBlock<Data_T> *blocks = result->m_blocks;

  // ... Read the isAllocated array

  {
    // Grab the data
    vector<uint8_t> isAllocated(numBlocks);
    OgIDataset<uint8_t> isAllocatedData = 
      location.findDataset<uint8_t>("block_is_allocated_data");
    if (!isAllocatedData.isValid()) {
      throw MissingGroupException("Couldn't find block_is_allocated_data: ");
    }
    isAllocatedData.getData(0, &isAllocated[0], OGAWA_THREAD);
    // Allocate the blocks
    for (size_t i = 0; i < numBlocks; ++i) {
      blocks[i].isAllocated = isAllocated[i];
      if (!dynamicLoading && isAllocated[i]) {
        blocks[i].resize(numVoxels);
      }
    }
  }

  // ... Read the emptyValue array ---

  {
    // Grab the data
    vector<Data_T> emptyValue(numBlocks);
    OgIDataset<Data_T> emptyValueData = 
      location.findDataset<Data_T>("block_empty_value_data");
    if (!emptyValueData.isValid()) {
      throw MissingGroupException("Couldn't find block_empty_value_data: ");
    }
    emptyValueData.getData(0, &emptyValue[0], OGAWA_THREAD);
    // Fill in the field
    for (size_t i = 0; i < numBlocks; ++i) {
      blocks[i].emptyValue = emptyValue[i];
    }
  }

  // Read the data ---

  if (occupiedBlocks > 0) {
    if (dynamicLoading) {
      // Defer loading to the sparse cache
      result->setupReferenceBlocks();
    } else {
      // Read the data directly. The memory is already allocated
      OgSparseDataReader<Data_T> reader(location, numVoxels, occupiedBlocks);
      for (size_t i = 0; i < numBlocks; ++i) {
        if (blocks[i].isAllocated) {
          reader.readBlock(i, blocks[i].data);
        }
      }
    }
  }

  return result;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
