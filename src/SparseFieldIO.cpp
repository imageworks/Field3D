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
const std::string SparseFieldIO::k_dataWindowStr("data_window");
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
                    const std::string &layerPath)
{
  Box3i extents, dataW;
  int components;
  int blockOrder;
  int numBlocks;
  V3i blockRes;
  
  if (layerGroup == -1) {
    Log::print(Log::SevWarning, "Bad layerGroup.");
    return FieldBase::Ptr();
  }

  int version;
  if (!readAttribute(layerGroup, k_versionAttrName, 1, version)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_versionAttrName);

  if (version != k_versionNumber) 
    throw UnsupportedVersionException("SparseField version not supported: " +
                                      str(version));

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
    if (isHalf) {
      SparseField<half>::Ptr field(new SparseField<half>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<half>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } else if (isFloat) {
      SparseField<float>::Ptr field(new SparseField<float>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<float>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } else if (isDouble) {
      SparseField<double>::Ptr field(new SparseField<double>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<double>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    }
  } else if (components == 3) {
    if (isHalf) {
      SparseField<V3h>::Ptr field(new SparseField<V3h>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<V3h>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } else if (isFloat) {
      SparseField<V3f>::Ptr field(new SparseField<V3f>);
      field->setSize(extents, dataW);
      field->setBlockOrder(blockOrder);
      readData<V3f>(layerGroup, numBlocks, filename, layerPath, field);
      result = field;      
    } else if (isDouble) {
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

bool
SparseFieldIO::write(hid_t layerGroup, FieldBase::Ptr field)
{
  if (layerGroup == -1) {
    Log::print(Log::SevWarning, "Bad layerGroup.");
    return false;
  }

  // Add version attribute
  if (!writeAttribute(layerGroup, FieldIO::versionAttrName, 
                    1, k_versionNumber)) {
    Log::print(Log::SevWarning, "Error adding version attribute.");
    return false;
  }

  SparseField<half>::Ptr halfField = 
    dynamic_pointer_cast<SparseField<half> >(field);
  SparseField<float>::Ptr floatField = 
    dynamic_pointer_cast<SparseField<float> >(field);
  SparseField<double>::Ptr doubleField = 
    dynamic_pointer_cast<SparseField<double> >(field);
  SparseField<V3h>::Ptr vecHalfField = 
    dynamic_pointer_cast<SparseField<V3h> >(field);
  SparseField<V3f>::Ptr vecFloatField = 
    dynamic_pointer_cast<SparseField<V3f> >(field);
  SparseField<V3d>::Ptr vecDoubleField = 
    dynamic_pointer_cast<SparseField<V3d> >(field);

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

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
