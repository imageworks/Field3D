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

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "InitIO.h"
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
// Anonymous namespace
//----------------------------------------------------------------------------//

namespace {

//----------------------------------------------------------------------------//

template <typename Data_T>
struct ReadThreadingState
{
  ReadThreadingState(const OgIGroup &i_location, 
                     Sparse::SparseBlock<Data_T> *i_blocks, 
                     const size_t i_numVoxels, 
                     const size_t i_numBlocks,
                     const size_t i_numOccupiedBlocks, 
                     const bool i_isCompressed,
                     const std::vector<size_t> &i_blockIdxToDatasetIdx)
    : location(i_location),
      blocks(i_blocks),
      numVoxels(i_numVoxels), 
      numBlocks(i_numBlocks),
      numOccupiedBlocks(i_numOccupiedBlocks),
      isCompressed(i_isCompressed), 
      blockIdxToDatasetIdx(i_blockIdxToDatasetIdx), 
      nextBlockToRead(0)
  { }
  // Data members
  const OgIGroup &location;
  Sparse::SparseBlock<Data_T> *blocks;
  const size_t numVoxels;
  const size_t numBlocks;
  const size_t numOccupiedBlocks;
  const bool   isCompressed;
  const std::vector<size_t> &blockIdxToDatasetIdx;
  size_t nextBlockToRead;
  // Mutexes
  boost::mutex readMutex;
};

//----------------------------------------------------------------------------//

template <typename Data_T>
class ReadBlockOp
{
public:
  ReadBlockOp(ReadThreadingState<Data_T> &state, const size_t threadId)
    : m_state(state)
  { 
    // Set up the compression cache
    const uLong srcLen      = m_state.numVoxels * sizeof(Data_T);
    const uLong cmpLenBound = compressBound(srcLen);
    m_cache.resize(cmpLenBound);
    // Initialize the reader
    m_readerPtr.reset(
      new OgSparseDataReader<Data_T>(m_state.location, m_state.numVoxels, 
                                     m_state.numOccupiedBlocks,
                                     m_state.isCompressed));
    m_reader = m_readerPtr.get();
    // Set the thread id
    m_reader->setThreadId(threadId);
  }
  void operator() ()
  {
    // Get next block to read
    size_t blockIdx;
    {
      boost::mutex::scoped_lock lock(m_state.readMutex);
      blockIdx = m_state.nextBlockToRead;
      m_state.nextBlockToRead++;
    }
    // Loop over blocks until we run out
    while (blockIdx < m_state.numBlocks) {
      if (m_state.blocks[blockIdx].isAllocated) {
        const size_t datasetIdx = m_state.blockIdxToDatasetIdx[blockIdx];
        m_reader->readBlock(datasetIdx, m_state.blocks[blockIdx].data);
      }
      // Get next block idx
      {
        boost::mutex::scoped_lock lock(m_state.readMutex);
        blockIdx = m_state.nextBlockToRead;
        m_state.nextBlockToRead++;
      }
    }
  }
private:
  // Data members ---
  ReadThreadingState<Data_T> &m_state;
  std::vector<uint8_t> m_cache;
  boost::shared_ptr<OgSparseDataReader<Data_T> > m_readerPtr;
  OgSparseDataReader<Data_T> *m_reader;
};

//----------------------------------------------------------------------------//

template <typename Data_T>
struct ThreadingState
{
  ThreadingState(OgOCDataset<Data_T> &i_data, 
                 Sparse::SparseBlock<Data_T> *i_blocks, 
                 const size_t i_numVoxels, 
                 const size_t i_numBlocks,
                 const std::vector<uint8_t> &i_isAllocated)
    : data(i_data),
      blocks(i_blocks),
      numVoxels(i_numVoxels), 
      numBlocks(i_numBlocks),
      isAllocated(i_isAllocated), 
      nextBlockToCompress(0),
      nextBlockToWrite(0)
  { 
    // Find first in-use block
    for (size_t i = 0; i < numBlocks; ++i) {
      if (blocks[i].isAllocated) {
        nextBlockToCompress = i;
        nextBlockToWrite = i;
        return;
      }
    }
    // If we get here, there are no active blocks. Set to numBlocks
    nextBlockToCompress = numBlocks;
    nextBlockToWrite = numBlocks;
  }
  // Data members
  OgOCDataset<Data_T> &data;
  Sparse::SparseBlock<Data_T> *blocks;
  const size_t numVoxels;
  const size_t numBlocks;
  const std::vector<uint8_t> isAllocated;
  size_t nextBlockToCompress;
  size_t nextBlockToWrite;
  // Mutexes
  boost::mutex compressMutex;
};

//----------------------------------------------------------------------------//

template <typename Data_T>
class WriteBlockOp
{
public:
  WriteBlockOp(ThreadingState<Data_T> &state, const size_t threadId)
    : m_state(state), m_threadId(threadId)
  { 
    const uLong srcLen      = m_state.numVoxels * sizeof(Data_T);
    const uLong cmpLenBound = compressBound(srcLen);
    m_cache.resize(cmpLenBound);
  }
  void operator() ()
  {
    const int level = 1;
    // Get next block to compress
    size_t blockIdx;
    {
      boost::mutex::scoped_lock lock(m_state.compressMutex);
      blockIdx = m_state.nextBlockToCompress;
      // Step counter to next
      while (m_state.nextBlockToCompress < m_state.numBlocks) {
        m_state.nextBlockToCompress++;
        if (m_state.blocks[m_state.nextBlockToCompress].isAllocated) {
          break;
        }
      }
    }
    // Loop over blocks until we run out
    while (blockIdx < m_state.numBlocks) {
      if (m_state.blocks[blockIdx].isAllocated) {
        // Block data as bytes
        const uint8_t *srcData = 
          reinterpret_cast<const uint8_t *>(m_state.blocks[blockIdx].data);
        // Length of compressed data is stored here
        const uLong srcLen      = m_state.numVoxels * sizeof(Data_T);
        const uLong cmpLenBound = compressBound(srcLen);
        uLong cmpLen            = cmpLenBound;
        // Perform compression
        const int status = compress2(&m_cache[0], &cmpLen, 
                                     srcData, srcLen, level);
        // Error check
        if (status != Z_OK) {
          std::cout << "ERROR: Couldn't compress in SparseFieldIO." << std::endl
                    << "  Level:  " << level << std::endl
                    << "  Status: " << status << std::endl
                    << "  srcLen: " << srcLen << std::endl
                    << "  cmpLenBound: " << cmpLenBound << std::endl
                    << "  cmpLen: " << cmpLen << std::endl;
          return;
        }
        // Wait to write data
        while (m_state.nextBlockToWrite != blockIdx) {
          // Spin
          boost::this_thread::sleep(boost::posix_time::microseconds(1));
        }
        // Do the writing
        m_state.data.addData(cmpLen, &m_cache[0]);
        // Let next block write
        while (m_state.nextBlockToWrite < m_state.numBlocks){
          // Increment to next
          m_state.nextBlockToWrite++;
          if (m_state.blocks[m_state.nextBlockToWrite].isAllocated) {
            break;
          }
        }
      }
      // Get next block idx
      {
        boost::mutex::scoped_lock lock(m_state.compressMutex);
        blockIdx = m_state.nextBlockToCompress;
        // Step counter to next
        while (m_state.nextBlockToCompress < m_state.numBlocks) {
          m_state.nextBlockToCompress++;
          if (m_state.blocks[m_state.nextBlockToCompress].isAllocated) {
            break;
          }
        }
      }
    }
  }
private:
  // Data members ---
  ThreadingState<Data_T> &m_state;
  std::vector<uint8_t> m_cache;
  const size_t m_threadId;
};

//----------------------------------------------------------------------------//

} // Anonymous namespace

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
const std::string SparseFieldIO::k_isCompressed("data_is_compressed");

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
    throw MissingGroupException("Invalid group in SparseFieldIO::read()");
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

  // Check if the data is compressed ---

  OgIAttribute<uint8_t> isCompressedAttr = 
    layerGroup.findAttribute<uint8_t>(k_isCompressed);
  if (!isCompressedAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_isCompressed);
  }
  
  // Finally, read the data ---

  FieldBase::Ptr result;
  
  OgDataType typeOnDisk;

  if (isCompressedAttr.value() == 0) {
    typeOnDisk = layerGroup.datasetType(k_dataStr);
  } else {
    typeOnDisk = layerGroup.compressedDatasetType(k_dataStr);
  }

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
  const size_t occupiedBlocks = occupiedBlocksAttr.value();

  // Set up the dynamic read info ---

  if (dynamicLoading) {
    // Set up the field reference
    //! \todo The valuesPerBlock is wrong. Fix
    result->addReference(filename, layerPath, valuesPerBlock, numVoxels,
                         occupiedBlocks);
  }

  // Read the block info data sets ---

  SparseBlock<Data_T> *blocks = result->m_blocks;

  // ... Read the isAllocated array and set up the block mapping array
  std::vector<size_t> blockIdxToDatasetIdx(numBlocks);

  {
    // Grab the data
    vector<uint8_t> isAllocated(numBlocks);
    OgIDataset<uint8_t> isAllocatedData = 
      location.findDataset<uint8_t>("block_is_allocated_data");
    if (!isAllocatedData.isValid()) {
      throw MissingGroupException("Couldn't find block_is_allocated_data: ");
    }
    isAllocatedData.getData(0, &isAllocated[0], OGAWA_THREAD);
    // Allocate the blocks and set up the block mapping array
    for (size_t i = 0, nextBlockOnDisk = 0; i < numBlocks; ++i) {
      blocks[i].isAllocated = isAllocated[i];
      if (!dynamicLoading && isAllocated[i]) {
        blocks[i].resize(numVoxels);
        // Update the block mapping array
        blockIdxToDatasetIdx[i] = nextBlockOnDisk;
        nextBlockOnDisk++;
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

  // Check whether data is compressed
  OgIAttribute<uint8_t> isCompressedAttr = 
    location.findAttribute<uint8_t>(k_isCompressed);
  const bool isCompressed = isCompressedAttr.value() != 0;

  if (occupiedBlocks > 0) {
    if (dynamicLoading) {
      // Defer loading to the sparse cache
      result->setupReferenceBlocks();
    } else {
      // Threading state
      ReadThreadingState<Data_T> state(location, blocks, numVoxels, numBlocks,
                                       occupiedBlocks, isCompressed,
                                       blockIdxToDatasetIdx);
      // Number of threads
      const size_t numThreads = numIOThreads();
      // Launch threads
      boost::thread_group threads;
      for (size_t i = 0; i < numThreads; ++i) {
        threads.create_thread(ReadBlockOp<Data_T>(state, i));
      }
      threads.join_all();
    }
  }

  return result;
}

//----------------------------------------------------------------------------//
// Template implementations
//----------------------------------------------------------------------------//

//! \todo Tune the chunk size of the gzip call
template <class Data_T>
bool SparseFieldIO::writeInternal(hid_t layerGroup, 
                                  typename SparseField<Data_T>::Ptr field)
{
  using namespace std;
  using namespace Exc;
  using namespace Hdf5Util;
  using namespace Sparse;

  Box3i ext(field->extents()), dw(field->dataWindow());

  int components = FieldTraits<Data_T>::dataDims();

  int valuesPerBlock = (1 << (field->m_blockOrder * 3)) * components;

  // Add extents attribute ---

  int extents[6] = 
    { ext.min.x, ext.min.y, ext.min.z, ext.max.x, ext.max.y, ext.max.z };

  if (!writeAttribute(layerGroup, k_extentsStr, 6, extents[0])) {
    Msg::print(Msg::SevWarning, "Error adding size attribute.");
    return false;
  }

  // Add data window attribute ---

  int dataWindow[6] = 
    { dw.min.x, dw.min.y, dw.min.z, dw.max.x, dw.max.y, dw.max.z };

  if (!writeAttribute(layerGroup, k_dataWindowStr, 6, dataWindow[0])) {
    Msg::print(Msg::SevWarning, "Error adding size attribute.");
    return false;
  }

  // Add components attribute ---

  if (!writeAttribute(layerGroup, k_componentsStr, 1, components)) {
    Msg::print(Msg::SevWarning, "Error adding components attribute.");
    return false;
  }

  // Add block order attribute ---

  int blockOrder = field->m_blockOrder;

  if (!writeAttribute(layerGroup, k_blockOrderStr, 1, blockOrder)) {
    Msg::print(Msg::SevWarning, "Error adding block order attribute.");
    return false;
  }

  // Add number of blocks attribute ---
  
  V3i &blockRes = field->m_blockRes;
  int numBlocks = blockRes.x * blockRes.y * blockRes.z;

  if (!writeAttribute(layerGroup, k_numBlocksStr, 1, numBlocks)) {
    Msg::print(Msg::SevWarning, "Error adding number of blocks attribute.");
    return false;
  }

  // Add block resolution in each dimension ---

  if (!writeAttribute(layerGroup, k_blockResStr, 3, blockRes.x)) {
    Msg::print(Msg::SevWarning, "Error adding block res attribute.");
    return false;
  }

  // Add the bits per component attribute ---

  int bits = DataTypeTraits<Data_T>::h5bits();
  if (!writeAttribute(layerGroup, k_bitsPerComponentStr, 1, bits)) {
    Msg::print(Msg::SevWarning, "Error adding bits per component attribute.");
    return false;    
  }

  // Write the block info data sets ---
  
  SparseBlock<Data_T> *blocks = field->m_blocks;

  // ... Write the isAllocated array
  {
    vector<char> isAllocated(numBlocks);
    for (int i = 0; i < numBlocks; ++i) {
      isAllocated[i] = static_cast<char>(blocks[i].isAllocated);
    }
    writeSimpleData<char>(layerGroup, "block_is_allocated_data", isAllocated);
  }

  // ... Write the emptyValue array
  {
    vector<Data_T> emptyValue(numBlocks);
    for (int i = 0; i < numBlocks; ++i) {
      emptyValue[i] = static_cast<Data_T>(blocks[i].emptyValue);
    }
    writeSimpleData<Data_T>(layerGroup, "block_empty_value_data", emptyValue);
  }

  // Count the number of occupied blocks ---
  int occupiedBlocks = 0;
  for (int i = 0; i < numBlocks; ++i) {
    if (blocks[i].isAllocated) {
      occupiedBlocks++;
    }
  }

  if (!writeAttribute(layerGroup, k_numOccupiedBlocksStr, 1, occupiedBlocks)) {
    throw WriteAttributeException("Couldn't add attribute " + 
                                k_numOccupiedBlocksStr);
  }
  
  if (occupiedBlocks > 0) {

    // Make the memory data space
    hsize_t memDims[1];
    memDims[0] = valuesPerBlock;
    H5ScopedScreate memDataSpace(H5S_SIMPLE);
    H5Sset_extent_simple(memDataSpace.id(), 1, memDims, NULL);

    // Make the file data space
    hsize_t fileDims[2];
    fileDims[0] = occupiedBlocks;
    fileDims[1] = valuesPerBlock;
    H5ScopedScreate fileDataSpace(H5S_SIMPLE);
    H5Sset_extent_simple(fileDataSpace.id(), 2, fileDims, NULL);

    // Set up gzip property list
    bool gzipAvailable = checkHdf5Gzip();
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunkSize[2];
    chunkSize[0] = 1;
    chunkSize[1] = valuesPerBlock;
    if (gzipAvailable) {
      herr_t status = H5Pset_deflate(dcpl, 9);
      if (status < 0) {
        return false;
      }
      status = H5Pset_chunk(dcpl, 2, chunkSize);
      if (status < 0) {
        return false;
      }    
    }

    // Add the data set
    H5ScopedDcreate dataSet(layerGroup, k_dataStr, 
                            DataTypeTraits<Data_T>::h5type(), 
                            fileDataSpace.id(), 
                            H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dataSet.id() < 0)
      throw CreateDataSetException("Couldn't create data set in "
                                   "SparseFieldIO::writeInternal");

    // For each allocated block ---

    int nextBlockIdx = 0;
    hsize_t offset[2];
    hsize_t count[2];
    herr_t status;

    for (int i = 0; i < numBlocks; ++i) {
      if (blocks[i].isAllocated) {
        offset[0] = nextBlockIdx;  // Index of next block
        offset[1] = 0;             // Index of first data in block. Always 0
        count[0] = 1;              // Number of columns to read. Always 1
        count[1] = valuesPerBlock; // Number of values in one column
        status = H5Sselect_hyperslab(fileDataSpace.id(), H5S_SELECT_SET, 
                                     offset, NULL, count, NULL);
        if (status < 0) {
          throw WriteHyperSlabException(
            "Couldn't select slab " + 
            boost::lexical_cast<std::string>(nextBlockIdx));
        }
        Data_T *data = field->m_blocks[i].data;
        status = H5Dwrite(dataSet.id(), DataTypeTraits<Data_T>::h5type(), 
                          memDataSpace.id(), 
                          fileDataSpace.id(), H5P_DEFAULT, data);
        if (status < 0) {
          throw WriteHyperSlabException(
            "Couldn't write slab " + 
            boost::lexical_cast<std::string>(nextBlockIdx));
        }
        // Increment nextBlockIdx
        nextBlockIdx++;
      }
    }

  } // if occupiedBlocks > 0

  return true; 

}

//----------------------------------------------------------------------------//

template <class Data_T>
bool SparseFieldIO::writeInternal(OgOGroup &layerGroup, 
                                  typename SparseField<Data_T>::Ptr field)
{
  using namespace Exc;
  using namespace Sparse;

  SparseBlock<Data_T> *blocks = field->m_blocks;

  const int    components     = FieldTraits<Data_T>::dataDims();
  const int    bits           = DataTypeTraits<Data_T>::h5bits();
  const V3i   &blockRes       = field->m_blockRes;
  const size_t numBlocks      = blockRes.x * blockRes.y * blockRes.z;
  const size_t numVoxels      = (1 << (field->m_blockOrder * 3));
  
  const Box3i ext(field->extents()), dw(field->dataWindow());

  // Add attributes ---

  OgOAttribute<veci32_t> extMinAttr(layerGroup, k_extentsMinStr, ext.min);
  OgOAttribute<veci32_t> extMaxAttr(layerGroup, k_extentsMaxStr, ext.max);
  
  OgOAttribute<veci32_t> dwMinAttr(layerGroup, k_dataWindowMinStr, dw.min);
  OgOAttribute<veci32_t> dwMaxAttr(layerGroup, k_dataWindowMaxStr, dw.max);

  OgOAttribute<uint8_t> componentsAttr(layerGroup, k_componentsStr, components);

  OgOAttribute<uint8_t> bitsAttr(layerGroup, k_bitsPerComponentStr, bits);

  OgOAttribute<uint8_t> blockOrderAttr(layerGroup, k_blockOrderStr, 
                                       field->m_blockOrder);

  OgOAttribute<uint32_t> numBlocksAttr(layerGroup, k_numBlocksStr, numBlocks);

  OgOAttribute<veci32_t> blockResAttr(layerGroup, k_blockResStr, blockRes);

  OgOAttribute<uint8_t> isCompressedAttr(layerGroup, k_isCompressed, 1);
  
  // Write the isAllocated array
  std::vector<uint8_t> isAllocated(numBlocks);
  for (size_t i = 0; i < numBlocks; ++i) {
    isAllocated[i] = static_cast<uint8_t>(blocks[i].isAllocated);
  }
  OgODataset<uint8_t> isAllocatedData(layerGroup, "block_is_allocated_data");
  isAllocatedData.addData(numBlocks, &isAllocated[0]);

  // Write the emptyValue array
  std::vector<Data_T> emptyValue(numBlocks);
  for (size_t i = 0; i < numBlocks; ++i) {
    emptyValue[i] = static_cast<Data_T>(blocks[i].emptyValue);
  }
  OgODataset<Data_T> emptyValueData(layerGroup, "block_empty_value_data");
  emptyValueData.addData(numBlocks, &emptyValue[0]);
    
  // Count the number of occupied blocks
  int occupiedBlocks = 0;
  for (size_t i = 0; i < numBlocks; ++i) {
    if (blocks[i].isAllocated) {
      occupiedBlocks++;
    }
  }
  OgOAttribute<uint32_t> numOccupiedBlockAttr(layerGroup, 
                                              k_numOccupiedBlocksStr, 
                                              occupiedBlocks);

  // Add data to file ---

  // Create the compressed dataset regardless of whether there are blocks
  // to write.
  OgOCDataset<Data_T> data(layerGroup, k_dataStr);
  // Write data if there is any
  if (occupiedBlocks > 0) {
    // Threading state
    ThreadingState<Data_T> state(data, blocks, numVoxels, numBlocks, 
                                 isAllocated);
    // Number of threads
    const size_t numThreads = numIOThreads();
    // Launch threads
    boost::thread_group threads;
    for (size_t i = 0; i < numThreads; ++i) {
      threads.create_thread(WriteBlockOp<Data_T>(state, i));
    }
    threads.join_all();
  }

  return true;
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool SparseFieldIO::readData(hid_t location, 
                             int numBlocks, 
                             const std::string &filename, 
                             const std::string &layerPath, 
                             typename SparseField<Data_T>::Ptr result)
{
  using namespace std;
  using namespace Exc;
  using namespace Hdf5Util;
  using namespace Sparse;

  int occupiedBlocks;

  bool dynamicLoading = SparseFileManager::singleton().doLimitMemUse();

  int components = FieldTraits<Data_T>::dataDims();
  int numVoxels = (1 << (result->m_blockOrder * 3));
  int valuesPerBlock = numVoxels * components;
  
  // Read the number of occupied blocks ---

  if (!readAttribute(location, k_numOccupiedBlocksStr, 1, occupiedBlocks)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_numOccupiedBlocksStr);

  // Set up the dynamic read info ---

  if (dynamicLoading) {
    // Set up the field reference
    result->addReference(filename, layerPath,
                         valuesPerBlock, numVoxels, 
                         occupiedBlocks);
  }

  // Read the block info data sets ---

  SparseBlock<Data_T> *blocks = result->m_blocks;

  // ... Read the isAllocated array

  {
    vector<char> isAllocated(numBlocks);
    readSimpleData<char>(location, "block_is_allocated_data", isAllocated);
    for (int i = 0; i < numBlocks; ++i) {
      blocks[i].isAllocated = isAllocated[i];
      if (!dynamicLoading && isAllocated[i]) {
        blocks[i].resize(numVoxels);
      }
    }
  }

  // ... Read the emptyValue array ---

  {
    vector<Data_T> emptyValue(numBlocks);
    readSimpleData<Data_T>(location, "block_empty_value_data", emptyValue);
    for (int i = 0; i < numBlocks; ++i) {
      blocks[i].emptyValue = emptyValue[i];
    }
  }

  // Read the data ---

  if (occupiedBlocks > 0) {

    if (dynamicLoading) {

      result->setupReferenceBlocks();

    } else {
      
      size_t b = 0, bend = b + numBlocks;

      SparseDataReader<Data_T> reader(location, valuesPerBlock, occupiedBlocks);

      // We'll read at most 50meg at a time
      static const long maxMemPerPass = 50*1024*1024;

      for (int nextBlockIdx = 0;;) {

        long mem = 0;
        std::vector<Data_T*> memoryList;
        
        for (; b != bend && mem < maxMemPerPass; ++b) {
          if (blocks[b].isAllocated) {
            mem += sizeof(Data_T)*numVoxels;
            memoryList.push_back(blocks[b].data);
          }
        }

        // all done.
        if (!memoryList.size()) {
          break;
        }

        reader.readBlockList(nextBlockIdx, memoryList);
        nextBlockIdx += memoryList.size();
      }                           

    }

  } // if occupiedBlocks > 0

  return true;
  
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
