//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2018 Sony Pictures Imageworks Inc.,
 *                    Pixar Animation Studios
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

/*! \file CSparseFieldIO.cpp
  \brief Contains implementations of the CSparseFieldIO class.
*/

//----------------------------------------------------------------------------//

#include <boost/intrusive_ptr.hpp>

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "InitIO.h"
#include "CSparseField.h"
#include "CSparseFieldImpl.h"
#include "CSparseFieldIO.h"
#include "OgCSparseDataReader.h"
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

struct ReadThreadingState
{
  ReadThreadingState(const OgIGroup &i_location, 
                     CSparseFieldImpl *i_impl,
                     const size_t i_numAllocatedBlocks)
    : location(i_location),
      impl(i_impl),
      nextBlockToRead(0),
      numAllocatedBlocks(i_numAllocatedBlocks)
  { }
  // Data members
  const OgIGroup            &location;
  CSparseFieldImpl          *impl;
  size_t                     nextBlockToRead;
  const size_t               numAllocatedBlocks;
  // Mutexes
  boost::mutex readMutex;
};

//----------------------------------------------------------------------------//

template <typename Data_T>
class ReadBlockOp
{
public:
  ReadBlockOp(ReadThreadingState &state, const size_t threadId)
    : m_state(state)
  { 
    // Set up the compression cache
    const uLong srcLen = m_state.impl->numBlocks > 0 ? 
      m_state.impl->dataLength() : 1;
    const uLong cmpLenBound = compressBound(srcLen);
    m_cache.resize(cmpLenBound);
    // Initialize the reader
    m_readerPtr.reset(
      new OgCSparseDataReader<Data_T>(m_state.location, m_state.impl));
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
    while (blockIdx < m_state.numAllocatedBlocks) {
      m_reader->readBlock(blockIdx, m_state.impl);
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

  ReadThreadingState                              &m_state;
  std::vector<uint8_t>                             m_cache;
  boost::shared_ptr<OgCSparseDataReader<Data_T> >  m_readerPtr;
  OgCSparseDataReader<Data_T>                     *m_reader;
};

//----------------------------------------------------------------------------//

struct ThreadingState
{
  ThreadingState(OgOCDataset<uint8_t> &i_data, 
                 CSparseFieldImpl *i_impl)
    : data(i_data),
      impl(i_impl),
      nextBlockToCompress(0),
      nextBlockToWrite(0)
  { 
    // Find first in-use block
    for (size_t i = 0; i < impl->numBlocks; ++i) {
      if (impl->blockIsAllocated(i)) {
        nextBlockToCompress = i;
        nextBlockToWrite = i;
        return;
      }
    }
    // If we get here, there are no active blocks. Set to numBlocks
    nextBlockToCompress = impl->numBlocks;
    nextBlockToWrite = impl->numBlocks;
  }
  // Data members
  OgOCDataset<uint8_t> &data;
  CSparseFieldImpl    *impl;
  size_t               nextBlockToCompress;
  size_t               nextBlockToWrite;
  // Mutexes
  boost::mutex compressMutex;
};

//----------------------------------------------------------------------------//

template <typename Data_T>
class WriteBlockOp
{
public:
  WriteBlockOp(ThreadingState &state, const size_t threadId)
    : m_state(state), 
      m_srcLen(state.impl->dataLength()),
      m_threadId(threadId)
  { 
    const uLong cmpLenBound = compressBound(m_srcLen);
    const uLong numLayers   = m_state.impl->numLayers();
    // Set sizes
    m_arrayLength = cmpLenBound;
    m_numLayers = numLayers;
    // Allocate cache
    m_length.resize(m_numLayers);
    m_cache.resize(m_numLayers);
    for (size_t i = 0; i < m_numLayers; ++i) {
      m_cache[i].resize(m_arrayLength);
    }
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
      while (m_state.nextBlockToCompress < m_state.impl->numBlocks) {
        m_state.nextBlockToCompress++;
        if (m_state.impl->blockIsAllocated(m_state.nextBlockToCompress)) {
          break;
        }
      }
    }
    // Loop over blocks until we run out
    while (blockIdx < m_state.impl->numBlocks) {
      if (m_state.impl->blockIsAllocated(blockIdx)) {
        // For each layer
        for (int c = 0; c < m_numLayers; ++c) {
          // Block data as bytes
          const uint8_t *srcData = 
            reinterpret_cast<const uint8_t *>(m_state.impl->data(c, blockIdx));
          // Perform compression
          uLong cmpLen = m_arrayLength;
          const int status = compress2(&m_cache[c][0], &cmpLen, 
                                       srcData, m_srcLen, level);
          // Update length
          m_length[c] = cmpLen;
          // Error check
          if (status != Z_OK) {
            std::cout << "ERROR: Couldn't compress in CSparseFieldIO." << std::endl
                      << "  Level:  " << level << std::endl
                      << "  Status: " << status << std::endl
                      << "  srcLen: " << m_srcLen << std::endl
                      << "  cmpLenBound: " << m_arrayLength << std::endl
                      << "  cmpLen: " << cmpLen << std::endl;
            return;
          }
        }
        // Wait to write data
        while (m_state.nextBlockToWrite != blockIdx) {
          // Spin
          boost::this_thread::sleep(boost::posix_time::microseconds(1));
        }
        // Do the writing
        for (int c = 0; c < m_numLayers; ++c) {
          m_state.data.addData(m_length[c], &m_cache[c][0]);
        }
        // Let next block write (only once all components are written)
        while (m_state.nextBlockToWrite < m_state.impl->numBlocks){
          // Increment to next
          m_state.nextBlockToWrite++;
          if (m_state.impl->blockIsAllocated(m_state.nextBlockToWrite)) {
            break;
          }
        }
      }
      // Get next block idx
      {
        boost::mutex::scoped_lock lock(m_state.compressMutex);
        blockIdx = m_state.nextBlockToCompress;
        // Step counter to next
        while (m_state.nextBlockToCompress < m_state.impl->numBlocks) {
          m_state.nextBlockToCompress++;
          if (m_state.impl->blockIsAllocated(m_state.nextBlockToCompress)) {
            break;
          }
        }
      }
    }
  }
private:
  // Data members ---
  ThreadingState                     &m_state;
  size_t                              m_srcLen;
  size_t                              m_arrayLength;
  size_t                              m_numLayers;
  std::vector<std::vector<uint8_t> >  m_cache;
  std::vector<size_t>                 m_length;
  const size_t                        m_threadId;
};

//----------------------------------------------------------------------------//

} // Anonymous namespace

//----------------------------------------------------------------------------//
// Static members
//----------------------------------------------------------------------------//

const int         CSparseFieldIO::k_versionNumber(1);
const std::string CSparseFieldIO::k_versionAttrName("version");
const std::string CSparseFieldIO::k_extentsStr("extents");
const std::string CSparseFieldIO::k_extentsMinStr("extents_min");
const std::string CSparseFieldIO::k_extentsMaxStr("extents_max");
const std::string CSparseFieldIO::k_dataWindowStr("data_window");
const std::string CSparseFieldIO::k_dataWindowMinStr("data_window_min");
const std::string CSparseFieldIO::k_dataWindowMaxStr("data_window_max");
const std::string CSparseFieldIO::k_bitRateStr("bit_rate");
const std::string CSparseFieldIO::k_componentsStr("components");
const std::string CSparseFieldIO::k_dataStr("data");
const std::string CSparseFieldIO::k_dataTypeStr("data_type");
const std::string CSparseFieldIO::k_blockOrderStr("block_order");
const std::string CSparseFieldIO::k_numBlocksStr("num_blocks");
const std::string CSparseFieldIO::k_blockResStr("block_res");
const std::string CSparseFieldIO::k_blockMapStr("block_map");
const std::string CSparseFieldIO::k_blockEmptyValueStr("block_empty_value");

//----------------------------------------------------------------------------//

FieldBase::Ptr
CSparseFieldIO::read(hid_t layerGroup, const std::string &filename, 
                    const std::string &layerPath,
                    DataTypeEnum typeEnum)
{
  return nullptr;
}

//----------------------------------------------------------------------------//

FieldBase::Ptr 
CSparseFieldIO::read(const OgIGroup &layerGroup, const std::string &filename, 
                    const std::string &layerPath, OgDataType typeEnum)
{
  Box3i extents, dataW;
  int blockOrder;
  int numBlocks;
  int bitRate;
  V3i blockRes;
  
  if (!layerGroup.isValid()) {
    throw MissingGroupException("Invalid group in CSparseFieldIO::read()");
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
    throw UnsupportedVersionException("CSparseField version not supported: " +
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

  // Get bit rate ---

  OgIAttribute<uint8_t> bitRateAttr = 
    layerGroup.findAttribute<uint8_t>(k_bitRateStr);
  if (!bitRateAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_bitRateStr);
  }
  bitRate = bitRateAttr.value();

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
                                 "CSparseFieldIO::read()");
  }

  // Finally, read the data ---

  FieldBase::Ptr result;
  
  OgDataType typeOnDisk = layerGroup.compressedDatasetType(k_dataTypeStr);

  if (typeEnum == typeOnDisk) {
    if (typeEnum == F3DFloat32) {
      result = readData<float32_t>(layerGroup, extents, dataW, blockOrder,
                                   numBlocks, blockRes, bitRate, 
                                   filename, layerPath);
    } else if (typeEnum == F3DVec32) {
      result = readData<vec32_t>(layerGroup, extents, dataW, blockOrder,
                                 numBlocks, blockRes,
                                 bitRate, filename, layerPath);
    } 
  }

  return result;
}

//----------------------------------------------------------------------------//

bool
CSparseFieldIO::write(hid_t layerGroup, FieldBase::Ptr field)
{
  return false;
}

//----------------------------------------------------------------------------//

bool
CSparseFieldIO::write(OgOGroup &layerGroup, FieldBase::Ptr field)
{
  using namespace Exc;

  // Add version attribute
  OgOAttribute<int> version(layerGroup, k_versionAttrName, k_versionNumber);

  CSparseField<float>::Ptr floatField = 
    field_dynamic_cast<CSparseField<float> >(field);
  CSparseField<V3f>::Ptr vecFloatField = 
    field_dynamic_cast<CSparseField<V3f> >(field);

  bool success = true;

  if (floatField) {
    success = writeInternal<float>(layerGroup, floatField);
  } else if (vecFloatField) {
    success = writeInternal<V3f>(layerGroup, vecFloatField);
  } else {
    throw WriteLayerException("CSparseFieldIO does not support the given "
                              "CSparseField template parameter");
  }

  return success;
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename CSparseField<Data_T>::Ptr
CSparseFieldIO::readData(const OgIGroup &location, const Box3i &extents, 
                         const Box3i &dataW, const size_t blockOrder, 
                         const size_t numBlocks, const V3i &blockRes,
                         const int bitRate,
                         const std::string &filename, 
                         const std::string &layerPath)
{
  using namespace std;
  using namespace Exc;
  using namespace Sparse;
  
  typename CSparseField<Data_T>::Ptr result(new CSparseField<Data_T>);

  const int components = FieldTraits<Data_T>::dataDims();
  
  // Read the block info data sets ---

  CSparseFieldImpl *impl = result->impl();

  // ... Read the block map array and set up the block mapping array
  std::vector<size_t> blockIdxToDatasetIdx(numBlocks);

  int numAllocatedBlocks = 0;

  {
    // Grab the data
    vector<int32_t> blockMap(numBlocks);
    OgIDataset<int32_t> blockMapData = 
      location.findDataset<int32_t>(k_blockMapStr);
    if (!blockMapData.isValid()) {
      throw MissingGroupException("Couldn't find block_map dataset.");
    }
    blockMapData.getData(0, &blockMap[0], OGAWA_THREAD);
    // Configure the field
    result->configure(extents, dataW, blockOrder, 
                      blockRes, bitRate, blockMap);
    // Count blocks 
    for (size_t i = 0; i < numBlocks; ++i) {
      if (impl->blockIsAllocated(i)) {
        numAllocatedBlocks++;
      } 
    }
  }

  // ... Read the emptyValue array ---

  {
    OgIDataset<float32_t> emptyValueData = 
      location.findDataset<float32_t>(k_blockEmptyValueStr);
    if (!emptyValueData.isValid()) {
      throw MissingGroupException("Couldn't find block_empty_value_data: ");
    }
    if (emptyValueData.numDataElements() > 0) {
      for (int c = 0; c < components; ++c) {
        emptyValueData.getData(c, &impl->emptyBlocks[c][0], OGAWA_THREAD);
      }
    }
  }

  // Read the data ---

  if (numAllocatedBlocks > 0) {
    // Threading state
    ReadThreadingState state(location, impl, numAllocatedBlocks);
    // Number of threads
    const size_t numThreads = numIOThreads();
    // Launch threads
    boost::thread_group threads;
    for (size_t i = 0; i < numThreads; ++i) {
      threads.create_thread(ReadBlockOp<Data_T>(state, i));
    }
    threads.join_all();
  }

  return result;
}

//----------------------------------------------------------------------------//
// Template implementations
//----------------------------------------------------------------------------//

template <class Data_T>
bool CSparseFieldIO::writeInternal(OgOGroup &layerGroup, 
                                   typename CSparseField<Data_T>::Ptr field)
{
  using namespace Exc;
  using namespace Sparse;

  CSparseFieldImpl *impl = field->impl();

  const int     components    = FieldTraits<Data_T>::dataDims();
  const int     blockOrder    = impl->blockOrder;
  const V3i    &blockRes      = impl->blockRes;
  const size_t  numBlocks     = impl->numBlocks;
  const uint8_t bitRate       = impl->bitRate;
  const Box3i   ext           = field->extents();
  const Box3i   dw            = field->dataWindow();

  // Add attributes ---

  OgOAttribute<veci32_t> extMinAttr(layerGroup, k_extentsMinStr, ext.min);
  OgOAttribute<veci32_t> extMaxAttr(layerGroup, k_extentsMaxStr, ext.max);
  
  OgOAttribute<veci32_t> dwMinAttr(layerGroup, k_dataWindowMinStr, dw.min);
  OgOAttribute<veci32_t> dwMaxAttr(layerGroup, k_dataWindowMaxStr, dw.max);

  OgOAttribute<uint8_t> blockOrderAttr(layerGroup, k_blockOrderStr, blockOrder);

  OgOAttribute<uint32_t> numBlocksAttr(layerGroup, k_numBlocksStr, numBlocks);

  OgOAttribute<veci32_t> blockResAttr(layerGroup, k_blockResStr, blockRes);

  OgOAttribute<uint8_t> componentsAttr(layerGroup, k_componentsStr, components);

  OgOAttribute<uint8_t> bitRateAttr(layerGroup, k_bitRateStr, bitRate);

  // Write the blockMap array
  OgODataset<int32_t> blockMapData(layerGroup, k_blockMapStr);
  blockMapData.addData(numBlocks, &impl->blockMap[0]);

  // Write the emptyValue array
  OgODataset<float32_t> emptyValueData(layerGroup, k_blockEmptyValueStr);
  const size_t numEmpty = impl->numEmptyBlocks();
  if (numEmpty > 0) {
    for (int c = 0; c < components; ++c) {
      emptyValueData.addData(numEmpty, &impl->emptyBlocks[c][0]);
    }
  }

  // Add data to file ---

  // Create a dummy dataset with the appropriate type
  OgOCDataset<Data_T> dataType(layerGroup, k_dataTypeStr);

  // Create the compressed dataset regardless of whether there are blocks
  // to write.
  OgOCDataset<uint8_t> data(layerGroup, k_dataStr);
  // Write data if there is any
  if (impl->numAllocatedBlocks() > 0) {
    // Threading state
    ThreadingState state(data, impl);
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

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
