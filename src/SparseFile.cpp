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

/*! \file SparseFile.cpp
  \brief Contains implementations relating to reading of sparse field files.
*/

//----------------------------------------------------------------------------//

// SparseField.h includes SparseFile.h, but we need the definition of
// SparseBlock from SparseField.h, so just include that to get both
// files
#include "SparseField.h"

#include "OgIO.h"
#include "OgSparseDataReader.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Static instances
//----------------------------------------------------------------------------//

SparseFileManager *SparseFileManager::ms_singleton = 0;

//----------------------------------------------------------------------------//
// SparseFileManager
//----------------------------------------------------------------------------//

SparseFileManager & SparseFileManager::singleton()
{ 
  if (!ms_singleton) {
    ms_singleton = new SparseFileManager;
  }
  return *ms_singleton;
}

//----------------------------------------------------------------------------//

void SparseFileManager::setLimitMemUse(bool enabled) 
{
  m_limitMemUse = enabled;
}

//----------------------------------------------------------------------------//

bool SparseFileManager::doLimitMemUse() const
{ 
  return m_limitMemUse; 
}

//----------------------------------------------------------------------------//

void SparseFileManager::setMaxMemUse(float maxMemUse) 
{
  m_maxMemUse = maxMemUse;
  m_maxMemUseInBytes = static_cast<int64_t>(m_maxMemUse * 1024 * 1024);
}

//----------------------------------------------------------------------------//

template <class Data_T>
int64_t SparseFileManager::deallocateBlock(const SparseFile::CacheBlock &cb)
{
  int64_t bytesFreed = 0;
  SparseFile::Reference<Data_T> *reference = m_fileData.ref<Data_T>(cb.refIdx);

  // Note: we don't need to lock the block's mutex because
  // deallocateBlock() is only called while the SparseFileManager's
  // mutex is also locked (in flushCache() or deallocateBlocks()).
  // Don't lock the block, to make sure we don't have a deadlock by
  // holding two locks at the same time.  (Because addBlockToCache()
  // locks the manager but is also in a block-specific lock.)

  // lock the current block to make sure its blockUsed flag and ref
  // counts don't change
  // Note: this lock order is made consistent w/ allocate to prevent
  // deadlocks and crashes.

#if F3D_SHORT_MUTEX_ARRAY
  boost::mutex::scoped_lock 
    lock_B(reference->blockMutex[cb.blockIdx % reference->blockMutexSize]);
#else
  boost::mutex::scoped_lock lock_B(reference->blockMutex[cb.blockIdx]);
#endif
  
  // check whether the block is still in use
  if (reference->refCounts[cb.blockIdx] > 0)
    return bytesFreed;

  if (reference->blockUsed[cb.blockIdx]) {
    // the block was recently used according to Second-chance paging
    // algorithm, so skip it
    reference->blockUsed[cb.blockIdx] = false;
  }
  else {

    // the block wasn't in use, so free it
    reference->unloadBlock(cb.blockIdx);
    bytesFreed = reference->blockSize(cb.blockIdx);
    m_memUse -= bytesFreed;
    CacheList::iterator toRemove = m_nextBlock;
    ++m_nextBlock;
    m_blockCacheList.erase(toRemove);
  }
  return bytesFreed;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void SparseFileManager::deallocateBlock(CacheList::iterator &it)
{
  SparseFile::CacheBlock &cb = *it;
  SparseFile::Reference<Data_T> *reference = m_fileData.ref<Data_T>(cb.refIdx);
  int64_t bytesFreed = reference->blockSize(cb.blockIdx);
  m_memUse -= bytesFreed;
  reference->unloadBlock(cb.blockIdx);
  it = m_blockCacheList.erase(it);
}

//----------------------------------------------------------------------------//

void SparseFileManager::deallocateBlocks(int64_t bytesNeeded)
{
  boost::mutex::scoped_lock lock_A(m_mutex);

  while (m_blockCacheList.begin() != m_blockCacheList.end() &&
         m_maxMemUseInBytes-m_memUse < bytesNeeded) {

    if (m_nextBlock == m_blockCacheList.end())
      m_nextBlock = m_blockCacheList.begin();

    SparseFile::CacheBlock &cb = *m_nextBlock;

    // if bytesFreed is set to >0, then we've already freed a block
    // and advanced the "clock hand" iterator
    int64_t bytesFreed = 0;

    switch(cb.blockType) {
    case DataTypeHalf:
      bytesFreed = deallocateBlock<half>(cb);
      if (bytesFreed > 0) {
        continue;
      }
      break;
    case DataTypeFloat:
      bytesFreed = deallocateBlock<float>(cb);
      if (bytesFreed > 0) {
        continue;
      }
      break;
    case DataTypeDouble:
      bytesFreed = deallocateBlock<double>(cb);
      if (bytesFreed > 0) {
        continue;
      }
      break;
    case DataTypeVecHalf:
      bytesFreed = deallocateBlock<V3h>(cb);
      if (bytesFreed > 0) {
        continue;
      }
      break;
    case DataTypeVecFloat:
      bytesFreed = deallocateBlock<V3f>(cb);
      if (bytesFreed > 0) {
        continue;
      }
      break;
    case DataTypeVecDouble:
      bytesFreed = deallocateBlock<V3d>(cb);
      if (bytesFreed > 0) {
        continue;
      }
      break;
    case DataTypeUnknown:
    default:
      break;
    }
    ++m_nextBlock;
  }
}

//----------------------------------------------------------------------------//

void SparseFileManager::flushCache()
{
  boost::mutex::scoped_lock lock(m_mutex);

  CacheList::iterator it = m_blockCacheList.begin();
  while (it != m_blockCacheList.end()) {
    SparseFile::CacheBlock &cb = *it;

    switch(cb.blockType) {
    case DataTypeHalf:
      deallocateBlock<half>(it);
      break;
    case DataTypeFloat:
      deallocateBlock<float>(it);
      break;
    case DataTypeDouble:
      deallocateBlock<double>(it);
      break;
    case DataTypeVecHalf:
      deallocateBlock<V3h>(it);
      break;
    case DataTypeVecFloat:
      deallocateBlock<V3f>(it);
      break;
    case DataTypeVecDouble:
      deallocateBlock<V3d>(it);
      break;
    case DataTypeUnknown:
    default:
      break;
    }
  }
  m_nextBlock = m_blockCacheList.begin();
}

//----------------------------------------------------------------------------//

void SparseFileManager::addBlockToCache(DataTypeEnum blockType,
                                        int fileId, int blockIdx)
{
  // Note: this lock is obtained while we also have a lock on the
  // specific block (in activateBlock()), so we should make sure we
  // never lock the SparseFileManager and *then* a block, to ensure we
  // don't have a deadlock.
  //
  //  Note: this was changed so the order was consistent w/ dealloc
  //  again, see activateBlock()
  //  boost::mutex::scoped_lock lock(m_mutex);

  SparseFile::CacheBlock block(blockType, fileId, blockIdx);
  if (m_nextBlock == m_blockCacheList.end()) {
    m_blockCacheList.push_back(block);
  } else {
    m_blockCacheList.insert(m_nextBlock, block);
  }
}

//----------------------------------------------------------------------------//

SparseFileManager::SparseFileManager()
  : m_memUse(0),
    m_limitMemUse(false)
{
  setMaxMemUse(1000.0);
  m_nextBlock = m_blockCacheList.begin();
}

//----------------------------------------------------------------------------//

long long SparseFileManager::totalLoads()
{

  long long int numLoads = 0;

  for (size_t i=0; i<m_fileData.numRefs<half>(); i++) {
    numLoads += m_fileData.ref<half>(i)->totalLoads();
  }

  for (size_t i=0; i<m_fileData.numRefs<V3h>(); i++) {
    numLoads += m_fileData.ref<V3h>(i)->totalLoads();
  }

  for (size_t i=0; i<m_fileData.numRefs<float>(); i++) {
    numLoads += m_fileData.ref<float>(i)->totalLoads();
  }

  for (size_t i=0; i<m_fileData.numRefs<V3f>(); i++) {
    numLoads += m_fileData.ref<V3f>(i)->totalLoads();
  }

  for (size_t i=0; i<m_fileData.numRefs<double>(); i++) {
    numLoads += m_fileData.ref<double>(i)->totalLoads();
  }

  for (size_t i=0; i<m_fileData.numRefs<V3d>(); i++) {
    numLoads += m_fileData.ref<V3d>(i)->totalLoads();
  }
  return numLoads;
}

//----------------------------------------------------------------------------//

long long SparseFileManager::numLoadedBlocks()
{

  long long int numBlocks = 0;

  for (size_t i=0; i<m_fileData.numRefs<half>(); i++) {
    numBlocks += m_fileData.ref<half>(i)->numLoadedBlocks();
  }

  for (size_t i=0; i<m_fileData.numRefs<V3h>(); i++) {
    numBlocks += m_fileData.ref<V3h>(i)->numLoadedBlocks();
  }

  for (size_t i=0; i<m_fileData.numRefs<float>(); i++) {
    numBlocks += m_fileData.ref<float>(i)->numLoadedBlocks();
  }

  for (size_t i=0; i<m_fileData.numRefs<V3f>(); i++) {
    numBlocks += m_fileData.ref<V3f>(i)->numLoadedBlocks();
  }

  for (size_t i=0; i<m_fileData.numRefs<double>(); i++) {
    numBlocks += m_fileData.ref<double>(i)->numLoadedBlocks();
  }

  for (size_t i=0; i<m_fileData.numRefs<V3d>(); i++) {
    numBlocks += m_fileData.ref<V3d>(i)->numLoadedBlocks();
  }
  return numBlocks;
}

//----------------------------------------------------------------------------//

long long SparseFileManager::totalLoadedBlocks()
{

  long long int numBlocks = 0;

  for (size_t i=0; i<m_fileData.numRefs<half>(); i++) {
    numBlocks += m_fileData.ref<half>(i)->totalLoadedBlocks();
  }

  for (size_t i=0; i<m_fileData.numRefs<V3h>(); i++) {
    numBlocks += m_fileData.ref<V3h>(i)->totalLoadedBlocks();
  }

  for (size_t i=0; i<m_fileData.numRefs<float>(); i++) {
    numBlocks += m_fileData.ref<float>(i)->totalLoadedBlocks();
  }

  for (size_t i=0; i<m_fileData.numRefs<V3f>(); i++) {
    numBlocks += m_fileData.ref<V3f>(i)->totalLoadedBlocks();
  }

  for (size_t i=0; i<m_fileData.numRefs<double>(); i++) {
    numBlocks += m_fileData.ref<double>(i)->totalLoadedBlocks();
  }

  for (size_t i=0; i<m_fileData.numRefs<V3d>(); i++) {
    numBlocks += m_fileData.ref<V3d>(i)->totalLoadedBlocks();
  }
  return numBlocks;
}

//----------------------------------------------------------------------------//

float SparseFileManager::cacheFractionLoaded()
{
  return ((double)numLoadedBlocks())/std::max(1.0, ((double)totalLoadedBlocks()));
}

//----------------------------------------------------------------------------//

float SparseFileManager::cacheLoadsPerBlock()
{
  return ((double)totalLoads())/std::max(1.0, ((double)totalLoadedBlocks()));
}

//----------------------------------------------------------------------------//

float SparseFileManager::cacheEfficiency()
{
  return ((double)totalLoadedBlocks())/std::max(1.0, ((double)totalLoads()));
}

//----------------------------------------------------------------------------//

void SparseFileManager::resetCacheStatistics()
{

  for (size_t i=0; i<m_fileData.numRefs<half>(); i++) {
    m_fileData.ref<half>(i)->resetCacheStatistics();
  }

  for (size_t i=0; i<m_fileData.numRefs<V3h>(); i++) {
    m_fileData.ref<V3h>(i)->resetCacheStatistics();
  }

  for (size_t i=0; i<m_fileData.numRefs<float>(); i++) {
    m_fileData.ref<float>(i)->resetCacheStatistics();
  }

  for (size_t i=0; i<m_fileData.numRefs<V3f>(); i++) {
    m_fileData.ref<V3f>(i)->resetCacheStatistics();
  }

  for (size_t i=0; i<m_fileData.numRefs<double>(); i++) {
    m_fileData.ref<double>(i)->resetCacheStatistics();
  }

  for (size_t i=0; i<m_fileData.numRefs<V3d>(); i++) {
    m_fileData.ref<V3d>(i)->resetCacheStatistics();
  }
}

//----------------------------------------------------------------------------//

long long int SparseFileManager::memSize() const
{
  boost::mutex::scoped_lock lock(m_mutex);

  return sizeof(*this) + m_fileData.memSize() + 
    m_blockCacheList.size() * sizeof(SparseFile::CacheBlock);
}

//----------------------------------------------------------------------------//

long long int SparseFile::FileReferences::memSize() const 
{
  Mutex::scoped_lock lock(m_mutex);

  long long int size = 0;

  // Size of the std::deque's
  size += m_hRefs.size() * sizeof(Reference<half>::Ptr);
  size += m_vhRefs.size() * sizeof(Reference<V3h>::Ptr);
  size += m_fRefs.size() * sizeof(Reference<float>::Ptr);
  size += m_vfRefs.size() * sizeof(Reference<V3f>::Ptr);
  size += m_dRefs.size() * sizeof(Reference<double>::Ptr);
  size += m_vdRefs.size() * sizeof(Reference<V3d>::Ptr);

  // Size of the references themselves
  for (size_t i = 0, end = m_hRefs.size(); i < end; ++i) {
    size += m_hRefs[i]->memSize();
  }
  for (size_t i = 0, end = m_vhRefs.size(); i < end; ++i) {
    size += m_vhRefs[i]->memSize();
  }
  for (size_t i = 0, end = m_fRefs.size(); i < end; ++i) {
    size += m_fRefs[i]->memSize();
  }
  for (size_t i = 0, end = m_vfRefs.size(); i < end; ++i) {
    size += m_vfRefs[i]->memSize();
  }
  for (size_t i = 0, end = m_dRefs.size(); i < end; ++i) {
    size += m_dRefs[i]->memSize();
  }
  for (size_t i = 0, end = m_vdRefs.size(); i < end; ++i) {
    size += m_vdRefs[i]->memSize();
  }
  
  return size;
}

//----------------------------------------------------------------------------//
// Template implementations
//----------------------------------------------------------------------------//

namespace SparseFile {

//----------------------------------------------------------------------------//

template <class Data_T>
void Reference<Data_T>::loadBlock(int blockIdx)
{
  boost::mutex::scoped_lock lock(m_mutex);

  // Allocate the block
#if F3D_NO_BLOCKS_ARRAY
  blocks[blockIdx].resize(numVoxels);
  assert(blocks[blockIdx].data != NULL);
  // Read the data
  assert(m_reader || m_ogReader);
  if (m_reader) {
    m_reader->readBlock(fileBlockIndices[blockIdx], *blocks[blockIdx].data);
  } else {
    m_ogReader->readBlock(fileBlockIndices[blockIdx], blocks[blockIdx].data);
  }
  // Mark block as loaded
  blockLoaded[blockIdx] = 1;
  // Track count
  m_numActiveBlocks++;
#else
  blocks[blockIdx]->resize(numVoxels);
  assert(blocks[blockIdx]->data != NULL);
  // Read the data
  assert(m_reader || m_ogReader);
  if (m_reader) {
    m_reader->readBlock(fileBlockIndices[blockIdx], *blocks[blockIdx]->data);
  } else {
    m_ogReader->readBlock(fileBlockIndices[blockIdx], blocks[blockIdx]->data);
  }
  // Mark block as loaded
  blockLoaded[blockIdx] = 1;
  // Track count
  m_numActiveBlocks++;
#endif
}

//----------------------------------------------------------------------------//

template <class Data_T>
void Reference<Data_T>::openFile()
{
  using namespace Exc;
  using namespace Hdf5Util;

  boost::mutex::scoped_lock lock_A(m_mutex);

  // check that the file wasn't already opened before obtaining the lock
  if (fileIsOpen()) {
    return;
  }

  // First try Ogawa ---

  m_ogArchive.reset(new Alembic::Ogawa::IArchive(filename));
  if (m_ogArchive->isValid()) {
    m_ogRoot.reset(new OgIGroup(*m_ogArchive));
    m_ogLayerGroup.reset(new OgIGroup(m_ogRoot->findGroup(layerPath)));
    if (m_ogLayerGroup->isValid()) {
      // Allocate the reader
      m_ogReaderPtr.reset(new OgSparseDataReader<Data_T>(*m_ogLayerGroup,
                                                         numVoxels,
                                                         occupiedBlocks,
                                                         true));
      m_ogReader = m_ogReaderPtr.get();
      // Done
      return;
    }
  }

  // Then, try HDF5 ---

  {
    // Hold the global lock
    GlobalLock lock(g_hdf5Mutex);
    // Open the file
    m_fileHandle = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (m_fileHandle >= 0) {
      // Open the layer group
      m_layerGroup.open(m_fileHandle, layerPath.c_str());
      if (m_layerGroup.id() < 0) {
        Msg::print(Msg::SevWarning, "In SparseFile::Reference::openFile: "
                   "Couldn't find layer group " + layerPath + 
                   " in .f3d file ");
        throw FileIntegrityException(filename);
      }
    } else {
      Msg::print(Msg::SevWarning, "In SparseFile::Reference::openFile: "
                 "Couldn't open HDF5 file ");
      throw NoSuchFileException(filename);
    }
  }

  // Re-allocate reader
  if (m_reader) {
    delete m_reader;
  }
  m_reader = new SparseDataReader<Data_T>(m_layerGroup.id(), 
                                          valuesPerBlock, 
                                          occupiedBlocks);
}

//----------------------------------------------------------------------------//

#define FIELD3D_INSTANTIATION_LOADBLOCK(type)                       \
  template                                                          \
  void Reference<type>::loadBlock(int blockIdx);                    \
  
FIELD3D_INSTANTIATION_LOADBLOCK(float16_t);
FIELD3D_INSTANTIATION_LOADBLOCK(float32_t);
FIELD3D_INSTANTIATION_LOADBLOCK(float64_t);
FIELD3D_INSTANTIATION_LOADBLOCK(vec16_t);
FIELD3D_INSTANTIATION_LOADBLOCK(vec32_t);
FIELD3D_INSTANTIATION_LOADBLOCK(vec64_t);

//----------------------------------------------------------------------------//

#define FIELD3D_INSTANTIATION_OPENFILE(type)                        \
  template                                                          \
  void Reference<type>::openFile();                                 \
  
FIELD3D_INSTANTIATION_OPENFILE(float16_t);
FIELD3D_INSTANTIATION_OPENFILE(float32_t);
FIELD3D_INSTANTIATION_OPENFILE(float64_t);
FIELD3D_INSTANTIATION_OPENFILE(vec16_t);
FIELD3D_INSTANTIATION_OPENFILE(vec32_t);
FIELD3D_INSTANTIATION_OPENFILE(vec64_t);

} // namespace SparseFile

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

