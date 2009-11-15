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

/*! \file SparseFile.cpp
  \brief Contains implementations relating to reading of sparse field files.
*/

//----------------------------------------------------------------------------//

// SparseField.h includes SparseFile.h, but we need the definition of
// SparseBlock from SparseField.h, so just include that to get both
// files
#include "SparseField.h"

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
  m_maxMemUseInBytes = static_cast<int>(m_maxMemUse * 1024*1024);
}

//----------------------------------------------------------------------------//

template <class Data_T>
int SparseFileManager::deallocateBlock(const SparseFile::CacheBlock &cb)
{
  int bytesFreed = 0;
  SparseFile::Reference<Data_T> &reference = m_fileData.ref<Data_T>(cb.refIdx);


  // Note: we don't need to lock the block's mutex because
  // deallocateBlock() is only called while the SparseFileManager's
  // mutex is also locked (in flushCache() or deallocateBlocks()).
  // Don't lock the block, to make sure we don't have a deadlock by
  // holding two locks at the same time.  (Because addBlockToCache()
  // locks the manager but is also in a block-specific lock.)

  // lock the current block to make sure its blockUsed flag and ref
  // counts don't change
  //boost::mutex::scoped_lock lock(reference.blockMutex[cb.blockIdx]);

  // check whether the block is still in use
  if (reference.refCounts[cb.blockIdx] > 0)
    return bytesFreed;

  if (reference.blockUsed[cb.blockIdx]) {
    // the block was recently used according to Second-chance paging
    // algorithm, so skip it
    reference.blockUsed[cb.blockIdx] = false;
  }
  else {
    // the block wasn't in use, so free it
    reference.unloadBlock(cb.blockIdx);
    bytesFreed = reference.blockSize(cb.blockIdx);
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
  SparseFile::Reference<Data_T> &reference = m_fileData.ref<Data_T>(cb.refIdx);
  int bytesFreed = reference.blockSize(cb.blockIdx);
  m_memUse -= bytesFreed;
  reference.unloadBlock(cb.blockIdx);
  it = m_blockCacheList.erase(it);
}

//----------------------------------------------------------------------------//

void SparseFileManager::deallocateBlocks(int bytesNeeded)
{
  boost::mutex::scoped_lock lock(m_mutex);

  while (m_blockCacheList.begin() != m_blockCacheList.end() &&
         m_maxMemUseInBytes-m_memUse < bytesNeeded) {
    if (m_nextBlock == m_blockCacheList.end())
      m_nextBlock = m_blockCacheList.begin();
    SparseFile::CacheBlock &cb = *m_nextBlock;

    // if bytesFreed is set to >0, then we've already freed a block
    // and advanced the "clock hand" iterator
    int bytesFreed = 0;

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
  boost::mutex::scoped_lock lock(m_mutex);

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

  for (int i=0; i<m_fileData.numRefs<half>(); i++) {
    numLoads += m_fileData.ref<half>(i).totalLoads();
  }

  for (int i=0; i<m_fileData.numRefs<V3h>(); i++) {
    numLoads += m_fileData.ref<V3h>(i).totalLoads();
  }

  for (int i=0; i<m_fileData.numRefs<float>(); i++) {
    numLoads += m_fileData.ref<float>(i).totalLoads();
  }

  for (int i=0; i<m_fileData.numRefs<V3f>(); i++) {
    numLoads += m_fileData.ref<V3f>(i).totalLoads();
  }

  for (int i=0; i<m_fileData.numRefs<double>(); i++) {
    numLoads += m_fileData.ref<double>(i).totalLoads();
  }

  for (int i=0; i<m_fileData.numRefs<V3d>(); i++) {
    numLoads += m_fileData.ref<V3d>(i).totalLoads();
  }
  return numLoads;
}

//----------------------------------------------------------------------------//

long long SparseFileManager::numLoadedBlocks()
{

  long long int numBlocks = 0;

  for (int i=0; i<m_fileData.numRefs<half>(); i++) {
    numBlocks += m_fileData.ref<half>(i).numLoadedBlocks();
  }

  for (int i=0; i<m_fileData.numRefs<V3h>(); i++) {
    numBlocks += m_fileData.ref<V3h>(i).numLoadedBlocks();
  }

  for (int i=0; i<m_fileData.numRefs<float>(); i++) {
    numBlocks += m_fileData.ref<float>(i).numLoadedBlocks();
  }

  for (int i=0; i<m_fileData.numRefs<V3f>(); i++) {
    numBlocks += m_fileData.ref<V3f>(i).numLoadedBlocks();
  }

  for (int i=0; i<m_fileData.numRefs<double>(); i++) {
    numBlocks += m_fileData.ref<double>(i).numLoadedBlocks();
  }

  for (int i=0; i<m_fileData.numRefs<V3d>(); i++) {
    numBlocks += m_fileData.ref<V3d>(i).numLoadedBlocks();
  }
  return numBlocks;
}

//----------------------------------------------------------------------------//

long long SparseFileManager::totalLoadedBlocks()
{

  long long int numBlocks = 0;

  for (int i=0; i<m_fileData.numRefs<half>(); i++) {
    numBlocks += m_fileData.ref<half>(i).totalLoadedBlocks();
  }

  for (int i=0; i<m_fileData.numRefs<V3h>(); i++) {
    numBlocks += m_fileData.ref<V3h>(i).totalLoadedBlocks();
  }

  for (int i=0; i<m_fileData.numRefs<float>(); i++) {
    numBlocks += m_fileData.ref<float>(i).totalLoadedBlocks();
  }

  for (int i=0; i<m_fileData.numRefs<V3f>(); i++) {
    numBlocks += m_fileData.ref<V3f>(i).totalLoadedBlocks();
  }

  for (int i=0; i<m_fileData.numRefs<double>(); i++) {
    numBlocks += m_fileData.ref<double>(i).totalLoadedBlocks();
  }

  for (int i=0; i<m_fileData.numRefs<V3d>(); i++) {
    numBlocks += m_fileData.ref<V3d>(i).totalLoadedBlocks();
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

  for (int i=0; i<m_fileData.numRefs<half>(); i++) {
    m_fileData.ref<half>(i).resetCacheStatistics();
  }

  for (int i=0; i<m_fileData.numRefs<V3h>(); i++) {
    m_fileData.ref<V3h>(i).resetCacheStatistics();
  }

  for (int i=0; i<m_fileData.numRefs<float>(); i++) {
    m_fileData.ref<float>(i).resetCacheStatistics();
  }

  for (int i=0; i<m_fileData.numRefs<V3f>(); i++) {
    m_fileData.ref<V3f>(i).resetCacheStatistics();
  }

  for (int i=0; i<m_fileData.numRefs<double>(); i++) {
    m_fileData.ref<double>(i).resetCacheStatistics();
  }

  for (int i=0; i<m_fileData.numRefs<V3d>(); i++) {
    m_fileData.ref<V3d>(i).resetCacheStatistics();
  }
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

