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

/*! \file SparseFile.h
  \brief Contains functions controlling the loading of sparse fields.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_SparseFile_H_
#define _INCLUDED_Field3D_SparseFile_H_

//----------------------------------------------------------------------------//

#include <vector>
#include <list>

#include <hdf5.h>

#include "Exception.h"
#include "Hdf5Util.h"
#include "SparseDataReader.h"
#include "DataTypeConversion.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Forward declarations
//----------------------------------------------------------------------------//

namespace Sparse {

  template <typename Data_T>
  struct SparseBlock;

}

//----------------------------------------------------------------------------//

namespace SparseFile {

//----------------------------------------------------------------------------//
// Reference
//----------------------------------------------------------------------------//

/*! \class Reference
  Handles information about and block loading for a single SparseField 
  as referenced on disk
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class Reference 
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef std::vector<Sparse::SparseBlock<Data_T>*> BlockPtrs;

  // Public data members -------------------------------------------------------

  std::string filename;
  std::string layerPath;
  int valuesPerBlock;
  int occupiedBlocks;
 
  //! Index in file for each block
  std::vector<int> fileBlockIndices;
  //! Whether each block is loaded. We don't use bools since vector<bool> 
  //! is weird
  std::vector<int> blockLoaded;
  //! Pointers to each block. This is so we can go in and manipulate them
  //! as we please
  BlockPtrs blocks;
  //! Flags of whether the blocks have been accessed since they were
  //! last considered for deallocation by the Second-chance/Clock
  //! caching system
  std::vector<bool> blockUsed;
  //! Per-block counts of the number of times each block has been
  //! loaded, for cache statistics
  std::vector<int> loadCounts;
  //! Per-block counts of the number of current references to the
  //! blocks.  If a block's ref count is non-zero, then the block
  //! shouldn't be unloaded.
  std::vector<int> refCounts;
  //! Allocated array of mutexes, one per block, to lock each block
  //! individually, for guaranteeing thread-safe updates of the ref
  //! counts
  boost::mutex *blockMutex;

  // Ctors, dtor ---------------------------------------------------------------

  //! Constructor. Requires the filename and layer path of the field to be known
  Reference(const std::string filename, const std::string layerPath);
  ~Reference();

  //! Copy constructor. Clears ref counts and rebuilds mutex array.
  Reference(const Reference &o);

  //! Assignment operator.  Clears ref counts and rebuilds mutex array.
  Reference & operator=(const Reference &o);

  // Main methods --------------------------------------------------------------

  //! Checks if the file used by this reference is open already
  bool fileIsOpen();
  //! Sets the number of blocks used by the SparseField we're supporting
  void setNumBlocks(int numBlocks);
  //! Opens the file. This is done just before the first request to loadBlock.
  //! This is delayed so that the original file open  has closed the file and
  //! doesn't cause any Hdf5 hiccups.
  void openFile();
  //! Loads the block with the given index into memory. We don't pass in 
  //! a reference to where the data should go since this is already know in the
  //! blocks data member.
  void loadBlock(int blockIdx);
  //! Unloads the block with the given index from memory.
  void unloadBlock(int blockIdx);
  //! Increment reference count on a block, indicates the block is
  //! currently in use, so prevents it from being unloaded
  void incBlockRef(int blockIdx);
  //! Decrement reference count on a block
  void decBlockRef(int blockIdx);
  //! Returns the number of bytes used by the data in the block
  int blockSize(int blockIdx) const;
  //! Returns the total number of loads of the blocks of this file,
  //! for cache statistics
  int totalLoads() const;
  //! Returns the total number of blocks that are currently loaded,
  //! for statistics.
  int numLoadedBlocks() const;
  //! Returns the total number of blocks that were ever loaded (max 1
  //! per block, not the number of blocks), for statistics.
  int totalLoadedBlocks() const;
  //! Returns the average number of loads per accessed block in this file,
  //! for cache statistics
  float averageLoads() const;
  //! Resets counts of total block loads
  void resetCacheStatistics();

private:

  //! Holds the Hdf5 handle to the file
  hid_t m_fileHandle;
  
  //! Hold the group containing the data open for the duration of the 
  //! Reference's existence.
  Hdf5Util::H5ScopedGopen m_layerGroup;

  //! Pointer to the reader object. NULL at construction time. Created in 
  //! openFile().
  SparseDataReader<Data_T> *m_reader;

  //! Mutex to prevent two threads from modifying conflicting data
  boost::mutex m_mutex;

};

//----------------------------------------------------------------------------//
// References
//----------------------------------------------------------------------------//

class FileReferences
{
public:

  // Main methods --------------------------------------------------------------

  //! Returns a reference to the index. This is specialized so that the
  //! correct data member is accessed.
  template <class Data_T>
  Reference<Data_T>& ref(int idx);

  //! Appends a reference to the collection. This is specialized so that the
  //! correct data member is accessed.
  template <class Data_T>
  int append(const Reference<Data_T>& ref);

  //! Returns the number of file references of the corresponding collection
  template <class Data_T>
    int numRefs() const;

private:

  // Data members --------------------------------------------------------------

  std::vector<Reference<half> > m_hRefs;
  std::vector<Reference<V3h> > m_vhRefs;
  std::vector<Reference<float> > m_fRefs;
  std::vector<Reference<V3f> > m_vfRefs;
  std::vector<Reference<double> > m_dRefs;
  std::vector<Reference<V3d> > m_vdRefs;

};

//----------------------------------------------------------------------------//

class CacheBlock {
public:
  DataTypeEnum blockType;
  int refIdx;
  int blockIdx;
  CacheBlock(DataTypeEnum blockTypeIn, int refIdxIn,  int blockIdxIn) :
    blockType(blockTypeIn), refIdx(refIdxIn), blockIdx(blockIdxIn) 
  { }
};

//----------------------------------------------------------------------------//

} // namespace SparseFile

//----------------------------------------------------------------------------//
// SparseFileManager
//----------------------------------------------------------------------------//

/*! \class SparseFileManager
  \ingroup file
  Handles specifics about reading sparse fields from disk. Its primary use
  is to control sparse fields read using memory limiting (dynamic loading).
*/

//----------------------------------------------------------------------------//

class SparseFileManager
{

public:

  // typedefs ------------------------------------------------------------------

  typedef std::list<SparseFile::CacheBlock> CacheList;

  // Main methods --------------------------------------------------------------

  //! Returns a reference to the singleton instance
  static SparseFileManager &singleton();

  //! Sets whether to limit memory usage and do dynamic loading for sparse 
  //! fields.
  void setLimitMemUse(bool enabled);

  //! Returns whether to limit memory usage and do dynamic loading for sparse 
  //! fields.
  bool doLimitMemUse() const;

  //! Sets the maximum memory usage by dynamically loaded sparse fields.
  //! \note This is not yet implemented.
  void setMaxMemUse(float maxMemUse);

  //! Reclaims the specified number of bytes by deallocating unneeded blocks
  void deallocateBlocks(int bytesNeeded);

  //! Flushes the entire block cache for all files, should probably
  //! only be used for debugging
  void flushCache();

  //! Adds the newly loaded block to the cache, managed by the paging algorithm
  void addBlockToCache(DataTypeEnum blockType, int fileId, int blockIdx);

  //! Returns the id of the next cache item. This is stored in the SparseField
  //! in order to reference its fields at a later time
  template <class Data_T>
  int getNextId(const std::string filename, const std::string layerPath);

  //! Returns a reference to the Reference object with the given index
  template <class Data_T>
  SparseFile::Reference<Data_T> &reference(int index);

  //! Called by SparseField when it's about to read from a block
  template <class Data_T>
  void activateBlock(int fileId, int blockIdx);

  //! Increments the usage reference count on the specified block,
  //! to prevent it from getting unloaded while it's still in use
  template <class Data_T>
  void incBlockRef(int fileId, int blockIdx);

  //! Decrements the usage reference count on the specified block,
  //! after its value is no longer being used
  template <class Data_T>
  void decBlockRef(int fileId, int blockIdx);

  //! Returns the total number of block loads in the cache
  long long totalLoads();

  //! Returns the total number of blocks currently loaded into cache
  long long numLoadedBlocks();

  //! Returns the total number of blocks loaded (max 1 per block) into cache
  long long totalLoadedBlocks();

  //! Computes the ratio of blocks in the cache to the total number of
  //! blocks that have been loaded (including unloaded blocks)
  float cacheFractionLoaded();

  //! Computes the overall loaded-blocks-to-load ratio for cached files
  float cacheLoadsPerBlock();

  //! Computes the efficiency, the ratio of the number of blocks ever
  //! loaded to the number of loads.  If this is <1, then there were reloads.
  float cacheEfficiency();

  //! Resets block load
  void resetCacheStatistics();

private:

  //! Private to prevent instantiation
  SparseFileManager();

  //! Pointer to singleton
  static SparseFileManager *ms_singleton;

  //! Utility function to attempt to deallocate a single block and
  //! advance the "hand"
  template <class Data_T>
  int deallocateBlock(const SparseFile::CacheBlock &cb);

  //! Utility function to deallocate a single block
  template <class Data_T>
  void deallocateBlock(CacheList::iterator &it);

  //! Max amount om memory to use in megabytes
  float m_maxMemUse;

  //! Max amount om memory to use in bytes
  int m_maxMemUseInBytes;

  //! Current amount of memory in use in bytes
  int m_memUse;

  //! Whether to limit memory use of sparse fields from disk. Enables the
  //! cache and dynamic loading when true.
  bool m_limitMemUse;

  //! Vector containing information for each of the managed fields.
  //! The order matches the index stored in each SparseField::m_fileId
  SparseFile::FileReferences m_fileData;

  //! List of dynamically loaded blocks to be considered for unloading
  //! when the cache is full.
  //! Currently using Second-chance/Clock paging algorithm.
  //! For a description of the algorithm, look at:
  //! http://en.wikipedia.org/wiki/Page_replacement_algorithm#Second-chance
  CacheList m_blockCacheList;

  //! Pointer to the next block to test for unloading in the cache,
  //! the "hand" of the clock
  CacheList::iterator m_nextBlock;

  //! Mutex to prevent multiple threads from deallocating blocks at
  //! the same time
  boost::mutex m_mutex;

};

//----------------------------------------------------------------------------//
// Reference implementations
//----------------------------------------------------------------------------//

namespace SparseFile {

//----------------------------------------------------------------------------//

template <class Data_T>
Reference<Data_T>::Reference(const std::string a_filename, 
                             const std::string a_layerPath)
  : filename(a_filename), layerPath(a_layerPath),
    valuesPerBlock(-1), occupiedBlocks(-1),
    blockMutex(NULL), m_fileHandle(-1), m_reader(NULL) { 
  /* Empty */ 
}

//----------------------------------------------------------------------------//

template <class Data_T>
Reference<Data_T>::~Reference()
{
  if (m_reader)
    delete m_reader;

  if (blockMutex)
    delete [] blockMutex;
}

//----------------------------------------------------------------------------//

template <class Data_T>
Reference<Data_T>::Reference(const Reference<Data_T> &o)
{
  m_reader = NULL;
  blockMutex = NULL;
  *this = o;
}

//----------------------------------------------------------------------------//

template <class Data_T>
Reference<Data_T> &
Reference<Data_T>::operator=(const Reference<Data_T> &o)
{
  // copy public member variables (where appropriate)
  filename = o.filename;
  layerPath = o.layerPath;
  valuesPerBlock = o.valuesPerBlock;
  occupiedBlocks = o.occupiedBlocks;
  fileBlockIndices = o.fileBlockIndices;
  blockLoaded = o.blockLoaded;
  blocks = o.blocks;
  blockUsed = o.blockUsed;
  loadCounts = o.loadCounts;
  refCounts = o.refCounts;
  if (blockMutex)
    delete[] blockMutex;
  blockMutex = new boost::mutex[blocks.size()];

  // copy private member variables (where appropriate)
  m_fileHandle = o.m_fileHandle;
  m_layerGroup = o.m_layerGroup;
  if (m_reader)
    delete m_reader;
  m_reader = NULL;

  return *this;
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool Reference<Data_T>::fileIsOpen()
{
  return m_fileHandle >= 0;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void Reference<Data_T>::setNumBlocks(int numBlocks)
{
  boost::mutex::scoped_lock lock(m_mutex);

  fileBlockIndices.resize(numBlocks);
  blockLoaded.resize(numBlocks, 0);
  blocks.resize(numBlocks, 0);
  blockUsed.resize(numBlocks, false);
  loadCounts.resize(numBlocks, 0);
  refCounts.resize(numBlocks, 0);
  if (blockMutex)
    delete[] blockMutex;
  blockMutex = new boost::mutex[numBlocks];
}

//----------------------------------------------------------------------------//

template <class Data_T>
void Reference<Data_T>::openFile()
{
  using namespace Exc;
  using namespace Hdf5Util;

  boost::mutex::scoped_lock lock(m_mutex);

  // check that the file wasn't already opened before obtaining the lock
  if (fileIsOpen()) {
    return;
  }

  m_fileHandle = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (m_fileHandle < 0)
    throw NoSuchFileException(filename);

  m_layerGroup.open(m_fileHandle, layerPath.c_str());
  if (m_layerGroup.id() < 0) {
    Msg::print(Msg::SevWarning, "In SparseFile::Reference::openFile: "
              "Couldn't find layer group " + layerPath + 
              " in .f3d file ");
    throw FileIntegrityException(filename);
  }

  m_reader = new SparseDataReader<Data_T>(m_layerGroup.id(), valuesPerBlock, 
                                          occupiedBlocks);
}

//----------------------------------------------------------------------------//

template <class Data_T>
void Reference<Data_T>::loadBlock(int blockIdx)
{
  boost::mutex::scoped_lock lock(m_mutex);
  // Allocate the block
  blocks[blockIdx]->data.resize(valuesPerBlock);
  assert(blocks[blockIdx]->data.size() > 0);
  // Read the data
  assert(m_reader);
  m_reader->readBlock(fileBlockIndices[blockIdx], blocks[blockIdx]->dataRef());
  // Mark block as loaded
  blockLoaded[blockIdx] = 1;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void Reference<Data_T>::unloadBlock(int blockIdx)
{
  // Deallocate the block
  std::vector<Data_T>().swap(blocks[blockIdx]->data);
  // Mark block as unloaded
  blockLoaded[blockIdx] = 0;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void Reference<Data_T>::incBlockRef(int blockIdx)
{
  boost::mutex::scoped_lock lock(blockMutex[blockIdx]);
  ++refCounts[blockIdx];
}

//----------------------------------------------------------------------------//

template <class Data_T>
void Reference<Data_T>::decBlockRef(int blockIdx)
{
  boost::mutex::scoped_lock lock(blockMutex[blockIdx]);
  --refCounts[blockIdx];
}

//----------------------------------------------------------------------------//

template <class Data_T>
int Reference<Data_T>::blockSize(int blockIdx) const
{
  return valuesPerBlock * sizeof(Data_T);
}

//----------------------------------------------------------------------------//

template <class Data_T>
int Reference<Data_T>::totalLoads() const
{
  std::vector<int>::const_iterator i = loadCounts.begin();
  std::vector<int>::const_iterator end = loadCounts.end();
  int numLoads = 0;
  for (; i != end; ++i)
    numLoads += *i;

  return numLoads;
}

//----------------------------------------------------------------------------//

template <class Data_T>
int Reference<Data_T>::numLoadedBlocks() const
{
  std::vector<int>::const_iterator i = blockLoaded.begin();
  std::vector<int>::const_iterator end = blockLoaded.end();
  int numBlocks = 0;
  for (; i != end; ++i)
    if (*i)
      numBlocks++;

  return numBlocks;
}

//----------------------------------------------------------------------------//

template <class Data_T>
int Reference<Data_T>::totalLoadedBlocks() const
{
  std::vector<int>::const_iterator i = loadCounts.begin();
  std::vector<int>::const_iterator li = blockLoaded.begin();
  std::vector<int>::const_iterator end = loadCounts.end();
  int numBlocks = 0;
  for (; i != end; ++i, ++li)
    if (*i || *li)
      numBlocks++;

  return numBlocks;
}

//----------------------------------------------------------------------------//

template <class Data_T>
float Reference<Data_T>::averageLoads() const
{
  std::vector<int>::const_iterator i = loadCounts.begin();
  std::vector<int>::const_iterator end = loadCounts.end();
  int numLoads = 0, numBlocks = 0;
  for (; i != end; ++i) {
    if (*i) {
      numLoads += *i;
      numBlocks++;
    }
  }

  return (float)numLoads / std::max(1, numBlocks);
}

//----------------------------------------------------------------------------//

template <class Data_T>
void Reference<Data_T>::resetCacheStatistics()
{
  std::vector<int>::iterator li = loadCounts.begin();
  std::vector<int>::iterator lend = loadCounts.end();
  for (; li != lend; ++li)
    *li = 0;
}

//----------------------------------------------------------------------------//

} // namespace SparseFile

//----------------------------------------------------------------------------//
// Specializations for FileReferences
//----------------------------------------------------------------------------//

namespace SparseFile {

//----------------------------------------------------------------------------//

template <>
inline Reference<half>& 
FileReferences::ref(int idx)
{
  return m_hRefs[idx];
}

//----------------------------------------------------------------------------//

template <>
inline Reference<V3h>& 
FileReferences::ref(int idx)
{
  return m_vhRefs[idx];
}

//----------------------------------------------------------------------------//

template <>
inline Reference<float>& 
FileReferences::ref(int idx)
{
  return m_fRefs[idx];
}

//----------------------------------------------------------------------------//

template <>
inline Reference<V3f>& 
FileReferences::ref(int idx)
{
  return m_vfRefs[idx];
}

//----------------------------------------------------------------------------//

template <>
inline Reference<double>& 
FileReferences::ref(int idx)
{
  return m_dRefs[idx];
}

//----------------------------------------------------------------------------//

template <>
inline Reference<V3d>& 
FileReferences::ref(int idx)
{
  return m_vdRefs[idx];
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::append(const Reference<half>& ref)
{
  m_hRefs.push_back(ref);
  return m_hRefs.size() - 1;
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::append(const Reference<V3h>& ref)
{
  m_vhRefs.push_back(ref);
  return m_vhRefs.size() - 1;
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::append(const Reference<float>& ref)
{
  m_fRefs.push_back(ref);
  return m_fRefs.size() - 1;
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::append(const Reference<V3f>& ref)
{
  m_vfRefs.push_back(ref);
  return m_vfRefs.size() - 1;
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::append(const Reference<double>& ref)
{
  m_dRefs.push_back(ref);
  return m_dRefs.size() - 1;
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::append(const Reference<V3d>& ref)
{
  m_vdRefs.push_back(ref);
  return m_vdRefs.size() - 1;
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::numRefs<half>() const
{
  return m_hRefs.size();
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::numRefs<V3h>() const
{
  return m_vhRefs.size();
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::numRefs<float>() const
{
  return m_fRefs.size();
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::numRefs<V3f>() const
{
  return m_vfRefs.size();
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::numRefs<double>() const
{
  return m_dRefs.size();
}

//----------------------------------------------------------------------------//

template <>
inline int FileReferences::numRefs<V3d>() const
{
  return m_vdRefs.size();
}

//----------------------------------------------------------------------------//
// Implementations for FileReferences
//----------------------------------------------------------------------------//

template <class Data_T>
Reference<Data_T>& FileReferences::ref(int idx)
{
  assert(false && "Do not use memory limiting on sparse fields that aren't "
         "simple scalars or vectors!");
  Msg::print(Msg::SevWarning, 
             "FileReferences::ref(): Do not use memory limiting on sparse "
             "fields that aren't simple scalars or vectors!");
  static Reference<Data_T> dummy("", "");
  return dummy;
}

//----------------------------------------------------------------------------//

template <class Data_T>
int FileReferences::append(const Reference<Data_T>& ref)
{
  assert(false && "Do not use memory limiting on sparse fields that aren't "
         "simple scalars or vectors!");
  Msg::print(Msg::SevWarning,
             "FileReferences::append(): Do not use memory limiting on sparse "
             "fields that aren't simple scalars or vectors!");
  return -1;    
}

//----------------------------------------------------------------------------//

template <class Data_T>
int FileReferences::numRefs() const
{
  assert(false && "Do not use memory limiting on sparse fields that aren't "
         "simple scalars or vectors!");
  Msg::print(Msg::SevWarning,
             "FileReferences::numRefs(): "
             "Do not use memory limiting on sparse "
             "fields that aren't "
             "simple scalars or vectors!");
  return -1;
}

//----------------------------------------------------------------------------//

} // namespace SparseFile

//----------------------------------------------------------------------------//
// SparseFileManager implementations
//----------------------------------------------------------------------------//

template <class Data_T>
int 
SparseFileManager::getNextId(const std::string filename, 
                             const std::string layerPath)
{
  using namespace SparseFile;

  int id = m_fileData.append(Reference<Data_T>(filename, layerPath));
  return id;
}

//----------------------------------------------------------------------------//

template <class Data_T>
SparseFile::Reference<Data_T> &
SparseFileManager::reference(int index)
{ 
  return m_fileData.ref<Data_T>(index); 
}

//----------------------------------------------------------------------------//

template <class Data_T>
void 
SparseFileManager::activateBlock(int fileId, int blockIdx)
{
  SparseFile::Reference<Data_T> &reference = m_fileData.ref<Data_T>(fileId);

  if (reference.fileBlockIndices[blockIdx] >= 0) {
    if (!reference.blockLoaded[blockIdx]) {
      int blockSize = reference.blockSize(blockIdx);
      if (m_limitMemUse) {
        // if we already have enough free memory, deallocateBlocks()
        // will just return
        deallocateBlocks(blockSize);
      }
      if (!reference.fileIsOpen()) {
        reference.openFile();
      }
      boost::mutex::scoped_lock lock(reference.blockMutex[blockIdx]);
      // check to see if it was loaded between when the function
      // started and we got the lock on the block
      if (!reference.blockLoaded[blockIdx]) {
        reference.loadBlock(blockIdx);
        reference.loadCounts[blockIdx]++;
        addBlockToCache(dataTypeToEnum<Data_T>(), fileId, blockIdx);
        m_memUse += blockSize;
      }
    }
  }
  reference.blockUsed[blockIdx] = true;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void 
SparseFileManager::incBlockRef(int fileId, int blockIdx)
{
  SparseFile::Reference<Data_T> &reference = m_fileData.ref<Data_T>(fileId);

  if (reference.fileBlockIndices[blockIdx] >= 0) {
    reference.incBlockRef(blockIdx);
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
void 
SparseFileManager::decBlockRef(int fileId, int blockIdx)
{
  SparseFile::Reference<Data_T> &reference = m_fileData.ref<Data_T>(fileId);

  if (reference.fileBlockIndices[blockIdx] >= 0) {
    reference.decBlockRef(blockIdx);
  }
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
