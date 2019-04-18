//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2018 Pixar Animation Studios
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

/*! \file CSparseField.cpp
  \brief Contains the CSparseField (Compressed SparseField) implementation
*/

//----------------------------------------------------------------------------//

#include <zfp/zfparray3.h>

#include "CSparseField.h"
#include "CSparseFieldImpl.h"

//----------------------------------------------------------------------------//

#define DEFAULT_CACHE_SIZE 512

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// CSparseFieldImpl implementations
//----------------------------------------------------------------------------//

CSparseFieldImpl::CSparseFieldImpl()
  : blockOrder(BLOCK_ORDER),
    blockRes(0),
    blockXYSize(0),
    numBlocks(0)
{
  
}

//----------------------------------------------------------------------------//

float CSparseFieldImpl::value(const int i, const int j, const int k, 
                              const int layer) const
{
  // Find block coord
  int bi, bj, bk;
  Sparse::getBlockCoord(i, j, k, bi, bj, bk, blockOrder);
  // Get the actual block
  const int id = Sparse::blockId(bi, bj, bk, blockRes.x, blockXYSize);
  const int bIdx = blockMap[id];
  if (bIdx >= 0) {
    // Find coord in block
    int vi, vj, vk;
    Sparse::getVoxelInBlock(i, j, k, vi, vj, vk, blockOrder);
    return blocks[layer][bIdx](vi, vj, vk);
  } else {
    return emptyBlocks[layer][-bIdx];
  }
  
}

//----------------------------------------------------------------------------//

template <typename Data_T>
bool CSparseFieldImpl::configureFrom(const SparseField<Data_T> &field)
{
  const int dataDims = FieldTraits<Data_T>::dataDims();

  // Set block properties
  blockOrder = field.blockOrder();
  blockRes = field.blockRes();
  blockXYSize = blockRes.x * blockRes.y;

  // Number of total blocks
  numBlocks = blockRes.x * blockRes.y * blockRes.z;  
  blockMap.resize(numBlocks);

  // Scalar or vector data
  blocks.resize(dataDims);
  emptyBlocks.resize(dataDims);

  // Create block index map
  int idx = 0;
  int emptyBlockIdx = -1;
  int blockMapIdx = 0;
  for (int bk = 0; bk < blockRes.z; ++bk) {
    for (int bj = 0; bj < blockRes.y; ++bj) {
      for (int bi = 0; bi < blockRes.x; ++bi, ++idx) {
        if (field.blockIsAllocated(bi, bj, bk)) {
          blockMap[idx] = blockMapIdx++;
        } else {
          blockMap[idx] = emptyBlockIdx--;
        }
      }
    }
  }

  // Resize arrays to number of allocated and unallocated blocks
  for (int i = 0; i < dataDims; ++i) {
    blocks[i].resize(blockMapIdx);
    emptyBlocks[i].resize(-emptyBlockIdx);
  }

  return true;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void CSparseFieldImpl::configure(const int a_blockOrder,
                                 const V3i &a_blockRes,
                                 const int a_bitRate,
                                 const std::vector<int> &a_blockMap)
{
  const int dataDims = FieldTraits<Data_T>::dataDims();

  // Block properties
  blockOrder  = a_blockOrder;
  blockRes    = a_blockRes;
  blockXYSize = blockRes.x * blockRes.y;
  bitRate     = a_bitRate;

  // Size of a block
  const int blockSize = 1 << blockOrder;

  // Number of total blocks
  numBlocks = blockRes.x * blockRes.y * blockRes.z;  
  blockMap.resize(numBlocks);

  // Scalar or vector data
  blocks.resize(dataDims);
  emptyBlocks.resize(dataDims);

  // Copy the block map
  blockMap = a_blockMap;

  // Create block index map
  int numAllocated = 0;
  int numEmpty = 0;
  for (int bIdx = 0; bIdx < numBlocks; ++bIdx) {
    if (blockMap[bIdx] >= 0) {
      numAllocated++;
    } else {
      numEmpty++;
    }
  }

  // Resize arrays to number of allocated and unallocated blocks
  for (int i = 0; i < dataDims; ++i) {
    // Allocate blocks
    blocks[i].resize(numAllocated);
    // Create ZFP arrays
    for (int b = 0; b < numAllocated; ++b) {
      blocks[i][b].set_rate(a_bitRate);
      blocks[i][b].resize(blockSize, blockSize, blockSize);
      blocks[i][b].set_cache_size(DEFAULT_CACHE_SIZE);
    }
    // Make space for empty block values (add one because first will have
    // index -1)
    emptyBlocks[i].resize(numEmpty + 1);
  }
}

//----------------------------------------------------------------------------//

template <typename T>
bool CSparseFieldImpl::compress(const SparseField<Imath::Vec3<T> > &field,
                                const int a_bitRate)
{
  if (!configureFrom(field)) {
    return false;
  }

  bitRate = a_bitRate;

  const int blockSize = std::pow(2, blockOrder);
  const int numVoxels = std::pow(blockSize, 3);

  std::vector<float> tempData(numVoxels);

  int idx = 0;
  for (int bk = 0; bk < blockRes.z; ++bk) {
    for (int bj = 0; bj < blockRes.y; ++bj) {
      for (int bi = 0; bi < blockRes.x; ++bi, ++idx) {
        const int bIdx = blockMap[idx];
        if (bIdx >= 0) {
          Imath::Vec3<T> *blockData = field.blockData(bi, bj, bk);
          // Copy allocated data
          for (int c = 0; c < 3; ++c) {
            // Configure block
            blocks[c][bIdx].set_rate(bitRate);
            blocks[c][bIdx].resize(blockSize, blockSize, blockSize);
            blocks[c][bIdx].set_cache_size(DEFAULT_CACHE_SIZE);
            // Copy data into temp array
            for (int v = 0; v < numVoxels; ++v) {
              tempData[v] = blockData[v][c];
            }
            // Ask ZFP to compress
            blocks[c][bIdx].set(&tempData[0]);
          }
        } else {
          // Copy unallocated value
          const V3f value = field.getBlockEmptyValue(bi, bj, bk);
          for (int c = 0; c < 3; ++c) {
            emptyBlocks[c][-bIdx] = value[c];
          }
        }
      }
    }
  }
    

  return true;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
bool CSparseFieldImpl::compress(const SparseField<Data_T> &field,
                                const int a_bitRate)
{
  if (!configureFrom(field)) {
    return false;
  }

  bitRate = a_bitRate;

  const int blockSize = std::pow(2, blockOrder);
  const int numVoxels = std::pow(blockSize, 3);

  std::vector<float> tempData(numVoxels);
  
  int idx = 0;
  for (int bk = 0; bk < blockRes.z; ++bk) {
    for (int bj = 0; bj < blockRes.y; ++bj) {
      for (int bi = 0; bi < blockRes.x; ++bi, ++idx) {
        const int bIdx = blockMap[idx];
        if (bIdx >= 0) {
          Data_T *blockData = field.blockData(bi, bj, bk);
          // Configure block
          blocks[0][bIdx].set_rate(bitRate);
          blocks[0][bIdx].resize(blockSize, blockSize, blockSize);
          blocks[0][bIdx].set_cache_size(DEFAULT_CACHE_SIZE);
          // Copy data into temp array
          for (int v = 0; v < numVoxels; ++v) {
            tempData[v] = blockData[v];
          }
          // Ask ZFP to compress
          blocks[0][bIdx].set(&tempData[0]);
        } else {
          // Copy unallocated value
          float value = field.getBlockEmptyValue(bi, bj, bk);
          emptyBlocks[0][-bIdx] = value;
        }
      }
    }
  }

  return true;
}

//----------------------------------------------------------------------------//

template <typename T>
bool CSparseFieldImpl::decompress(SparseField<Imath::Vec3<T> > &field)
{
  const int blockSize = std::pow(2, blockOrder);
  const int numVoxels = std::pow(blockSize, 3);

  std::vector<float> tempScalarData(numVoxels);

  int idx = 0;
  for (int bk = 0; bk < blockRes.z; ++bk) {
    for (int bj = 0; bj < blockRes.y; ++bj) {
      for (int bi = 0; bi < blockRes.x; ++bi, ++idx) {
        const int bIdx = blockMap[idx];
        if (bIdx >= 0) {
          // Trigger allocation
          field.allocateBlock(bi, bj, bk);
          // Uncompressed storage
          Imath::Vec3<T> *blockData = field.blockData(bi, bj, bk);
          // Copy data to temp storage
          for (int c = 0; c < 3; ++c) {
            // Ask ZFP to decompress
            blocks[c][bIdx].get(&tempScalarData[0]);
            // Copy to uncompressed storage
            for (int i = 0; i < numVoxels; ++i) {
              blockData[i][c] = tempScalarData[i];
            }
          }
        } else {
          // Copy unallocated value
          V3f vec;
          for (int c = 0; c < 3; ++c) {
            vec[c] = emptyBlocks[c][-bIdx];
          }
          field.setBlockEmptyValue(bi, bj, bk, vec);
        }
      }
    }
  }

  return true;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
bool CSparseFieldImpl::decompress(SparseField<Data_T> &field)
{
  const int blockSize = std::pow(2, blockOrder);
  const int numVoxels = std::pow(blockSize, 3);

  std::vector<float> tempData(numVoxels);

  int idx = 0;
  for (int bk = 0; bk < blockRes.z; ++bk) {
    for (int bj = 0; bj < blockRes.y; ++bj) {
      for (int bi = 0; bi < blockRes.x; ++bi, ++idx) {
        const int bIdx = blockMap[idx];
        if (bIdx >= 0) {
          // Trigger allocation
          field.allocateBlock(bi, bj, bk);
          // Uncompressed storage
          Data_T *blockData = field.blockData(bi, bj, bk);
          // Ask ZFP to decompress
          blocks[0][bIdx].get(&tempData[0]);
          // Put into block
          for (int i = 0; i < numVoxels; ++i) {
            blockData[i] = tempData[i];
          }
        } else {
          // Copy unallocated value
          field.setBlockEmptyValue(bi, bj, bk, emptyBlocks[0][-bIdx]);
        }
      }
    }
  }

  return true;
}

//----------------------------------------------------------------------------//

long long int CSparseFieldImpl::memSize() const
{
  long long int size = sizeof(*this);

  size += sizeof(int) * blockMap.size();
  size += sizeof(float) * emptyBlocks.size() * emptyBlocks[0].size();
  size += sizeof(Block) * blocks.size() * blocks[0].size();

  for (int c = 0; c < blocks.size(); ++c) {
    for (int b = 0; b < blocks[c].size(); ++b) {
      size += blocks[c][b].compressed_size();
    }
  }
    
  return size;
}

//----------------------------------------------------------------------------//

size_t CSparseFieldImpl::voxelCount() const
{
  return numBlocks * numValuesPerBlock();
}

//----------------------------------------------------------------------------//

size_t CSparseFieldImpl::numValuesPerBlock() const
{
  const int blockSize = std::pow(2, blockOrder);
  return blockSize * blockSize * blockSize;
}

//----------------------------------------------------------------------------//

size_t CSparseFieldImpl::numLayers() const
{
  return blocks.size();
}

//----------------------------------------------------------------------------//

size_t CSparseFieldImpl::numAllocatedBlocks() const
{
  return blocks[0].size();
}

//----------------------------------------------------------------------------//

size_t CSparseFieldImpl::numEmptyBlocks() const
{
  return emptyBlocks[0].size();
}

//----------------------------------------------------------------------------//

bool CSparseFieldImpl::blockIsAllocated(
  const int bi, const int bj, const int bk) const
{
  const int bIdx = Sparse::blockId(bi, bj, bk, blockRes.x, blockXYSize);
  return blockMap[bIdx] >= 0;
}

//----------------------------------------------------------------------------//

template <typename T>
void CSparseFieldImpl::setBlockEmptyValue(
  const int bi, const int bj, const int bk, const Imath::Vec3<T> &value) 
{
  const int bIdx = 
    blockMap[Sparse::blockId(bi, bj, bk, blockRes.x, blockXYSize)];
  if (bIdx >= 0) {
    for (int c = 0; c < 3; ++c) {
      emptyBlocks[c][-bIdx] = value[c];
    }
  } 
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void CSparseFieldImpl::setBlockEmptyValue(
  const int bi, const int bj, const int bk, const Data_T &value) 
{
  const int bIdx = 
    blockMap[Sparse::blockId(bi, bj, bk, blockRes.x, blockXYSize)];
  if (bIdx >= 0) {
    emptyBlocks[0][-bIdx] = value;
  } 
}

//----------------------------------------------------------------------------//

template <typename T>
void CSparseFieldImpl::getBlockEmptyValue(
  const int bi, const int bj, const int bk, Imath::Vec3<T> &value) const
{
  const int bIdx = 
    blockMap[Sparse::blockId(bi, bj, bk, blockRes.x, blockXYSize)];
  if (bIdx >= 0) {
    for (int c = 0; c < 3; ++c) {
      value[c] = emptyBlocks[c][-bIdx];
    }
  } 
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void CSparseFieldImpl::getBlockEmptyValue(
  const int bi, const int bj, const int bk, Data_T &value) const
{
  const int bIdx = 
    blockMap[Sparse::blockId(bi, bj, bk, blockRes.x, blockXYSize)];
  if (bIdx >= 0) {
    value = emptyBlocks[0][-bIdx];
  } 
}

//----------------------------------------------------------------------------//
// CSparseField specialized implementations
//----------------------------------------------------------------------------//

template <>
half CSparseField<half>::fastValue(int i, int j, int k) const
{
  Sparse::applyDataWindowOffset(i, j, k, FieldRes::m_dataWindow);

  return m_impl->value(i, j, k, 0);
}

//----------------------------------------------------------------------------//

template <>
float CSparseField<float>::fastValue(int i, int j, int k) const
{
  Sparse::applyDataWindowOffset(i, j, k, FieldRes::m_dataWindow);

  return m_impl->value(i, j, k, 0);
}

//----------------------------------------------------------------------------//

template <>
double CSparseField<double>::fastValue(int i, int j, int k) const
{
  Sparse::applyDataWindowOffset(i, j, k, FieldRes::m_dataWindow);

  return m_impl->value(i, j, k, 0);
}

//----------------------------------------------------------------------------//

template <>
V3h CSparseField<V3h>::fastValue(int i, int j, int k) const
{
  Sparse::applyDataWindowOffset(i, j, k, FieldRes::m_dataWindow);

  return V3h(m_impl->value(i, j, k, 0),
             m_impl->value(i, j, k, 1),
             m_impl->value(i, j, k, 2));
}

//----------------------------------------------------------------------------//

template <>
V3f CSparseField<V3f>::fastValue(int i, int j, int k) const
{
  Sparse::applyDataWindowOffset(i, j, k, FieldRes::m_dataWindow);

  return V3f(m_impl->value(i, j, k, 0),
             m_impl->value(i, j, k, 1),
             m_impl->value(i, j, k, 2));
}

//----------------------------------------------------------------------------//

template <>
V3d CSparseField<V3d>::fastValue(int i, int j, int k) const
{
  Sparse::applyDataWindowOffset(i, j, k, FieldRes::m_dataWindow);

  return V3d(m_impl->value(i, j, k, 0),
             m_impl->value(i, j, k, 1),
             m_impl->value(i, j, k, 2));
}

//----------------------------------------------------------------------------//

template <typename Data_T>
Data_T CSparseField<Data_T>::value(int i, int j, int k) const
{
  return fastValue(i, j, k);
}

//----------------------------------------------------------------------------//
// CSparseField implementations
//----------------------------------------------------------------------------//

template <class Data_T>
size_t CSparseField<Data_T>::numGrains() const
{
  return m_impl->numBlocks;
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool CSparseField<Data_T>::getGrainBounds(const size_t idx, Box3i &bounds) const
{
  // Block size
  const size_t blockSide = (1 << m_impl->blockOrder);
  // Block coordinate
  const V3i bCoord       = indexToCoord(idx, m_impl->blockRes);
  // Block bbox
  const V3i   start(bCoord * blockSide + base::m_dataWindow.min);
  const V3i   end  (start + Imath::V3i(blockSide - 1));
  // Bounds must be clipped against data window
  const Box3i unclipped(start, end);
  bounds = clipBounds(unclipped, base::m_dataWindow);
  // Whether it's a contiguous block
  return bounds == unclipped;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
CSparseField<Data_T>::CSparseField()
{
  m_implPtr.reset(new CSparseFieldImpl);
  m_impl = m_implPtr.get();
}

//----------------------------------------------------------------------------//

template <typename Data_T>
long long int CSparseField<Data_T>::memSize() const
{
  return m_impl->memSize() + sizeof(*this);
}

//----------------------------------------------------------------------------//

template <typename Data_T>
size_t CSparseField<Data_T>::voxelCount() const
{
  return m_impl->voxelCount();;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void CSparseField<Data_T>::configure(const Box3i &extents,
                                     const Box3i &dataWindow,
                                     const int    blockOrder,
                                     const V3i   &blockRes,
                                     const int    bitRate,
                                     const std::vector<int> &blockMap)
{
  FieldRes::m_extents    = extents;
  FieldRes::m_dataWindow = dataWindow;
  
  m_impl->configure<Data_T>(blockOrder, blockRes, bitRate, blockMap);
}

//----------------------------------------------------------------------------//

template <typename Data_T>
bool CSparseField<Data_T>::compress(const SparseField<Data_T> &field, 
                                    const int bitRate)
{
  // Copy field configuration. Note that we do not derive from ResizableField,
  // so setSize() is unavailable.
  FieldRes::m_extents    = field.extents();
  FieldRes::m_dataWindow = field.dataWindow();
  FieldRes::setMapping(field.mapping());

  // Copy name, attr, metadata
  FieldBase::name = field.name;
  FieldBase::attribute = field.attribute;
  FieldBase::copyMetadata(field);

  // Copy actual data
  if (!m_impl->compress(field, bitRate)) {
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------//
  
template <typename Data_T>
typename SparseField<Data_T>::Ptr 
CSparseField<Data_T>::decompress() const
{
  typename SparseField<Data_T>::Ptr sf(new SparseField<Data_T>);

  sf->setMapping(FieldRes::mapping());
  sf->setSize(FieldRes::m_extents, FieldRes::m_dataWindow);
  sf->name = FieldBase::name;
  sf->attribute = FieldBase::attribute;
  sf->copyMetadata(*this);

  m_impl->decompress(*sf);

  return sf;
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool CSparseField<Data_T>::blockIsAllocated(
  const int bi, const int bj, const int bk) const
{
  return m_impl->blockIsAllocated(bi, bj, bk);
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T CSparseField<Data_T>::getBlockEmptyValue(
  const int bi, const int bj, const int bk) const
{
  Data_T value(0.0);
  m_impl->getBlockEmptyValue(bi, bj, bk, value);
  return value;
}

//----------------------------------------------------------------------------//

template <class Data_T>
int CSparseField<Data_T>::blockSize() const
{
  return 1 << m_impl->blockOrder;
}

//----------------------------------------------------------------------------//

template <class Data_T>
int CSparseField<Data_T>::bitRate() const
{
  return m_impl->bitRate;
}

//----------------------------------------------------------------------------//

template <class Data_T>
V3i CSparseField<Data_T>::blockRes() const
{
  return m_impl->blockRes;
}

//----------------------------------------------------------------------------//

template <class Data_T>
CSparseFieldImpl* CSparseField<Data_T>::impl()
{
  return m_impl;
}

//----------------------------------------------------------------------------//
// Template instantiations
//----------------------------------------------------------------------------//

template class CSparseField<half>;
template class CSparseField<float>;
template class CSparseField<double>;
template class CSparseField<V3h>;
template class CSparseField<V3f>;
template class CSparseField<V3d>;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
