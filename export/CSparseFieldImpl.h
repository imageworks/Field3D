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

/*! \file CSparseFieldImpl.h
  \brief Contains the CSparseField implementation
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_CSparseFieldImpl_H_
#define _INCLUDED_Field3D_CSparseFieldImpl_H_

//----------------------------------------------------------------------------//

#include <zfp/zfparray3.h>

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Typedefs
//----------------------------------------------------------------------------//

typedef zfp::array3f CSparseBlock;
typedef std::vector<int> CSparseBlockMap;

//----------------------------------------------------------------------------//
// CSparseFieldImpl
//----------------------------------------------------------------------------//

struct CSparseFieldImpl
{
  // Typedefs ---

  typedef CSparseBlock Block;

  // Ctors ---

  CSparseFieldImpl();

  // Main methods ---

  //! Const voxel access
  float value(const int i, const int j, const int k, const int layer) const;

  //! Configures the compressed storage from the given sparse field
  template <typename Data_T>
  bool configureFrom(const SparseField<Data_T> &field);
  //! Configures at read-time
  //! \note Template argument is used to access traits classes.
  template <typename Data_T>
  void configure(const int blockOrder,
                 const V3i &blockRes,
                 const int bitRate,
                 const std::vector<int> &blockMap);
  //! Compresses a vector field
  template <typename Data_T>
  bool compress(const SparseField<Data_T> &field, const int bitRate);
  //! Compresses a scalar field
  template <typename T>
  bool compress(const SparseField<Imath::Vec3<T> > &field, const int bitRate);
  //! Decompresses into a scalar field
  template <typename Data_T>
  bool decompress(SparseField<Data_T> &field);
  //! Decompresses into a vector field
  template <typename T>
  bool decompress(SparseField<Imath::Vec3<T> > &field);

  long long int memSize() const;
  size_t voxelCount() const;

  //! Number of voxels times number of components
  size_t numValuesPerBlock() const;
  //! Number of layers (i.e. components). 1 for scalar, 3 for vector
  size_t numLayers() const;
  //! Number of allocated blocks
  size_t numAllocatedBlocks() const;
  //! Number of empty blocks
  size_t numEmptyBlocks() const;
  //! Whether the block is allocated, indexed linearly.
  bool blockIsAllocated(const int blockIdx) const
  { return blockMap[blockIdx] >= 0; }
  //! Whether the block is allocated, indexed spatially.
  bool blockIsAllocated(const int bi, const int bj, const int bk) const;

  //! Pointer to the compressed data
  uchar* data(const int layer, const int blockIdx)
  { return blocks[layer][blockMap[blockIdx]].compressed_data(); }

  //! Length of compressed data. This is the same for all blocks.
  //! Note, if the field is vector-valued, this is the length for a 
  //! single component.
  size_t dataLength() const
  { return blocks[0][0].compressed_size(); }

  template <typename T>
  void setBlockEmptyValue(const int bi, const int bj, const int bk,
                          const Imath::Vec3<T> &value);
  template <typename Data_T>
  void setBlockEmptyValue(const int bi, const int bj, const int bk,
                          const Data_T &value);

  template <typename T>
  void getBlockEmptyValue(const int bi, const int bj, const int bk,
                          Imath::Vec3<T> &value) const;
  template <typename Data_T>
  void getBlockEmptyValue(const int bi, const int bj, const int bk,
                          Data_T &value) const;

  // Data members ---

  //! Block order (size = 2^blockOrder)
  int blockOrder;
  //! Block array resolution
  V3i blockRes;
  //! Block array res.x * res.y
  int blockXYSize;
  //! Bit rate
  int bitRate;
  //! Array of blocks. Outer vector is the component (to support vectors).
  std::vector<std::vector<Block> > blocks;
  //! Array of 'empty' block values. Outer vector is the component.
  std::vector<std::vector<float> > emptyBlocks;
  //! Array of block indices. Maps a linear block index to index in blocks.
  //! \note Negative indices refer to 'empty' values, found in emptyBlocks
  CSparseBlockMap blockMap;
  //! Number of blocks in field.
  size_t numBlocks;
};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
