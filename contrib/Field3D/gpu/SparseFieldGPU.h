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

#ifndef _INCLUDED_Field3D_gpu_SparseFieldGPU_H_
#define _INCLUDED_Field3D_gpu_SparseFieldGPU_H_

#ifdef NVCC
#error "This file isn't compatible with NVCC compiler due to Field3D includes"
#endif

#include "Field3D/gpu/Traits.h"
#include "Field3D/gpu/buffer/Copy.h"

#ifdef INCLUDE_FIELD3D_CUDA
#include "Field3D/gpu/FieldInterpCuda.h"
#include "Field3D/gpu/buffer/BufferCuda.h"
#endif

#include "Field3D/gpu/buffer/BufferHost.h"

#include "Field3D/SparseField.h"
#include "Field3D/Types.h"

#include "Field3D/gpu/ns.h"

FIELD3D_GPU_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// forward declarations
template <typename A, typename B> struct SparseFieldSampler;
template <typename A> struct LinearFieldInterp;

//----------------------------------------------------------------------------//
// BlockFunctor
//----------------------------------------------------------------------------//
/*! A functor to determine whether a block is required
 * \note A user defined BlockFunctor can allow partial loading onto GPU
 *       eg via a bounding box or frustum check
 */
//----------------------------------------------------------------------------//

struct BlockFunctor
{
  //! determine whether block <<bi,bj,bk>> is required
  template <typename Field_T>
  bool operator()(Field_T& f, int bi, int bj, int bk) const;
};

//----------------------------------------------------------------------------//
// EveryBlockFunctor
//----------------------------------------------------------------------------//
//! A concrete BlockFunctor that indicates all blocks are required
//----------------------------------------------------------------------------//

struct EveryBlockFunctor : BlockFunctor
{
  template<typename Field_T>
  bool operator()(Field_T& f, int bi, int bj, int bk) const
  {
    return true;
  }
};

//----------------------------------------------------------------------------//
// SparseFieldGPU
//----------------------------------------------------------------------------//
//! Cuda layer for SparseFields
//----------------------------------------------------------------------------//

template <typename Data_T>
struct SparseFieldGPU : public RefBase
{
  typedef boost::intrusive_ptr<SparseFieldGPU> Ptr;
  typedef typename GpuFieldTraits<Data_T>::value_type value_type;
  typedef typename GpuFieldTraits<Data_T>::cuda_value_type cuda_value_type;
  typedef typename GpuFieldTraits<Data_T>::interpolation_type
    interpolation_type;
  typedef SparseFieldSampler<value_type, interpolation_type> sampler_type;
  typedef LinearFieldInterp<sampler_type> LinearInterp;
  typedef boost::shared_ptr<LinearInterp> LinearInterpPtr;

  typedef SparseField<Data_T> field3d_type;
  typedef typename field3d_type::Ptr field_ptr;

  // Main methods --------------------------------------------------------------

  //! access the Field3D field
  field_ptr getField()
  {
    return m_field;
  }

  //! set the Field3D field
  void setField(field_ptr& src)
  {
    m_field = src;
  }

  /*! host to device transfer for blocks
   *  \note There is an index table and data buffer, with the data buffer
   *        containing both empty values and block data
   */
  template<typename IntBuffer_T, typename Buffer_T, typename BlockFunctor_T>
  void hostToDevice(IntBuffer_T& blockTable,
                    Buffer_T& buffer,
                    int& emptyValueOffset,
                    BlockFunctor_T& bf) const;

  template <typename IntBuffer_T, typename Buffer_T>
  void hostToDevice(IntBuffer_T& blockTable,
                    Buffer_T& buffer,
                    int& emptyValueOffset) const
  {
    EveryBlockFunctor f;
    return hostToDevice(blockTable, buffer, emptyValueOffset, f);
  }

#ifdef INCLUDE_FIELD3D_CUDA

  //! manufacture an interpolator for device using a BlockFunctor
  template <typename BlockFunctor_T>
  LinearInterpPtr getLinearInterpolatorDevice(BlockFunctor_T& bf) const;

  //! manufacture an interpolator for device
  LinearInterpPtr getLinearInterpolatorDevice() const
  {
    EveryBlockFunctor f;
    return getLinearInterpolatorDevice(f);
  }

  //! manufacture an interpolator for host using a BlockFunctor
  template <typename BlockFunctor_T>
  LinearInterpPtr getLinearInterpolatorHost(BlockFunctor_T& bf) const;

  //! manufacture an interpolator for host
  LinearInterpPtr getLinearInterpolatorHost() const
  {
    EveryBlockFunctor f;
    return getLinearInterpolatorHost(f);
  }

#endif

  //! Number of allocated blocks that meet BlockFunctor requirements
  template <typename BlockFunctor_T>
  int requiredBlockCount(const BlockFunctor_T& bf) const;

  int deviceElementsRequiredForBlocks(int required_block_count) const;

  //! Project the required number of Data_T elements
  //  for block cache + emptyValues
  int deviceElementsRequired(int required_block_count) const;

  // Data members --------------------------------------------------------------

private:
  field_ptr m_field;

#ifdef INCLUDE_FIELD3D_CUDA
  mutable BufferCuda< int > m_blockTableCuda;
  mutable BufferCuda< Data_T > m_bufferCuda;
#endif
  mutable BufferHost< int > m_blockTableHost;
  mutable BufferHost< Data_T > m_bufferHost;

  mutable int m_blockCount;
  mutable int m_texMemSize;
};

//----------------------------------------------------------------------------//
// SparseFieldGPU implementations
//----------------------------------------------------------------------------//

template<typename Data_T>
template<typename IntBuffer_T, typename Buffer_T, typename BlockFunctor_T>
void SparseFieldGPU<Data_T>::hostToDevice(IntBuffer_T& blockTable,
                                          Buffer_T& buffer,
                                          int& emptyValueOffset,
                                          BlockFunctor_T& bf) const
{
  int required_block_count = requiredBlockCount(bf);
  int gpu_cache_element_count =
  deviceElementsRequired(required_block_count);

  const V3i br = m_field->blockRes();
  const int block_count = br.x * br.y * br.z; // total block count

  buffer.resize(gpu_cache_element_count);

  std::vector<Data_T> host_empty_values(block_count);
  // initialize to empty value
  std::vector<int> host_block_table(block_count, -1);

  int i = 0;
  int allocated_block_index = 0;

  // allow room for empty values
  //int write_index = block_count;
  int write_index = 0;
  int bs = m_field->blockSize();
  int block_element_count = bs * bs * bs;

  typename field3d_type::block_iterator bi(m_field->blockBegin()),
  be(m_field->blockEnd());
  for (; bi != be; ++bi, ++i)
  {
    host_empty_values[i] = m_field->getBlockEmptyValue(bi.x, bi.y, bi.z);
    size_t block_element_count = m_field->blockSize()
    * m_field->blockSize()
    * m_field->blockSize();
    if( m_field->blockIsAllocated(bi.x, bi.y, bi.z)
        && bf(*this, bi.x, bi.y, bi.z) )
    {
      const Data_T* ptr = m_field->blockData(bi.x, bi.y, bi.z);
      host_block_table[i] = write_index;
      assert(block.data.size() == block_element_count);

      Field3D::Gpu::copy(ptr,
                         ptr + block_element_count,
                         buffer.begin() + write_index);
      write_index += block_element_count;
    }
  }

  assert(i == block_count);

  emptyValueOffset = write_index;

  // host -> device for empty values (stored after blocks)
  Field3D::Gpu::copy(host_empty_values.begin(),
                     host_empty_values.end(),
                     buffer.begin() + write_index);

  // host->device for block table;
  blockTable.resize(host_block_table.size() );
  Field3D::Gpu::copy(host_block_table.begin(),
                     host_block_table.end(),
                     blockTable.begin());

  m_texMemSize = gpu_cache_element_count * sizeof(Data_T);
  m_blockCount = required_block_count;
}

//----------------------------------------------------------------------------//
template <typename Data_T>
template <typename BlockFunctor_T>
int SparseFieldGPU<Data_T>::requiredBlockCount(const BlockFunctor_T& bf) const
{
  typename field3d_type::block_iterator bi(m_field->blockBegin()),
  be(m_field->blockEnd());
  int result = 0;
  for (; bi != be; ++bi)
  {
    if( m_field->blockIsAllocated(bi.x, bi.y, bi.z)
        && bf(*this, bi.x, bi.y, bi.z))
    {
      ++result;
    }
  }
  return result;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
int SparseFieldGPU<Data_T>::deviceElementsRequiredForBlocks
  (int required_block_count) const
{
  // block data
  int bs = m_field->blockSize();
  return required_block_count * bs * bs * bs;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
int SparseFieldGPU<Data_T>::deviceElementsRequired(int required_block_count)
  const
{
  int result = 0;

  result += deviceElementsRequiredForBlocks(required_block_count);

  // sparse block emptyValues
  V3i br = m_field->blockRes();
  result += br.x * br.y * br.z;

  return result;
}

//----------------------------------------------------------------------------//

#ifdef INCLUDE_FIELD3D_CUDA

template <typename Data_T>
template <typename BlockFunctor_T>
typename SparseFieldGPU<Data_T>::LinearInterpPtr
SparseFieldGPU<Data_T>::getLinearInterpolatorDevice(BlockFunctor_T& bf) const
{
  int emptyValueOffset;
  hostToDevice(m_blockTableCuda, m_bufferCuda, emptyValueOffset, bf);

  return LinearInterpPtr(
      new LinearInterp(
          sampler_type(m_field->dataResolution(),
                       m_field->dataWindow(),
                       m_blockCount,
                       m_field->blockOrder(),
                       m_field->blockRes(),
                       thrust::raw_pointer_cast(&m_blockTableCuda[0]),
                       emptyValueOffset,
                       (cuda_value_type*)
                       thrust::raw_pointer_cast(&m_bufferCuda[0]),
                       m_texMemSize)));
}

//----------------------------------------------------------------------------//

template <typename Data_T>
template <typename BlockFunctor_T>
typename SparseFieldGPU<Data_T>::LinearInterpPtr
SparseFieldGPU<Data_T>::getLinearInterpolatorHost(BlockFunctor_T& bf) const
{
  int emptyValueOffset;
  hostToDevice(m_blockTableHost, m_bufferHost, emptyValueOffset, bf);

  return LinearInterpPtr(
      new LinearInterp(
          sampler_type(m_field->dataResolution(),
                       m_field->dataWindow(),
                       m_blockCount,
                       m_field->blockOrder(),
                       m_field->blockRes(),
                       &m_blockTableHost[0],
                       emptyValueOffset,
                       (cuda_value_type*) &m_bufferHost[0],
                       m_texMemSize)));
}
#endif

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
