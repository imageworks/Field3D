//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2013 Sony Pictures Imageworks Inc
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

/*! \file MIPUtil.h
  \brief Contains MIP-related utility functions
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_MIPUtil_H_
#define _INCLUDED_Field3D_MIPUtil_H_

//----------------------------------------------------------------------------//

#include <vector>

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "Resample.h"
#include "SparseField.h"
#include "Types.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Functions
//----------------------------------------------------------------------------//

//! Computes the origin/offset of a field. 
V3i computeOffset(const FieldRes &f);

//! Constructs a MIP representation of the given field, with optional 
//! offset vector. The offset vector indicates the 'true' voxel space
//! coordinate of the (0, 0, 0) voxel, such that a consistent voxel placement
//! can be used for the MIP levels.
template <typename MIPField_T, typename Filter_T>
typename MIPField_T::Ptr
makeMIP(const typename MIPField_T::NestedType &base, const int minSize,
        const V3i &offset, const size_t numThreads);

//! Constructs a MIP representation of the given field.
template <typename MIPField_T, typename Filter_T>
typename MIPField_T::Ptr
makeMIP(const typename MIPField_T::NestedType &base, const int minSize,
        const size_t numThreads);

//----------------------------------------------------------------------------//
// Implementation details
//----------------------------------------------------------------------------//

namespace detail {

  //--------------------------------------------------------------------------//

  extern const std::string k_mipOffsetStr;

  //--------------------------------------------------------------------------//

  //! Used to delegate the choice of bit depth to process at
  template <typename T>
  struct ComputationType
  {
    typedef T type;
  };

  //! Specialization for half float
  template <>
  struct ComputationType<Field3D::half>
  {
    typedef float type;
  };

  //--------------------------------------------------------------------------//

  FIELD3D_API V3i mipResolution(const V3i &baseRes, const size_t level, 
                                const V3i &add);

  //--------------------------------------------------------------------------//

  //! Constant size for all dense fields
  template <typename Data_T>
  size_t threadingBlockSize(const DenseField<Data_T> & /* f */)
  {
    return 16;
  }
  
  //! Use block size for sparse fields
  template <typename Data_T>
  size_t threadingBlockSize(const SparseField<Data_T> &f)
  {
    return f.blockSize();
  }

  //--------------------------------------------------------------------------//

  template <typename Data_T>
  bool checkInputEmpty(const SparseField<Data_T> &src, 
                       const SparseField<Data_T> &/*tgt*/, 
                       const Box3i &tgtBox, const float support,
                       const size_t dim)
  {
    const int intSupport = static_cast<int>(std::ceil(support * 0.5));
    const int pad        = std::max(0, intSupport);
    Box3i     tgtBoxPad  = tgtBox;
    tgtBoxPad.min[dim]  -= pad;
    tgtBoxPad.max[dim]  += pad;
    Box3i     srcBoxPad  = tgtBoxPad;
    srcBoxPad.min[dim]  *= 2;
    srcBoxPad.max[dim]  *= 2;

    // Get the block coordinates
    const Box3i dbsBounds = blockCoords(clipBounds(srcBoxPad, src.dataWindow()),
                                        &src);

    static boost::mutex mutex;
    boost::mutex::scoped_lock lock(mutex);

    // Check all blocks
    for (int k = dbsBounds.min.z; k <= dbsBounds.max.z; ++k) {
      for (int j = dbsBounds.min.y; j <= dbsBounds.max.y; ++j) {
        for (int i = dbsBounds.min.x; i <= dbsBounds.max.x; ++i) {
          if (src.blockIsAllocated(i, j, k) ||
              src.getBlockEmptyValue(i, j, k) != static_cast<Data_T>(0)) {
            return false;
          }
        }
      } 
    }

    // No hits. Empty
    return true;
  }

  //--------------------------------------------------------------------------//

  //! Fallback version always returns false
  template <typename Field_T>
  bool checkInputEmpty(const Field_T &/*src*/, const Field_T &/*tgt*/, 
                       const Box3i &/*tgtBox*/, const float /*support*/,
                       const size_t /*dim*/)
  {
    return false;
  }

  //--------------------------------------------------------------------------//

  template <typename Field_T, typename FilterOp_T, bool IsAnalytic_T>
  struct MIPSeparableThreadOp
  {
    typedef typename Field_T::value_type T;

    MIPSeparableThreadOp(const Field_T &src, Field_T &tgt, 
                         const size_t level, const V3i &add,
                         const FilterOp_T &filterOp, 
                         const size_t dim, 
                         const std::vector<Box3i> &blocks,
                         size_t &nextIdx, boost::mutex &mutex)
      : m_src(src),
        m_tgt(tgt),
        m_filterOp(filterOp), 
        m_level(level), 
        m_add(add), 
        m_dim(dim),
        m_blocks(blocks),
        m_nextIdx(nextIdx),
        m_mutex(mutex),
        m_numBlocks(blocks.size())
    {
      // Empty
    }

    void operator() () 
    {
      using namespace std;

      // Defer to ComputationType to determine the processing data type
      typedef typename Field_T::value_type           Data_T;
      typedef typename ComputationType<Data_T>::type Value_T;

      // To ensure we don't sample outside source data
      Box3i srcDw = m_src.dataWindow();

      // Coordinate frame conversion constants
      const float tgtToSrcMult    = 2.0;
      const float filterCoordMult = 1.0f / (tgtToSrcMult);
    
      // Filter info, support size in target space
      const float support = m_filterOp.support();

      // Get next index to process
      size_t idx;
      {
        boost::mutex::scoped_lock lock(m_mutex);
        idx = m_nextIdx;
        m_nextIdx++;
      }
      // Keep going while there is data to process
      while (idx < m_numBlocks) {
        // Grab the bounds
        const Box3i box =  m_blocks[idx];
        // Early exit if input blocks are all empty
        if (!detail::checkInputEmpty(m_src, m_tgt, box, support, m_dim)) {
          // For each output voxel
          for (int k = box.min.z; k <= box.max.z; ++k) {
            for (int j = box.min.y; j <= box.max.y; ++j) {
              for (int i = box.min.x; i <= box.max.x; ++i) {
                Value_T accumValue(m_filterOp.initialValue());
                if (IsAnalytic_T) {
                  // Transform from current point in target frame to source frame
                  const int   curTgt = V3i(i, j, k)[m_dim];
                  const float curSrc = discToCont(curTgt) * tgtToSrcMult - m_add[m_dim];
                  // Find interval
                  int startSrc = 
                    static_cast<int>(std::floor(curSrc - support * tgtToSrcMult));
                  int endSrc   = 
                    static_cast<int>(std::ceil(curSrc + support * 
                                               tgtToSrcMult)) - 1;
                  // Clamp coordinates
                  startSrc     = std::max(startSrc, srcDw.min[m_dim]);
                  endSrc       = std::min(endSrc, srcDw.max[m_dim]);
                  // Loop over source voxels
                  for (int s = startSrc; s <= endSrc; ++s) {
                    // Source index
                    const int xIdx = m_dim == 0 ? s : i;
                    const int yIdx = m_dim == 1 ? s : j;
                    const int zIdx = m_dim == 2 ? s : k;
                    // Source voxel in continuous coords
                    const float srcP   = discToCont(s);
                    // Compute filter weight in source space (twice as wide)
                    const float weight = m_filterOp.eval(std::abs(srcP - curSrc) *
                                                         filterCoordMult);
                    // Value
                    const Value_T value = m_src.fastValue(xIdx, yIdx, zIdx);
                    // Update
                    if (weight > 0.0f) {
                      FilterOp_T::op(accumValue, value);
                    }
                  }
                  // Update final value
                  if (accumValue != 
                      static_cast<Value_T>(m_filterOp.initialValue())) {
                    m_tgt.fastLValue(i, j, k) = accumValue;
                  }
                } else {
                  float accumWeight  = 0.0f;
                  // Transform from current point in target frame to source frame
                  const int   curTgt = V3i(i, j, k)[m_dim];
                  const float curSrc = discToCont(curTgt) * tgtToSrcMult - m_add[m_dim];
                  // Find interval
                  int startSrc = 
                    static_cast<int>(std::floor(curSrc - support * tgtToSrcMult));
                  int endSrc   = 
                    static_cast<int>(std::ceil(curSrc + support * 
                                               tgtToSrcMult)) - 1;
                  // Clamp coordinates
                  startSrc     = std::max(startSrc, srcDw.min[m_dim]);
                  endSrc       = std::min(endSrc, srcDw.max[m_dim]);
                  // Loop over source voxels
                  for (int s = startSrc; s <= endSrc; ++s) {
                    // Source index
                    const int xIdx = m_dim == 0 ? s : i;
                    const int yIdx = m_dim == 1 ? s : j;
                    const int zIdx = m_dim == 2 ? s : k;
                    // Source voxel in continuous coords
                    const float srcP   = discToCont(s);
                    // Compute filter weight in source space (twice as wide)
                    const float weight = m_filterOp.eval(std::abs(srcP - curSrc) *
                                                         filterCoordMult);
                    // Value
                    const Value_T value = m_src.fastValue(xIdx, yIdx, zIdx);
                    // Update
                    accumWeight += weight;
                    accumValue  += value * weight;
                  }
                  // Update final value
                  if (accumWeight > 0.0f && 
                      accumValue != static_cast<Value_T>(0.0)) {
                    m_tgt.fastLValue(i, j, k) = accumValue / accumWeight;
                  }
                } // if (IsAnalytic_T)
              }
            }
          }
        } // Empty input
        // Get next index
        {
          boost::mutex::scoped_lock lock(m_mutex);
          idx = m_nextIdx;
          m_nextIdx++;
        }
      }
    }

  private:

    // Data members ---

    const Field_T            &m_src;
    Field_T                  &m_tgt;
    const FilterOp_T         &m_filterOp;
    const size_t              m_level;
    const V3i                &m_add;
    const size_t              m_dim;
    const std::vector<Box3i> &m_blocks;
    size_t                   &m_nextIdx;
    boost::mutex             &m_mutex;
    const size_t              m_numBlocks;
    
  };

  //--------------------------------------------------------------------------//

  //! Threaded implementation of separable MIP filtering
  template <typename Field_T, typename FilterOp_T>
  void mipSeparable(const Field_T &src, Field_T &tgt, 
                    const V3i &oldRes, const V3i &newRes, const size_t level, 
                    const V3i &add, const FilterOp_T &filterOp, 
                    const size_t dim, const size_t numThreads)
  {
    using namespace std;

    // To ensure we don't sample outside source data
    Box3i srcDw = src.dataWindow();

    // Compute new res
    V3i res;
    if (dim == 2) {
      res = newRes;
    } else if (dim == 1) {
      res = V3i(newRes.x, newRes.y, oldRes.z);
    } else {
      res = V3i(newRes.x, oldRes.y, oldRes.z);
    }

    // Resize new field
    tgt.setSize(res);
    
    // Determine granularity
    const size_t blockSize = threadingBlockSize(src);

    // Build block list
    std::vector<Box3i> blocks;
    for (int k = 0; k < res.z; k += blockSize) {
      for (int j = 0; j < res.y; j += blockSize) {
        for (int i = 0; i < res.x; i += blockSize) {
          Box3i box;
          // Initialize block size
          box.min = V3i(i, j, k);
          box.max = box.min + V3i(blockSize - 1);
          // Clip against resolution
          box.max.x = std::min(box.max.x, res.x - 1);
          box.max.y = std::min(box.max.y, res.y - 1);
          box.max.z = std::min(box.max.z, res.z - 1);
          // Add to list
          blocks.push_back(box);
        }
      }
    }

    // Next index counter and mutex
    size_t nextIdx = 0;
    boost::mutex mutex;

    // Launch threads ---

    boost::thread_group threads;

    for (size_t i = 0; i < numThreads; ++i) {
      threads.create_thread(
        MIPSeparableThreadOp<Field_T, FilterOp_T, FilterOp_T::isAnalytic >
        (src, tgt, level, add, filterOp, 
         dim, blocks, nextIdx, mutex));
    }

    // Join
    threads.join_all();
  }

  //--------------------------------------------------------------------------//

  template <typename Field_T, typename FilterOp_T>
  void mipResample(const Field_T &base, const Field_T &src, Field_T &tgt, 
                   const size_t level, const V3i &offset, 
                   const FilterOp_T &filterOp, 
                   const size_t numThreads)
  {
    using std::ceil;

    // Odd-numbered offsets need a pad of one in the negative directions
    const V3i add((offset.x % 2 == 0) ? 0 : 1,
                  (offset.y % 2 == 0) ? 0 : 1,
                  (offset.z % 2 == 0) ? 0 : 1);

    // Compute new res
    const Box3i baseDw  = base.dataWindow();
    const V3i   baseRes = baseDw.size() + V3i(1);
    const V3i   newRes  = mipResolution(baseRes, level, add);

    // Source res
    const Box3i srcDw  = src.dataWindow();
    const V3i   srcRes = srcDw.size() + V3i(1);

    // Temporary field for y component
    Field_T tmp;

    // X axis (src into tgt)
    mipSeparable(src, tgt, srcRes, newRes, level, add, filterOp, 0, numThreads);
    // Y axis (tgt into temp)
    mipSeparable(tgt, tmp, srcRes, newRes, level, add, filterOp, 1, numThreads);
    // Z axis (temp into tgt)
    mipSeparable(tmp, tgt, srcRes, newRes, level, add, filterOp, 2, numThreads);

    // Update final target with mapping and metadata
    tgt.name      = base.name;
    tgt.attribute = base.attribute;
    tgt.setMapping(base.mapping());
    tgt.copyMetadata(base);
  }

  //--------------------------------------------------------------------------//

  FIELD3D_API
  FieldMapping::Ptr adjustedMIPFieldMapping(const FieldRes *base,
                                            const V3i &baseRes,
                                            const Box3i &extents, 
                                            const size_t level);

  //--------------------------------------------------------------------------//

} // namespace detail

//----------------------------------------------------------------------------//
// Function implementations
//----------------------------------------------------------------------------//

template <typename MIPField_T, typename Filter_T>
typename MIPField_T::Ptr
makeMIP(const typename MIPField_T::NestedType &base, const int minSize,
        const size_t numThreads)
{
  // By default, there is no offset
  const V3i zero(0);
  // Call out to perform actual work
  return makeMIP<MIPField_T, Filter_T>(base, minSize, zero, numThreads);
}

//----------------------------------------------------------------------------//

template <typename MIPField_T, typename Filter_T>
typename MIPField_T::Ptr
makeMIP(const typename MIPField_T::NestedType &base, const int minSize,
        const V3i &baseOffset, const size_t numThreads)
{
  using namespace Field3D::detail;

  typedef typename MIPField_T::value_type    Data_T;
  typedef typename MIPField_T::NestedType    Src_T;
  typedef typename Src_T::Ptr                SrcPtr;
  typedef typename MIPField_T::Ptr           MIPPtr;
  typedef std::vector<typename Src_T::Ptr>   SrcVec;

  if (base.extents() != base.dataWindow()) {
    return MIPPtr();
  }
  
  // Initialize output vector with base resolution
  SrcVec result;
  result.push_back(field_dynamic_cast<Src_T>(base.clone()));

  // Iteration variables
  V3i res    = base.extents().size() + V3i(1);
  V3i offset = baseOffset;
  
  // Loop until minimum size is found
  size_t level = 1;
  while ((res.x > minSize || res.y > minSize || res.z > minSize) &&
         (res.x > 2 && res.y > 2 && res.z > 2)) {
    // Perform filtering
    SrcPtr nextField(new Src_T);
    mipResample(base, *result.back(), *nextField, level, offset, 
                Filter_T(), numThreads);
    // Add to vector of filtered fields
    result.push_back(nextField);
    // Set up for next iteration
    res = nextField->dataWindow().size() + V3i(1);
    // ... offset needs to be rounded towards negative inf, not towards zero
    for (int i = 0; i < 3; ++i) {
      if (offset[i] < 0) {
        offset[i] = (offset[i] - 1) / 2;
      } else {
        offset[i] /= 2;
      }
    }
    level++;
  }

  MIPPtr mipField(new MIPField_T);
  mipField->name = base.name;
  mipField->attribute = base.attribute;
  mipField->copyMetadata(base);
  mipField->setMIPOffset(baseOffset);
  mipField->setup(result);

  return mipField;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
