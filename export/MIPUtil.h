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
#include "TemporalField.h"
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

  V3i mipResolution(const V3i &baseRes, const size_t level, const V3i &add);

  //--------------------------------------------------------------------------//

  template <typename Data_T>
  bool checkInputEmpty(const SparseField<Data_T> &src, 
                       const SparseField<Data_T> &tgt, 
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

  template <typename Data_T>
  bool checkInputEmpty(const TemporalField<Data_T> &src, 
                       const TemporalField<Data_T> &tgt, 
                       const Box3i &tgtBox, const float support,
                       const size_t dim)
  {
    typedef typename TemporalField<Data_T>::Block Block;
    typedef typename Block::State                 BlockState;

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
          const Block      *block = src.block(i, j, k);
          const BlockState  state = block->state();
          if (state == Block::ActiveState) {
            return false;
          }
          if (state == Block::EmptyState) {
            if (block->values()[0] != static_cast<Data_T>(0)) {
              return false;
            }
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
  bool checkInputEmpty(const Field_T &src, const Field_T &tgt, 
                       const Box3i &tgtBox, const float support,
                       const size_t dim)
  {
    return false;
  }

  //--------------------------------------------------------------------------//

  template <typename Field_T, typename FilterOp_T, bool IsAnalytic_T>
  struct MIPSeparableThreadOp
  {

    //------------------------------------------------------------------------//

    typedef typename Field_T::value_type T;

    //------------------------------------------------------------------------//

    MIPSeparableThreadOp(const Field_T &src, Field_T &tgt, 
                         const size_t level, const V3i &add,
                         const FilterOp_T &filterOp, 
                         const size_t dim, 
                         size_t &nextIdx, boost::mutex &mutex)
      : m_src(src),
        m_tgt(tgt),
        m_filterOp(filterOp), 
        m_level(level), 
        m_add(add), 
        m_dim(dim),
        m_nextIdx(nextIdx),
        m_mutex(mutex),
        m_numGrains(tgt.numGrains())
    {
      // Empty
    }

    //------------------------------------------------------------------------//

    //! Returns temporal value
    T voxelValue(const TemporalField<T> &src, 
                 const int i, const int j, const int k, const float time)
    {
      return src.fastValue(i, j, k, time);
    }

    //------------------------------------------------------------------------//

    //! Returns non-temporal value
    template <template <typename X> class Field2_T>
    T voxelValue(const Field2_T<T> &src, 
                 const int i, const int j, const int k, const float /* time */)
    {
      return src.fastValue(i, j, k);
    }

    //------------------------------------------------------------------------//

    //! Sets temporal sample value
    void setVoxelValue(TemporalField<T> &tgt, 
                       const int i, const int j, const int k, const int t,
                       const T &value)
    {
      tgt.sampleValues(i, j, k)[t] = value;
    }

    //------------------------------------------------------------------------//

    //! Sets non-temporal value
    template <template <typename X> class Field2_T>
    void setVoxelValue(Field2_T<T> &tgt, 
                       const int i, const int j, const int k, const int /* t */,
                       const T &value) 
    {
      tgt.fastLValue(i, j, k) = value;
    }

    //------------------------------------------------------------------------//

    void filter(const Field_T &src, Field_T &tgt, 
                const int i, const int j, const int k, const int t,
                const float time)
    {
      // Defer to ComputationType to determine the processing data type
      typedef typename Field_T::value_type           Data_T;
      typedef typename ComputationType<Data_T>::type Value_T;

      // To ensure we don't sample outside source data
      Box3i srcDw = src.dataWindow();

      // Coordinate frame conversion constants
      const float tgtToSrcMult    = 2.0;
      const float filterCoordMult = 1.0f / (tgtToSrcMult);
    
      // Filter info, support size in target space
      const float support = m_filterOp.support();

      Value_T accumValue(m_filterOp.initialValue());

      // Transform from current point in target frame to source frame
      const int   curTgt = V3i(i, j, k)[m_dim];
      const float curSrc = discToCont(curTgt) * tgtToSrcMult - m_add[m_dim];
      // Find interval
      int startSrc = 
        static_cast<int>(std::floor(curSrc - support * tgtToSrcMult));
      int endSrc = 
        static_cast<int>(std::ceil(curSrc + support * tgtToSrcMult)) - 1;
      // Clamp coordinates
      startSrc = std::max(startSrc, srcDw.min[m_dim]);
      endSrc   = std::min(endSrc, srcDw.max[m_dim]);

      if (IsAnalytic_T) {

        // Analytic ---

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
          const Value_T value = voxelValue(src, xIdx, yIdx, zIdx, time);
          // Update
          if (weight > 0.0f) {
            FilterOp_T::op(accumValue, value);
          }
        }

        // Update final value
        if (accumValue != 
            static_cast<Value_T>(m_filterOp.initialValue())) {
          setVoxelValue(tgt, i, j, k, t, accumValue);
        }

      } else {

        // Non-analytic ---
        
        float accumWeight  = 0.0f;
        
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
          const Value_T value = voxelValue(src, xIdx, yIdx, zIdx, time);
          // Update
          accumWeight += weight;
          accumValue  += value * weight;
        }

        // Update final value
        if (accumWeight > 0.0f && 
            accumValue != static_cast<Value_T>(0.0)) {
          setVoxelValue(tgt, i, j, k, t, accumValue / accumWeight);
        }

      }

    }

    //------------------------------------------------------------------------//

    void processVoxel(const TemporalField<T> &src, TemporalField<T> &tgt,
                      const int i, const int j, const int k)
    {
      for (int t = 0, tEnd = tgt.numSamples(i, j, k); t < tEnd; ++t) {
        const float time = tgt.sampleTimes(i, j, k)[t];
        filter(src, tgt, i, j, k, t, time);
      }
    }

    //------------------------------------------------------------------------//

    template <template <typename X> class Field2_T>
    void processVoxel(const Field2_T<T> &src, Field2_T<T> &tgt,
                      const int i, const int j, const int k)
    {
      filter(src, tgt, i, j, k, 0, 0.0f);
    }

    //------------------------------------------------------------------------//

    bool setupBlock(const TemporalField<T> &src, TemporalField<T> &tgt,
                    const size_t idx)
    {
      typedef typename TemporalField<T>::Block Block;

      // To ensure we don't sample outside source data
      Box3i srcDw = src.dataWindow();

      // Coordinate frame conversion constants
      const float tgtToSrcMult    = 2.0;
    
      // Filter info, support size in target space
      const float support = m_filterOp.support();

      // Grab reference to temporal block
      Box3i box;
      tgt.getGrainBounds(idx, box);
      Block *block = tgt.block(box);

      // Error check
      if (!block) {
        std::cout << "ERROR: Bad temporal block bounds: " 
                  << box.min << "-" << box.max << std::endl;
        return false;
      }

      // Block info
      const int    blockOrder = tgt.blockOrder();
      const int    blockSize  = 1 << blockOrder;
      const size_t numVoxels  = Block::numVoxelsPerBlock(blockOrder);

      // Allocate storage for block data
      int *offsets = new int[numVoxels + 1];

      // Allocate temp storage for min/max time per voxel
      const float maxF = std::numeric_limits<float>::max();
      std::vector<float> minTimes(numVoxels, maxF), maxTimes(numVoxels, -maxF);

      // For each target voxel, count the number of source voxels and source
      // samples, divide by number of source voxels to keep track of 
      // total number of samples needed
      
      // Iteration variables
      size_t numBlockSamples = 0;      
      size_t oIdx            = 0;
      
      for (int k = box.min.z; k < box.min.z + blockSize; ++k) {
        for (int j = box.min.y; j < box.min.y + blockSize; ++j) {
          for (int i = box.min.x; i < box.min.x + blockSize; ++i, ++oIdx) {
            // Record current offset as current sample count
            offsets[oIdx] = numBlockSamples;
            // Transform from current point in target frame to source frame
            const int   curTgt = V3i(i, j, k)[m_dim];
            const float curSrc = 
              discToCont(curTgt) * tgtToSrcMult - m_add[m_dim];
            // Find interval
            int startSrc = 
              static_cast<int>(std::floor(curSrc - support * tgtToSrcMult));
            int endSrc = 
              static_cast<int>(std::ceil(curSrc + support * tgtToSrcMult)) - 1;
            // Clamp coordinates
            startSrc = std::max(startSrc, srcDw.min[m_dim]);
            endSrc   = std::min(endSrc, srcDw.max[m_dim]);
            // Only check voxels in processing box
            if (box.intersects(V3i(i, j, k))) {
              // Number of source voxels and samples
              size_t numSamples = 0;
              // Loop over source voxels
              for (int s = startSrc; s <= endSrc; ++s) {
                // Increment number of samples
                const int xIdx = m_dim == 0 ? s : i;
                const int yIdx = m_dim == 1 ? s : j;
                const int zIdx = m_dim == 2 ? s : k;
                const size_t numVoxelSamples = src.numSamples(xIdx, yIdx, zIdx);
                numSamples = std::max(numVoxelSamples, numSamples);
                // Update min/max time
                for (size_t iTime = 0; iTime < numVoxelSamples; ++iTime) {
                  const float time = src.sampleTimes(xIdx, yIdx, zIdx)[iTime];
                  minTimes[oIdx] = std::min(minTimes[oIdx], time);
                  maxTimes[oIdx] = std::max(maxTimes[oIdx], time);
                }
              }
              // Compute number of samples to keep
              numBlockSamples += numSamples;
            }
          }
        }
      }

      // Record the very last offset
      offsets[oIdx] = numBlockSamples;
      
      // Allocate storage for time data
      float *times = new float[numBlockSamples];
      
      // Reset oIdx to zero so we can loop again
      oIdx = 0;

      // Iteration variables
      size_t sIdx = 0;

      for (int k = box.min.z; k < box.min.z + blockSize; ++k) {
        for (int j = box.min.y; j < box.min.y + blockSize; ++j) {
          for (int i = box.min.x; i < box.min.x + blockSize; ++i, ++oIdx) {
            if (box.intersects(V3i(i, j, k))) {
              // Space samples evenly in time
              const size_t numSamples = offsets[oIdx + 1] - offsets[oIdx];
              for (size_t s = 0; s < numSamples; ++s, ++sIdx) {
                const float t = static_cast<float>(s) / (numSamples - 1);
                if (minTimes[oIdx] <= maxTimes[oIdx]) {
                  times[sIdx] = Imath::lerp(minTimes[oIdx], maxTimes[oIdx], t);
                } else {
                  times[sIdx] = Imath::lerp(-1.0, 1.0, t);
                }
              }
            }
          }
        }
      }

      // Allocate storage for value data
      T *values = new T[numBlockSamples];
      std::fill_n(values, numBlockSamples, T(0.0));

      // Add arrays to the block
      block->setArrays(offsets, times, values);

      return true;
    }

    //------------------------------------------------------------------------//

    template <template <typename X> class Field2_T>
    bool setupBlock(const Field2_T<T> &src, Field2_T<T> &tgt, const size_t idx)
    {
      return true;
    }

    //------------------------------------------------------------------------//

    void operator() () 
    {
      using namespace std;

      // Defer to ComputationType to determine the processing data type
      typedef typename Field_T::value_type           Data_T;
      typedef typename ComputationType<Data_T>::type Value_T;

      // To ensure we don't sample outside source data
      Box3i srcDw = m_src.dataWindow();

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
      while (idx < m_numGrains) {
        // Grab the bounds
        Box3i box;
        m_tgt.getGrainBounds(idx, box);
        // Early exit if input blocks are all empty
        if (!detail::checkInputEmpty(m_src, m_tgt, box, support, m_dim)) {
          // Set up the block (specialized for TemporalField, pass-through
          // for others)
          if (setupBlock(m_src, m_tgt, idx)) {
            // For each output voxel
            for (int k = box.min.z; k <= box.max.z; ++k) {
              for (int j = box.min.y; j <= box.max.y; ++j) {
                for (int i = box.min.x; i <= box.max.x; ++i) {
                  processVoxel(m_src, m_tgt, i, j, k);
                }
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

    //------------------------------------------------------------------------//

  private:

    // Data members ---

    const Field_T            &m_src;
    Field_T                  &m_tgt;
    const FilterOp_T         &m_filterOp;
    const size_t              m_level;
    const V3i                &m_add;
    const size_t              m_dim;
    size_t                   &m_nextIdx;
    boost::mutex             &m_mutex;
    const size_t              m_numGrains;
    
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
    
    // Next index counter and mutex
    size_t nextIdx = 0;
    boost::mutex mutex;

    // Launch threads ---

    boost::thread_group threads;

    for (size_t i = 0; i < numThreads; ++i) {
      threads.create_thread(
        MIPSeparableThreadOp<Field_T, FilterOp_T, FilterOp_T::isAnalytic >
        (src, tgt, level, add, filterOp, dim, nextIdx, mutex));
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
