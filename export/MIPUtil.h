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
#include <boost/thread/condition.hpp>

#include "Resample.h"
#include "Types.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Functions
//----------------------------------------------------------------------------//

//! Constructs a MIP representation of the given field.
template <class MIPField_T>
typename MIPField_T::Ptr
makeMIP(const typename MIPField_T::NestedType &base, const int minSize = 32);

//----------------------------------------------------------------------------//
// Implementation details
//----------------------------------------------------------------------------//

namespace detail {

  //--------------------------------------------------------------------------//

  V3i mipResolution(const V3i &baseRes, const size_t level);

  //--------------------------------------------------------------------------//

  template <typename Field_T, typename FilterOp_T>
  void mipSeparable(const Field_T &src, Field_T &tgt, 
                    const V3i &oldRes, const V3i &newRes, const size_t level, 
                    const FilterOp_T &filterOp, const size_t dim)
  {
    typedef typename Field_T::value_type T;

    using namespace std;

    // To ensure we don't sample outside source data
    Box3i srcDw = src.dataWindow();

    // Filter info
    const float support = filterOp.support();

    // Compute new res
    V3i res;
    if (dim == 2) {
      res = newRes;
    } else if (dim == 1) {
      res = V3i(newRes.x, newRes.y, oldRes.z);
    } else {
      res = V3i(newRes.x, oldRes.y, oldRes.z);
    }

    // Coordinate frame conversion constants
    const float tgtToSrcMult    = static_cast<float>(1 << level);
    const float filterCoordMult = 1.0f / (tgtToSrcMult);
    
    // Resize new field
    tgt.setSize(res);

    // For each output voxel
    for (int k = 0; k < res.z; ++k) {
      for (int j = 0; j < res.y; ++j) {
        for (int i = 0; i < res.x; ++i) {
          T     accumValue   = static_cast<T>(0.0);
          float accumWeight  = 0.0f;
          // Transform from current point in target frame to source frame
          const int   curTgt = V3i(i, j, k)[dim];
          const float curSrc = discToCont(curTgt) * tgtToSrcMult;
          // Find interval
          int startSrc = 
            static_cast<int>(std::floor(curSrc - support * tgtToSrcMult));
          int endSrc   = 
            static_cast<int>(std::ceil(curSrc + support * tgtToSrcMult)) - 1;
          startSrc     = std::max(startSrc, srcDw.min[dim]);
          endSrc       = std::min(endSrc, srcDw.max[dim]);
          // Loop over source voxels
          for (int s = startSrc; s <= endSrc; ++s) {
            // Source index
            const int xIdx = dim == 0 ? s : i;
            const int yIdx = dim == 1 ? s : j;
            const int zIdx = dim == 2 ? s : k;
            // Source voxel in continuous coords
            const float srcP   = discToCont(s);
            // Compute filter weight in source space (twice as wide)
            const float weight = filterOp(std::abs(srcP - curSrc) * 
                                          filterCoordMult);
            // Value
            const T value      = src.fastValue(xIdx, yIdx, zIdx);
            // Update
            accumWeight += weight;
            accumValue  += value * weight;
          }
          // Update final value
          if (accumWeight > 0.0f && accumValue != static_cast<T>(0.0)) {
            tgt.fastLValue(i, j, k) = accumValue / accumWeight;
          }
        }
      }
    }

  }

  //--------------------------------------------------------------------------//

  template <typename Field_T, typename FilterOp_T>
  void mipResample(const Field_T &src, Field_T &tgt, const size_t level, 
                   const FilterOp_T &filterOp)
  {
    using std::ceil;

    // Compute new res
    const Box3i srcDw  = src.dataWindow();
    const V3i   oldRes = srcDw.size() + V3i(1);
    const V3i   newRes = mipResolution(oldRes, level);

    // Temporary field for y component
    Field_T tmp;

    // X axis (src into tgt)
    mipSeparable(src, tgt, oldRes, newRes, level, filterOp, 0);
    // Y axis (tgt into temp)
    mipSeparable(tgt, tmp, oldRes, newRes, level, filterOp, 1);
    // Z axis (temp into tgt)
    mipSeparable(tmp, tgt, oldRes, newRes, level, filterOp, 2);

    // Update final target with mapping and metadata
    tgt.name      = src.name;
    tgt.attribute = src.attribute;
    tgt.setMapping(src.mapping());
    tgt.copyMetadata(src);
  }

  //--------------------------------------------------------------------------//

  FieldMapping::Ptr adjustedMIPFieldMapping(const FieldMapping::Ptr baseMapping,
                                            const V3i &baseRes,
                                            const size_t level);

  //--------------------------------------------------------------------------//

} // namespace detail

//----------------------------------------------------------------------------//
// Function implementations
//----------------------------------------------------------------------------//

template <class MIPField_T>
typename MIPField_T::Ptr
makeMIP(const typename MIPField_T::NestedType &base, const int minSize)
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
  V3i res = base.extents().size() + V3i(1);
  
  // Loop until minimum size is found
  size_t level = 1;
  while (res.x > minSize || res.y > minSize || res.z > minSize) {
    // Perform filtering
    SrcPtr nextField(new Src_T);
    mipResample(base, *nextField, level, MitchellFilter());
    // Add to vector of filtered fields
    result.push_back(nextField);
    // Set up for next iteration
    res = nextField->dataWindow().size() + V3i(1);
    level++;
  }

  MIPPtr mipField(new MIPField_T);
  mipField->setup(result);
  mipField->name = base.name;
  mipField->attribute = base.attribute;
  mipField->copyMetadata(base);

  return mipField;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
