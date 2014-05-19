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

/*! \file Resample.cpp
  Contains implementations of resampling-related functions.
*/

//----------------------------------------------------------------------------//

#include "Resample.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//

namespace detail {

  //--------------------------------------------------------------------------//

  Box3i srcSupportBBox(const V3f &tgtP, const float support, const V3i &doUpres,
                       const V3f &srcSize, const V3f &tgtSize)
  {
    Box3i srcBox;
    for (int dim = 0; dim < 3; ++dim) {
      if (doUpres[dim]) {
        srcBox.min[dim] = 
          static_cast<int>(std::floor(tgtP[dim] * tgtSize[dim] / srcSize[dim] -
                                      support));
        srcBox.max[dim] = 
          static_cast<int>(std::ceil(tgtP[dim] * tgtSize[dim] / srcSize[dim] +
                                     support)) - 1;
      } else {
        srcBox.min[dim] = 
          static_cast<int>(std::floor((tgtP[dim] - support) *
                                      tgtSize[dim] / srcSize[dim]));
        srcBox.max[dim] = 
          static_cast<int>(std::ceil((tgtP[dim] + support) *
                                     tgtSize[dim] / srcSize[dim]));
      }
    }
    return srcBox;
  }

  //--------------------------------------------------------------------------//

  std::pair<int, int>
  srcSupportBBox(const float &tgtP, const float support, const bool doUpres, 
                 const float &srcSize, const float &tgtSize)
  {
    std::pair<int, int> srcInterval;
    if (doUpres) {
      srcInterval.first = 
        static_cast<int>(std::floor(tgtP * tgtSize / srcSize - support));
      srcInterval.second = 
        static_cast<int>(std::ceil(tgtP * tgtSize / srcSize + support)) - 1;
    } else {
      srcInterval.first = 
        static_cast<int>(std::floor((tgtP - support) * tgtSize / srcSize));
      srcInterval.second = 
        static_cast<int>(std::ceil((tgtP + support) * tgtSize / srcSize));
    }
    return srcInterval;
  }

  //--------------------------------------------------------------------------//

  V3f getDist(const V3i &doUpres, const V3f &srcP, const V3f &tgtP, 
              const V3f &srcSize, const V3f &tgtSize)
  {
    V3f dist;
    for (int dim = 0; dim < 3; ++dim) {
      if (doUpres[dim]) {
        const float tgtSrc = tgtP[dim] * tgtSize[dim] / srcSize[dim];
        dist[dim]          = std::abs(tgtSrc - srcP[dim]);
      } else {
        const float srcTgt = srcP[dim] * srcSize[dim] / tgtSize[dim];
        dist[dim]          = std::abs(srcTgt - tgtP[dim]);
      } 
    }
    return dist;
  }

  //--------------------------------------------------------------------------//

  float getDist(const bool doUpres, const float &srcP, const float &tgtP, 
                const float &srcSize, const float &tgtSize)
  {
    if (doUpres) {
      const float tgtSrc = tgtP * tgtSize / srcSize;
      return std::abs(tgtSrc - srcP);
    } else {
      const float srcTgt = srcP * srcSize / tgtSize;
      return std::abs(srcTgt - tgtP);
    } 
  }

  //--------------------------------------------------------------------------//

} // namespace detail

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
