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

/*! \file MIPUtil.cpp
  Contains implementations of resampling-related functions.
*/

//----------------------------------------------------------------------------//

#include "MIPUtil.h"

#include <boost/foreach.hpp>

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//

namespace detail {

  //--------------------------------------------------------------------------//

  V3i mipResolution(const V3i &baseRes, const size_t level)
  {
    const float factor = 1.0 / (1 << level);
    const V3f   floatRes(baseRes);
    return V3i(static_cast<int>(ceil(floatRes.x * factor)),
               static_cast<int>(ceil(floatRes.y * factor)),
               static_cast<int>(ceil(floatRes.z * factor)));
  }

  //--------------------------------------------------------------------------//

  FieldMapping::Ptr adjustedMIPFieldMapping(const FieldMapping::Ptr mapping, 
                                            const V3i &baseRes,
                                            const size_t level)
  {
    typedef MatrixFieldMapping::MatrixCurve MatrixCurve;

    if (MatrixFieldMapping::Ptr mfm = 
        field_dynamic_cast<MatrixFieldMapping>(mapping)) {
      // Determine padding
      const int mult          = 1 << level;
      const V3i currentRes    = mipResolution(baseRes, level);
      const V3i currentAtBase = currentRes * mult;
      const V3f padding       = V3f(currentAtBase) / V3f(baseRes);
      // Grab the matrices
      const MatrixCurve::SampleVec lsToWsSamples = mfm->localToWorldSamples();
      // New mapping to construct
      MatrixFieldMapping::Ptr newMapping(new MatrixFieldMapping);
      // For each matrix, append the padding
      BOOST_FOREACH (const MatrixCurve::Sample &sample, lsToWsSamples) {
        M44d lsToWs = sample.second;
        M44d scaling;
        scaling.setScale(padding);
        newMapping->setLocalToWorld(sample.first, scaling * lsToWs);
      }
      // Done
      return newMapping;
    } else {
      // For non-uniform grids, there is nothing we can do. 
      return mapping;
    }
  }

  //--------------------------------------------------------------------------//

} // namespace detail

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
