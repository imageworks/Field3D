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

/*! \file Resample.h
  \brief Contains functions for resampling fields
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_Resample_H_
#define _INCLUDED_Field3D_Resample_H_

#include "DenseField.h"
#include "SparseField.h"

//----------------------------------------------------------------------------//

/* TODO LIST

 * x Implement dumb, dense resampling
 * x For SparseField, only write non-zero results
 * x Implement more filters
 * For SparseField, be smart about which blocks are computed
 * x Multi-threading using boost
 * Multi-threading using TBB

 */

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Resizing functions 
//----------------------------------------------------------------------------//

//! Resamples the source field into the target field, such that the 
//! new data window is @dataWindow.
//! \note This will query filter.isSeparable() and call separableResample()
//! if possible.
//! \note The extents of the field will be reset to match the data window.
//! This should 
template <typename Field_T, typename FilterOp_T>
bool resample(const Field_T &src, Field_T &tgt, const V3i &newRes,
              const FilterOp_T &filter);

//----------------------------------------------------------------------------//
// BoxFilter
//----------------------------------------------------------------------------//

struct BoxFilter
{
  float operator() (const float t) const
  {
    if (t <= 0.5f) {
      return 1.0f;
    } else {
      return 0.0f;
    }
  }
  float support() const
  { 
    return 0.5f; 
  }
  bool isSeparable() const
  { 
    return true; 
  }
};

//----------------------------------------------------------------------------//
// TriangleFilter
//----------------------------------------------------------------------------//

struct TriangleFilter
{
  float operator() (const float t) const
  {
    if (t > 1.0) {
      return 0.0f;
    }
    return 1.0f - t;
  }
  float support() const
  {
    return 1.0f;
  }
  bool isSeparable() const
  { 
    return true; 
  }
};

//----------------------------------------------------------------------------//
// GaussianFilter
//----------------------------------------------------------------------------//

struct GaussianFilter
{
  GaussianFilter(const float alpha = 2.0, const float width = 2.0)
    : m_alpha(alpha), 
      m_exp(std::exp(-alpha * width * width))
  { /* Empty */ }
  float operator() (const float x) const
  {
    return std::max(0.0f, std::exp(-m_alpha * x * x) - m_exp);
  }
  float support() const
  {
    return 2.0f;
  }
  bool isSeparable() const
  { 
    return true; 
  }
private:
  const float m_alpha, m_exp;
};

//----------------------------------------------------------------------------//
// MitchellFilter
//----------------------------------------------------------------------------//

struct MitchellFilter
{
  MitchellFilter(const float B = 1.0 / 3.0, const float C = 1.0 / 3.0)
    : m_B(B), m_C(C)
  { /* Empty */ }
  float operator() (const float x) const
  {
    const float ax = std::abs(x);
    if (ax < 1) {
      return ((12 - 9 * m_B - 6 * m_C) * ax * ax * ax +
              (-18 + 12 * m_B + 6 * m_C) * ax * ax + (6 - 2 * m_B)) / 6;
    } else if ((ax >= 1) && (ax < 2)) {
      return ((-m_B - 6 * m_C) * ax * ax * ax +
              (6 * m_B + 30 * m_C) * ax * ax + (-12 * m_B - 48 * m_C) *
              ax + (8 * m_B + 24 * m_C)) / 6;
    } else {
      return 0;
    }
  }
  float support() const
  {
    return 2.0f;
  }
  bool isSeparable() const
  { 
    return true; 
  }
private:
  const float m_B, m_C;
};

//----------------------------------------------------------------------------//
// Implementation details
//----------------------------------------------------------------------------//

namespace detail {

  //--------------------------------------------------------------------------//

  Box3i srcSupportBBox(const V3f &tgtP, const float support, const V3i &doUpres,
                       const V3f &srcSize, const V3f &tgtSize);

  //--------------------------------------------------------------------------//

  std::pair<int, int>
  srcSupportBBox(const float &tgtP, const float support, const bool doUpres, 
                 const float &srcSize, const float &tgtSize);

  //--------------------------------------------------------------------------//

  V3f getDist(const V3i &doUpres, const V3f &srcP, const V3f &tgtP, 
              const V3f &srcSize, const V3f &tgtSize);

  //--------------------------------------------------------------------------//

  float getDist(const bool doUpres, const float &srcP, const float &tgtP, 
                const float &srcSize, const float &tgtSize);

  //--------------------------------------------------------------------------//

  template <typename Field_T, typename FilterOp_T>
  void separable(const Field_T &src, Field_T &tgt, const V3i &newRes,
                 const FilterOp_T &filterOp, const size_t dim)
  {
    typedef typename Field_T::value_type T;

    const V3i   srcRes    = src.dataWindow().size() + V3i(1);
    const float srcDomain = V3f(srcRes)[dim];
    const float tgtDomain = V3f(newRes)[dim];
    const float srcSize   = 1.0 / srcDomain;
    const float tgtSize   = 1.0 / tgtDomain;

    // Filter info
    const float support = filterOp.support();

    // Check if we're up-res'ing
    const bool doUpres = newRes[dim] > srcRes[dim] ? 1 : 0;

    // Resize the target
    tgt.setSize(newRes);

    // For each output voxel
    for (int k = 0; k < newRes.z; ++k) {
      for (int j = 0; j < newRes.y; ++j) {
        for (int i = 0; i < newRes.x; ++i) {
          T     accumValue  = static_cast<T>(0.0);
          float accumWeight = 0.0f;
          // Current position in target coordinates
          const float tgtP = discToCont(V3i(i, j ,k)[dim]);
          // Transform support to source coordinates
          std::pair<int, int> srcInterval = 
            srcSupportBBox(tgtP, support, doUpres, srcSize, tgtSize);
          // Clip against new data window
          srcInterval.first = 
            std::max(srcInterval.first, src.dataWindow().min[dim]);
          srcInterval.second = 
            std::min(srcInterval.second, src.dataWindow().max[dim]);
          // For each input voxel
          for (int s = srcInterval.first; s <= srcInterval.second; ++s) {
            // Index
            const int xIdx = dim == 0 ? s : i;
            const int yIdx = dim == 1 ? s : j;
            const int zIdx = dim == 2 ? s : k;
            // Value
            const T value      = src.fastValue(xIdx, yIdx, zIdx);
            // Weights
            const float srcP   = discToCont(V3i(xIdx, yIdx, zIdx)[dim]);
            const float dist   = getDist(doUpres, srcP, tgtP, srcSize, tgtSize);
            const float weight = filterOp(dist);
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

  //! Resamples the source field into the target field, using separable
  //! execution, which is faster than resample().
  //! \note The extents of the field will be reset to match the data window.
  template <typename Field_T, typename FilterOp_T>
  bool separableResample(const Field_T &src, Field_T &tgt, const V3i &newRes,
                         const FilterOp_T &filterOp)
  {
    using namespace detail;
  
    typedef typename Field_T::value_type T;

    if (!filterOp.isSeparable()) {
      return false;
    }

    if (!src.dataWindow().hasVolume()) {
      return false;
    }

    if (src.dataWindow().min != V3i(0)) {
      return false;
    }

    // Temporary field for y component
    Field_T tmp;

    // Cache the old resolution
    V3i oldRes = src.dataWindow().size() + V3i(1);
    V3i xRes(newRes.x, oldRes.y, oldRes.z);
    V3i yRes(newRes.x, newRes.y, oldRes.z);
    V3i zRes(newRes.x, newRes.y, newRes.z);

    // X axis (src into tgt)
    separable(src, tgt, xRes, filterOp, 0);
    // Y axis (tgt into temp)
    separable(tgt, tmp, yRes, filterOp, 1);
    // Z axis (temp into tgt)
    separable(tmp, tgt, zRes, filterOp, 2);

    // Update final target with mapping and metadata
    tgt.name      = src.name;
    tgt.attribute = src.attribute;
    tgt.setMapping(src.mapping());
    tgt.copyMetadata(src);

    return true;
  }

  //--------------------------------------------------------------------------//

} // namespace detail

//----------------------------------------------------------------------------//
// Resizing function implementations
//----------------------------------------------------------------------------//

template <typename Field_T, typename FilterOp_T>
bool resample(const Field_T &src, Field_T &tgt, const V3i &newRes,
              const FilterOp_T &filterOp)
{
  using namespace detail;
  
  typedef typename Field_T::value_type T;

  if (filterOp.isSeparable()) {
    return detail::separableResample(src, tgt, newRes, filterOp);
  }

  if (!src.dataWindow().hasVolume()) {
    return false;
  }

  if (src.dataWindow().min != V3i(0)) {
    return false;
  }

  const V3i srcRes    = src.dataWindow().size() + V3i(1);
  const V3f srcDomain = V3f(srcRes);
  const V3f tgtDomain = V3f(newRes);
  const V3f srcSize   = V3f(1.0) / srcDomain;
  const V3f tgtSize   = V3f(1.0) / tgtDomain;

  // Filter info
  const float support = filterOp.support();

  // Resize the target to the new resolution
  tgt.name      = src.name;
  tgt.attribute = src.attribute;
  tgt.setSize(newRes);
  tgt.setMapping(src.mapping());
  tgt.copyMetadata(src);

  // Check if we're up-res'ing
  const V3i doUpres(newRes.x > srcRes.x ? 1 : 0,
                    newRes.y > srcRes.y ? 1 : 0,
                    newRes.z > srcRes.z ? 1 : 0);

  // For each output voxel
  for (int k = 0; k < newRes.z; ++k) {
    for (int j = 0; j < newRes.y; ++j) {
      for (int i = 0; i < newRes.x; ++i) {
        T     accumValue  = static_cast<T>(0.0);
        float accumWeight = 0.0f;
        // Current position in target coordinates
        const V3f tgtP = discToCont(V3i(i, j ,k));
        // Transform support to source coordinates
        Box3i srcBox = srcSupportBBox(tgtP, support, doUpres, srcSize, tgtSize);
        // Clip against new data window
        srcBox = clipBounds(srcBox, src.dataWindow());
        // For each input voxel
        for (int sk = srcBox.min.z; sk <= srcBox.max.z; ++sk) {
          for (int sj = srcBox.min.y; sj <= srcBox.max.y; ++sj) {
            for (int si = srcBox.min.x; si <= srcBox.max.x; ++si) {
              // Value
              const T value      = src.fastValue(si, sj, sk);
              // Weights
              const V3f srcP     = discToCont(V3i(si, sj, sk));
              const V3f dist     = getDist(doUpres, srcP, tgtP, 
                                           srcSize, tgtSize);
              const float weight = 
                filterOp(dist.x) * filterOp(dist.y) * filterOp(dist.z);
              // Update
              accumWeight += weight;
              accumValue  += value * weight;
            }
          }
        }
        // Update final value
        if (accumWeight > 0.0f && accumValue != static_cast<T>(0.0)) {
          tgt.fastLValue(i, j, k) = accumValue / accumWeight;
        }
      }
    }
  }

  return true;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard

