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
// Filter
//----------------------------------------------------------------------------//

struct Filter
{
  // Typedefs ---

  typedef boost::shared_ptr<Filter>       Ptr;
  typedef boost::shared_ptr<const Filter> CPtr;

  // To be overridden by subclasses ---

  //! Evaluates the filter at coordinate 't'
  virtual float eval(const float t) const = 0;
  //! Radial width of the filter (half of diameter)
  virtual float support()           const = 0;

  // May be overridden by subclasses ---

  //! Initial value (zero by default, but need to be different for min/max)
  virtual float initialValue()      const
  { return 0.0f; }

};

//----------------------------------------------------------------------------//
// BoxFilter
//----------------------------------------------------------------------------//

struct BoxFilter : public Filter
{
  // Typedefs
  typedef boost::shared_ptr<BoxFilter>       Ptr;
  typedef boost::shared_ptr<const BoxFilter> CPtr;

  static const bool isAnalytic = false;

  // Ctors
  BoxFilter()
    : m_width(1.0)
  { }
  BoxFilter(const float width)
    : m_width(width)
  { }
  // From Filter base class 
  virtual float eval(const float x) const
  {
    const float t = x / m_width;
    if (t <= 0.5f) {
      return 1.0f;
    } else {
      return 0.0f;
    }
  }
  virtual float support() const
  { 
    return 0.5f * m_width; 
  }
  template <typename Value_T>
  static void op(Value_T &accumValue, const Value_T value) 
  { /* no-op */ }
private:
  const float m_width;
};

//----------------------------------------------------------------------------//
// MinFilter
//----------------------------------------------------------------------------//

struct MinFilter : public Filter
{
  // Typedefs
  typedef boost::shared_ptr<MinFilter>       Ptr;
  typedef boost::shared_ptr<const MinFilter> CPtr;

  static const bool isAnalytic = true;

  // Ctors
  MinFilter()
    : m_width(1.0)
  { }
  MinFilter(const float width)
    : m_width(width)
  { }
  // From Filter base class 
  virtual float eval(const float x) const
  {
    const float t = x / m_width;
    if (t <= 0.5f) {
      return 1.0f;
    } else {
      return 0.0f;
    }
  }
  virtual float support() const
  { 
    return 0.5f * m_width; 
  }
  virtual float initialValue() const
  {
    return std::numeric_limits<float>::max();
  }

  template <typename T>
  static void op(Imath::Vec3<T> &accumValue, const Imath::Vec3<T> value)
  {
    accumValue.x = std::min(accumValue.x, value.x);
    accumValue.y = std::min(accumValue.y, value.y);
    accumValue.z = std::min(accumValue.z, value.z);
  }

  template <typename Value_T>
  static void op(Value_T &accumValue, const Value_T value)
  {
    accumValue = std::min(accumValue, value);
  }

private:
  const float m_width;
};

//----------------------------------------------------------------------------//
// MaxFilter
//----------------------------------------------------------------------------//

struct MaxFilter : public Filter
{
  // Typedefs
  typedef boost::shared_ptr<MaxFilter>       Ptr;
  typedef boost::shared_ptr<const MaxFilter> CPtr;

  static const bool isAnalytic = true;

  // Ctors
  MaxFilter()
    : m_width(1.0)
  { }
  MaxFilter(const float width)
    : m_width(width)
  { }
  // From Filter base class 
  virtual float eval(const float x) const
  {
    const float t = x / m_width;
    if (t <= 0.5f) {
      return 1.0f;
    } else {
      return 0.0f;
    }
  }
  virtual float support() const
  { 
    return 0.5f * m_width; 
  }
  virtual float initialValue() const
  {
    return -std::numeric_limits<float>::max();
  }


  template <typename T>
  static void op(Imath::Vec3<T> &accumValue, const Imath::Vec3<T> value)
  {
    accumValue.x = std::max(accumValue.x, value.x);
    accumValue.y = std::max(accumValue.y, value.y);
    accumValue.z = std::max(accumValue.z, value.z);
  }

  template <typename Value_T>
  static void op(Value_T &accumValue, const Value_T value)
  {
    accumValue = std::max(accumValue, value);
  }

private:
  const float m_width;
};

//----------------------------------------------------------------------------//
// TriangleFilter
//----------------------------------------------------------------------------//

struct TriangleFilter : public Filter
{
  // Typedefs
  typedef boost::shared_ptr<TriangleFilter>       Ptr;
  typedef boost::shared_ptr<const TriangleFilter> CPtr;

  static const bool isAnalytic = false;

  // Ctors
  TriangleFilter()
    : m_width(1.0)
  { }
  TriangleFilter(const float width)
    : m_width(width)
  { }
  // From Filter base class 
  virtual float eval(const float x) const
  {
    const float t = x / m_width;
    if (t > 1.0) {
      return 0.0f;
    }
    return 1.0f - t;
  }
  virtual float support() const
  {
    return 1.0f * m_width;
  }
  template <typename Value_T>
  static void op(Value_T &/*accumValue*/, const Value_T /*value*/)
  { /* No-op */ }
private:
  const float m_width;
};

//----------------------------------------------------------------------------//
// GaussianFilter
//----------------------------------------------------------------------------//

struct GaussianFilter : public Filter
{
  // Typedefs
  typedef boost::shared_ptr<GaussianFilter>       Ptr;
  typedef boost::shared_ptr<const GaussianFilter> CPtr;

  static const bool isAnalytic = false;

  // Ctor 
  GaussianFilter(const float alpha = 2.0, const float width = 2.0)
    : m_alpha(alpha), 
      m_exp(std::exp(-alpha * width * width)),
      m_width(width)
  { /* Empty */ }
  // From Filter base class 
  virtual float eval(const float t) const
  {
    const float x = t / m_width;
    return std::max(0.0f, std::exp(-m_alpha * x * x) - m_exp);
  }
  virtual float support() const
  {
    return 2.0f * m_width;
  }
  template <typename Value_T>
  static void op(Value_T &accumValue, const Value_T value)
  { /* No-op */ }
private:
  const float m_alpha, m_exp, m_width;
};

//----------------------------------------------------------------------------//
// MitchellFilter
//----------------------------------------------------------------------------//

struct MitchellFilter : public Filter
{
  // Typedefs
  typedef boost::shared_ptr<MitchellFilter>       Ptr;
  typedef boost::shared_ptr<const MitchellFilter> CPtr;

  static const bool isAnalytic = false;

  // Ctor 
  MitchellFilter(const float width = 1.0, 
                 const float B = 1.0 / 3.0, const float C = 1.0 / 3.0)
    : m_B(B), m_C(C), m_width(width)
  { /* Empty */ }
  // From Filter base class 
  virtual float eval(const float x) const
  {
    const float ax = std::abs(x / m_width);
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
  virtual float support() const
  {
    return 2.0f * m_width;
  }
  template <typename Value_T>
  static void op(Value_T &accumValue, const Value_T value)
  { /* No-op */ }
private:
  const float m_B, m_C;
  const float m_width;
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

  template <typename Field_T, typename FilterOp_T, bool IsAnalytic_T>
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
          T accumValue(filterOp.initialValue());
          if (IsAnalytic_T) {
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
              const T value = src.fastValue(xIdx, yIdx, zIdx);
              // Weights
              const float srcP   = discToCont(V3i(xIdx, yIdx, zIdx)[dim]);
              const float dist   = getDist(doUpres, srcP, tgtP, srcSize, tgtSize);
              const float weight = filterOp.eval(dist);
              // Update
              if (weight > 0.0f) {
                FilterOp_T::op(accumValue, value);
              }
            }
            // Update final value
            if (accumValue != static_cast<T>(filterOp.initialValue())) {
              tgt.fastLValue(i, j, k) = accumValue;
            }
          } else {
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
              const float weight = filterOp.eval(dist);
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
    separable<Field_T, FilterOp_T, FilterOp_T::isAnalytic>(src, tgt, xRes, filterOp, 0);
    // Y axis (tgt into temp)
    separable<Field_T, FilterOp_T, FilterOp_T::isAnalytic>(tgt, tmp, yRes, filterOp, 1);
    // Z axis (temp into tgt)
    separable<Field_T, FilterOp_T, FilterOp_T::isAnalytic>(tmp, tgt, zRes, filterOp, 2);

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
  return detail::separableResample(src, tgt, newRes, filterOp);
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard

