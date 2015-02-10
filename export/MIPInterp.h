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

/*! \file MIPInterp.h
  \brief Contains MIPInterp class
  \ingroup field
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_MIPInterp_H_
#define _INCLUDED_Field3D_MIPInterp_H_

#include "MIPField.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// MIPInterp 
//----------------------------------------------------------------------------//

template <typename MIPField_T>
class MIPLinearInterp : boost::noncopyable
{
public:

  // Typedefs ---

  typedef typename MIPField_T::NestedType  FieldType;
  typedef typename FieldType::LinearInterp LinearInterpType;
  typedef typename FieldType::value_type   value_type;

  // Ctors ---

  //! Must be constructed with a MIP field to operate on
  MIPLinearInterp(const MIPField_T &mip);

  // Main methods ---

  //! Performs interpolation. A MIP field interpolation requires a spot
  //! size (which may be zero, forcing a lookup in the 0-level field).
  value_type sample(const V3d &vsP, const float wsSpotSize) const;

private:

  // Structs ---

  struct InterpInfo
  {

    // Ctors ---

    InterpInfo()
      : lower(0), upper(0), lerpT(0.0f)
    { }
    InterpInfo(const size_t l, const size_t u, const float t)
      : lower(l), upper(u), lerpT(t)
    { }

    // Data members ---

    //! Finest level
    size_t lower;
    //! Coarser level
    size_t upper;
    //! Parametric position between finest and coarser
    float  lerpT;
  };

  // Utility methods ---

  //! Computes between which levels to interpolate
  InterpInfo interpInfo(const float wsSpotSize) const;

  // Data members ---

  //! Const reference to MIP field
  const MIPField_T& m_mip;
  //! Min world space voxel size for each MIP level
  std::vector<float> m_wsVoxelSize;
  //! Linear interpolator
  LinearInterpType m_interp;

};

//----------------------------------------------------------------------------//
// MIPLinearInterp implementations
//----------------------------------------------------------------------------//

template <typename MIPField_T>
MIPLinearInterp<MIPField_T>::MIPLinearInterp(const MIPField_T &mip)
  : m_mip(mip)
{
  // Base voxel size (represents finest level)
  const V3f   wsVoxelSize    = mip.mapping()->wsVoxelSize(0, 0, 0);
  const float wsMinVoxelSize = 
    std::min(std::min(wsVoxelSize.x, wsVoxelSize.y), wsVoxelSize.z);
  // All subsequent levels are a 2x mult on the base voxel size
  for (size_t i = 0, end = mip.numLevels(); i < end; ++i) {
    const float factor = std::pow(2.0f, static_cast<float>(i));
    m_wsVoxelSize.push_back(wsMinVoxelSize * factor);
  }
}

//----------------------------------------------------------------------------//

template <typename MIPField_T>
typename MIPLinearInterp<MIPField_T>::value_type
MIPLinearInterp<MIPField_T>::sample(const V3d &vsP, 
                                    const float wsSpotSize) const
{
  const InterpInfo i = interpInfo(wsSpotSize);

  if (i.lower == i.upper) {
    // Special case where spot size is not in-between levels
    if (i.lower == 0) {
      // Special case for 0-level
      return m_interp.sample(*m_mip.rawMipLevel(0), vsP);
    } else {
      // Not the 0-level, so we must transform vsP
      V3f mipVsP;
      m_mip.getVsMIPCoord(vsP, i.lower, mipVsP);
      return m_interp.sample(*m_mip.rawMipLevel(i.lower), mipVsP);
    }
  } else {
    // Quadrilinear interpolation
    V3f mipVsP0, mipVsP1;
    m_mip.getVsMIPCoord(V3f(vsP), i.lower, mipVsP0);
    m_mip.getVsMIPCoord(V3f(vsP), i.upper, mipVsP1);
    const value_type v0 = m_interp.sample(*m_mip.rawMipLevel(i.lower), mipVsP0);
    const value_type v1 = m_interp.sample(*m_mip.rawMipLevel(i.upper), mipVsP1);
    return FIELD3D_LERP(v0, v1, i.lerpT);
  }
}

//----------------------------------------------------------------------------//

template <typename MIPField_T>
typename MIPLinearInterp<MIPField_T>::InterpInfo
MIPLinearInterp<MIPField_T>::interpInfo(const float wsSpotSize) const
{
  const size_t numLevels = m_mip.numLevels();
  // First case, spot size smaller than first level
  if (wsSpotSize <= m_wsVoxelSize[0]) {
    return InterpInfo();
  }
  // Check in-between sizes
  for (size_t i = 1, end = numLevels; i < end; ++i) {
    if (wsSpotSize <= m_wsVoxelSize[i]) {
      const float lerpT = FIELD3D_LERPFACTOR(wsSpotSize, m_wsVoxelSize[i - 1],
                                             m_wsVoxelSize[i]);
      return InterpInfo(i - 1, i, lerpT);
    }
  }
  // Final case, spot size larger or equal to highest level
  return InterpInfo(numLevels - 1, numLevels - 1, 0.0f);
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard

