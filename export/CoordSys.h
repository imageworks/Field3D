//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2014 Sony Pictures Imageworks Inc
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

/*! \file CoordSys.h
  \brief Contains utility functions for constructing coordinate systems. 
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_CoordSys_H_
#define _INCLUDED_Field3D_CoordSys_H_

//----------------------------------------------------------------------------//

// System includes
#include <cmath>

#include "Types.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Utility functions
//----------------------------------------------------------------------------//

//! Constructs a coordinate systems given a set of basis vectors and an origin.
template <typename T>
FIELD3D_MTX_T<T> coordinateSystem(const FIELD3D_VEC3_T<T> &e1,
                                  const FIELD3D_VEC3_T<T> &e2,
                                  const FIELD3D_VEC3_T<T> &e3,
                                  const FIELD3D_VEC3_T<T> &origin);

//! Constructs a coordinate system given a bounding box
template <typename T>
FIELD3D_MTX_T<T> 
coordinateSystem(const FIELD3D_BOX_T<FIELD3D_VEC3_T<T> > &wsBounds);

//! Constructs a coordinate system that has its lower left corner at an 
//! even multiplier of the voxel-size, to ensure that voxel centers don't
//! shift as the domain grows
template <typename T>
FIELD3D_MTX_T<T> 
coordinateSystem(const FIELD3D_BOX_T<FIELD3D_VEC3_T<T> >  &wsBounds, 
                 const FIELD3D_VEC3_T<T>                  &wsVoxelSize,
                 Box3i                                    &extents);

//! Constructs a coordinate system that has its lower left corner at an 
//! even multiplier of the voxel-size, to ensure that voxel centers don't
//! shift as the domain grows
template <typename T>
FIELD3D_MTX_T<T> 
coordinateSystem(const FIELD3D_BOX_T<FIELD3D_VEC3_T<T> >  &wsBounds, 
                 const FIELD3D_VEC3_T<T>                  &wsVoxelSize);

//----------------------------------------------------------------------------//
// Detail namespace
//----------------------------------------------------------------------------//

namespace detail {
  
  //--------------------------------------------------------------------------//

  //! Floor function for Vec3
  template <typename T>
  FIELD3D_VEC3_T<T>
  floor(const FIELD3D_VEC3_T<T> &v)
  {
    return FIELD3D_VEC3_T<T>(std::floor(v.x), std::floor(v.y), std::floor(v.z));
  }

  //--------------------------------------------------------------------------//

  //! Ceil function for Vec3
  template <typename T>
  FIELD3D_VEC3_T<T>
  ceil(const FIELD3D_VEC3_T<T> &v)
  {
    return FIELD3D_VEC3_T<T>(std::ceil(v.x), std::ceil(v.y), std::ceil(v.z));
  }

  //--------------------------------------------------------------------------//

} // Detail namespace

//----------------------------------------------------------------------------//
// Template implementations
//----------------------------------------------------------------------------//

template <typename T>
FIELD3D_MTX_T<T> coordinateSystem(const FIELD3D_VEC3_T<T> &e1,
                                  const FIELD3D_VEC3_T<T> &e2,
                                  const FIELD3D_VEC3_T<T> &e3,
                                  const FIELD3D_VEC3_T<T> &origin)
{
  FIELD3D_MTX_T<T>  m;
  m[0][0] = e1.x;
  m[0][1] = e1.y;
  m[0][2] = e1.z;
  m[1][0] = e2.x;
  m[1][1] = e2.y;
  m[1][2] = e2.z;
  m[2][0] = e3.x;
  m[2][1] = e3.y;
  m[2][2] = e3.z;
  m[3][0] = origin.x;
  m[3][1] = origin.y;
  m[3][2] = origin.z;
  return m;
}

//----------------------------------------------------------------------------//

template <typename T>
FIELD3D_MTX_T<T> 
coordinateSystem(const FIELD3D_BOX_T<FIELD3D_VEC3_T<T> >  &wsBounds, 
                 const FIELD3D_VEC3_T<T>                  &wsVoxelSize,
                 Box3i                                    &extents)
{
  const FIELD3D_VEC3_T<T> voxelMin = 
    detail::floor<T>(wsBounds.min / wsVoxelSize) * wsVoxelSize;
  const FIELD3D_VEC3_T<T> voxelMax = 
    detail::ceil<T>(wsBounds.max / wsVoxelSize) * wsVoxelSize;

  // Resolution
  extents.min = V3i(detail::floor<T>(voxelMin / wsVoxelSize) + V3f(0.5));
  extents.max = V3i(detail::floor<T>(voxelMax / wsVoxelSize) + V3f(0.5));

  // Bounding box
  const FIELD3D_BOX_T<FIELD3D_VEC3_T<T> > box(voxelMin, voxelMax);

  return coordinateSystem(box);
}

//----------------------------------------------------------------------------//

template <typename T>
FIELD3D_MTX_T<T> 
coordinateSystem
(const FIELD3D_BOX_T<FIELD3D_VEC3_T<T> >  &wsBounds, 
 const FIELD3D_VEC3_T<T>                  &wsVoxelSize)
{
  Box3i dummy;
  return coordinateSystem(wsBounds, wsVoxelSize, dummy);
}

//----------------------------------------------------------------------------//

template <typename T>
FIELD3D_MTX_T<T> 
coordinateSystem
(const FIELD3D_BOX_T<FIELD3D_VEC3_T<T> > &wsBounds)
{
  FIELD3D_VEC3_T<T> e1(wsBounds.max.x - wsBounds.min.x, 0, 0);
  FIELD3D_VEC3_T<T> e2(0, wsBounds.max.y - wsBounds.min.y, 0);
  FIELD3D_VEC3_T<T> e3(0, 0, wsBounds.max.z - wsBounds.min.z);
  FIELD3D_VEC3_T<T> origin(wsBounds.min);
  return coordinateSystem(e1, e2, e3, origin);
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
