//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks
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

/*! \file FieldMapping.cpp
  \brief Contains the FieldMapping base class and the NullFieldMapping and
  MatrixFieldMapping subclass implementations.
*/

//----------------------------------------------------------------------------//

#include <iostream>
#include <vector>

#include "FieldMapping.h"
#include "Hdf5Util.h"
#include "Types.h"

//----------------------------------------------------------------------------//

using namespace boost;
using namespace std;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Field3D namespaces
//----------------------------------------------------------------------------//

using namespace Exc;
using namespace Hdf5Util;

//----------------------------------------------------------------------------//
// Local namespace
//----------------------------------------------------------------------------//

namespace {
  const string k_nullMappingName("NullFieldMapping");
  const string k_matrixMappingName("MatrixFieldMapping");
  const string k_frustumMappingName("FrustumFieldMapping");
}

//----------------------------------------------------------------------------//
// FieldMapping
//----------------------------------------------------------------------------//

FieldMapping::FieldMapping()
  : m_origin(V3i(0)),
    m_res(V3i(1)),
    m_counter(0)
{ 
  /* Empty */ 
}

//----------------------------------------------------------------------------//

FieldMapping::FieldMapping(const Box3i &extents)
{ 
  m_counter = 0;
  setExtents(extents);
}
//----------------------------------------------------------------------------//

FieldMapping::FieldMapping(const FieldMapping& src)
{
  *this = src;
  m_counter = 0;
}

//----------------------------------------------------------------------------//
    
FieldMapping & FieldMapping::operator = (const FieldMapping &src)
{
  m_origin = src.m_origin;
  m_res = src.m_res;
  return *this;
}

//----------------------------------------------------------------------------//

FieldMapping::~FieldMapping()
{ 
  /* Empty */ 
}

//----------------------------------------------------------------------------//

void FieldMapping::setExtents(const Box3i &extents)
{ 
  m_origin = extents.min;
  m_res = extents.max - extents.min + V3i(1);
  extentsChanged();
}

//----------------------------------------------------------------------------//

void FieldMapping::localToVoxel(const V3d &lsP, V3d &vsP) const
{ 
  vsP = m_origin + lsP * m_res;
}

//----------------------------------------------------------------------------//

void FieldMapping::localToVoxel(const V3d &lsP, V3d &vsP,
                                float /*time*/) const
{ 
  vsP = m_origin + lsP * m_res;
}

//----------------------------------------------------------------------------//

void FieldMapping::localToVoxel(std::vector<V3d>::const_iterator lsP, 
                                std::vector<V3d>::const_iterator end, 
                                std::vector<V3d>::iterator vsP) const
{ 
  for (; lsP != end; ++lsP, ++vsP) {
    *vsP = m_origin + *lsP * m_res;
  }
}
//----------------------------------------------------------------------------//

void FieldMapping::voxelToLocal(const V3d &vsP, V3d &lsP) const
{ 
  lsP.x = FIELD3D_LERPFACTOR(vsP.x, m_origin.x, m_origin.x + m_res.x);
  lsP.y = FIELD3D_LERPFACTOR(vsP.y, m_origin.y, m_origin.y + m_res.y);
  lsP.z = FIELD3D_LERPFACTOR(vsP.z, m_origin.z, m_origin.z + m_res.z);
}

//----------------------------------------------------------------------------//
// NullFieldMapping
//----------------------------------------------------------------------------//

std::string NullFieldMapping::typeName() const
{
  return k_nullMappingName;
}

//----------------------------------------------------------------------------//

bool NullFieldMapping::isIdentical(FieldMapping::Ptr other, 
                                   double tolerance) const
{
  // For null mappings it's simple - if the other one is also a null mapping
  // then true, otherwise it's false.
  
  return other->typeName() == k_nullMappingName;
}

//----------------------------------------------------------------------------//

FieldMapping::Ptr NullFieldMapping::clone() const
{
  return Ptr(new NullFieldMapping(*this));
}

//----------------------------------------------------------------------------//
// MatrixFieldMapping
//----------------------------------------------------------------------------//

MatrixFieldMapping::MatrixFieldMapping()
  : FieldMapping()
{ 
  makeIndentity();
}

//----------------------------------------------------------------------------//

MatrixFieldMapping::MatrixFieldMapping(const Box3i &extents)
  : FieldMapping(extents)
{ 
  makeIndentity();
}

//----------------------------------------------------------------------------//

void MatrixFieldMapping::setLocalToWorld(const M44d &lsToWs)
{
  m_lsToWs = lsToWs;
  updateTransform();
}

//----------------------------------------------------------------------------//

void MatrixFieldMapping::makeIndentity()
{
  m_lsToWs.makeIdentity();
  updateTransform();
}

//----------------------------------------------------------------------------//

void MatrixFieldMapping::extentsChanged()
{ 
  updateTransform();
}

//----------------------------------------------------------------------------//

std::string MatrixFieldMapping::typeName() const
{
  return k_matrixMappingName;
}

//----------------------------------------------------------------------------//

bool MatrixFieldMapping::isIdentical(FieldMapping::Ptr other, 
                                     double tolerance) const
{
  if (other->typeName() != k_matrixMappingName) {
    return false;
  } else {
    MatrixFieldMapping::Ptr mm = 
      dynamic_pointer_cast<MatrixFieldMapping>(other);
    if (mm) {
      // first preserve the same test as before:
      if (mm->m_lsToWs.equalWithRelError(m_lsToWs, tolerance) && 
          mm->m_wsToVs.equalWithRelError(m_wsToVs, tolerance)) {
        return true;
      }

      // In case of precision problems, do a more robust test by
      // decomposing the matrices and comparing the components

      V3d s1, r1, t1, sh1;
      V3d s2, r2, t2, sh2;

      // Compare local-to-world matrices
      if (!FIELD3D_EXTRACT_SHRT(m_lsToWs, s1, sh1, r1, t1, false)) {
        return false;
      }
      if (!FIELD3D_EXTRACT_SHRT(mm->m_lsToWs, s2, sh2, r2, t2, false)) {
        return false;
      }
      if (!s1.equalWithRelError(s2, tolerance) ||
          !r1.equalWithAbsError(r2, tolerance) ||
          !t1.equalWithRelError(t2, tolerance)) {
        return false;
      }

      // Compare world-to-voxel matrices
      if (!FIELD3D_EXTRACT_SHRT(m_wsToVs, s1, sh1, r1, t1, false)) {
        return false;
      }
      if (!FIELD3D_EXTRACT_SHRT(mm->m_wsToVs, s2, sh2, r2, t2, false)) {
        return false;
      }
      if (!s1.equalWithRelError(s2, tolerance) ||
          !r1.equalWithAbsError(r2, tolerance) ||
          !t1.equalWithRelError(t2, tolerance)) {
        return false;
      }
      return true;
    } else {
      return false;
    }
  }
  return false;
}

//----------------------------------------------------------------------------//

void MatrixFieldMapping::updateTransform() 
{
  m_wsToLs = m_lsToWs.inverse();
  M44d lsToVs;
  getLocalToVoxelMatrix(lsToVs);
  m_wsToVs = m_wsToLs * lsToVs;
  m_vsToWs = m_wsToVs.inverse();

  // Precompute the voxel size
  V3d voxelOrigin, nextVoxel;
  m_vsToWs.multVecMatrix(V3d(0, 0, 0), voxelOrigin);
  m_vsToWs.multVecMatrix(V3d(1, 0, 0), nextVoxel);
  m_wsVoxelSize.x = (nextVoxel - voxelOrigin).length();
  m_vsToWs.multVecMatrix(V3d(0, 1, 0), nextVoxel);
  m_wsVoxelSize.y = (nextVoxel - voxelOrigin).length();
  m_vsToWs.multVecMatrix(V3d(0, 0, 1), nextVoxel);
  m_wsVoxelSize.z = (nextVoxel - voxelOrigin).length();
}

//----------------------------------------------------------------------------//

void MatrixFieldMapping::getLocalToVoxelMatrix(M44d &result)
{
  // Local to voxel is a scale by the resolution of the field, offset
  // to the origin of the extents
  M44d scaling, translation;
  scaling.setScale(m_res);
  translation.setTranslation(m_origin);
  result = scaling * translation;
}

//----------------------------------------------------------------------------//

FieldMapping::Ptr MatrixFieldMapping::clone() const
{
  return Ptr(new MatrixFieldMapping(*this));
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
