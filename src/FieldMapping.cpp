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

/*! \file FieldMapping.cpp
  \brief Contains the FieldMapping base class and the NullFieldMapping and
  MatrixFieldMapping subclass implementations.
*/

//----------------------------------------------------------------------------//

#include <iostream>
#include <vector>

#include "Field.h"
#include "FieldMapping.h"
#include "Types.h"

#ifdef WIN32
#define isnan(__x__) _isnan(__x__) 
#endif

//----------------------------------------------------------------------------//

using namespace boost;
using namespace std;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Field3D namespaces
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
// Local namespace
//----------------------------------------------------------------------------//

namespace {
  
  // Strings ---

  const string k_mappingName("FieldMapping");
  const string k_nullMappingName("NullFieldMapping");
  const string k_matrixMappingName("MatrixFieldMapping");
  const string k_frustumMappingName("FrustumFieldMapping");

  // Functions ---

  template <class Matrix_T>
  bool checkMatricesIdentical(const Matrix_T &m1, const Matrix_T &m2,
                              double tolerance)
  {
    if (m1.equalWithRelError(m2, tolerance)) {
      return true;
    }
    V3d s1, r1, t1, sh1, s2, r2, t2, sh2;
    if (!FIELD3D_EXTRACT_SHRT(m1, s1, sh1, r1, t1, false)) {
      return false;
    }
    if (!FIELD3D_EXTRACT_SHRT(m2, s2, sh2, r2, t2, false)) {
      return false;
    }
    if (!s1.equalWithRelError(s2, tolerance) ||
        !r1.equalWithAbsError(r2, tolerance) ||
        !t1.equalWithRelError(t2, tolerance)) {
      return false;
    }
    return true;
  }

}

//----------------------------------------------------------------------------//
// FieldMapping
//----------------------------------------------------------------------------//

FieldMapping::FieldMapping()
  : RefBase(), 
    m_origin(V3i(0)),
    m_res(V3i(1))
{ 
  /* Empty */ 
}

//----------------------------------------------------------------------------//

FieldMapping::FieldMapping(const Box3i &extents)
  : RefBase()
{ 
  setExtents(extents);
}

//----------------------------------------------------------------------------//

FieldMapping::~FieldMapping()
{ 
  /* Empty */ 
}

//----------------------------------------------------------------------------//


std::string FieldMapping::className() const
{
  return  std::string(classType());
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

void FieldMapping::voxelToLocal(const V3d &vsP, V3d &lsP) const
{ 
  lsP.x = FIELD3D_LERPFACTOR(vsP.x, m_origin.x, m_origin.x + m_res.x);
  lsP.y = FIELD3D_LERPFACTOR(vsP.y, m_origin.y, m_origin.y + m_res.y);
  lsP.z = FIELD3D_LERPFACTOR(vsP.z, m_origin.z, m_origin.z + m_res.z);
}

//----------------------------------------------------------------------------//
// NullFieldMapping
//----------------------------------------------------------------------------//

std::string NullFieldMapping::className() const
{
  return std::string(classType());
}

//----------------------------------------------------------------------------//

bool NullFieldMapping::isIdentical(FieldMapping::Ptr other, 
                                   double tolerance) const
{
  // For null mappings it's simple - if the other one is also a null mapping
  // then true, otherwise it's false.
  
  return other->className() == k_nullMappingName;
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
  makeIdentity();
}

//----------------------------------------------------------------------------//

MatrixFieldMapping::MatrixFieldMapping(const Box3i &extents)
  : FieldMapping(extents)
{ 
  makeIdentity();
}

//----------------------------------------------------------------------------//

void MatrixFieldMapping::setLocalToWorld(const M44d &lsToWs)
{
  if (m_lsToWsCurve.numSamples() > 0) {
    makeIdentity();
  }
  setLocalToWorld(0.0f, lsToWs);
}

//----------------------------------------------------------------------------//

void MatrixFieldMapping::setLocalToWorld(float t, const M44d &lsToWs)
{
  m_lsToWsCurve.addSample(t, lsToWs);
  updateTransform();
}

//----------------------------------------------------------------------------//

void MatrixFieldMapping::makeIdentity()
{
  m_lsToWsCurve.clear();
  updateTransform();
}

//----------------------------------------------------------------------------//

void MatrixFieldMapping::extentsChanged()
{ 
  updateTransform();
}

//----------------------------------------------------------------------------//

std::string MatrixFieldMapping::className() const
{
  return std::string(classType());
}

//----------------------------------------------------------------------------//

bool MatrixFieldMapping::isIdentical(FieldMapping::Ptr other, 
                                     double tolerance) const
{
  typedef MatrixFieldMapping::MatrixCurve::SampleVec SampleVec;

  if (other->className() != k_matrixMappingName) {
    return false;
  } else {

    MatrixFieldMapping::Ptr mm = 
      FIELD_DYNAMIC_CAST<MatrixFieldMapping>(other);

    if (mm) {

      const SampleVec lsToWs1 = m_lsToWsCurve.samples();
      const SampleVec lsToWs2 = mm->m_lsToWsCurve.samples();
      const SampleVec vsToWs1 = m_vsToWsCurve.samples();
      const SampleVec vsToWs2 = mm->m_vsToWsCurve.samples();

      size_t numSamples = lsToWs1.size();
      
      // First check if time sample counts differ
      // lsToWs and vsToWs are guaranteed to have same sample count.
      if (lsToWs1.size() != lsToWs2.size()) {
        return false;
      }
      
      // Then check if all time samples match, then check localToWorld 
      // and voxelToWorld matrices
      for (size_t i = 0; i < numSamples; ++i) {
        if (lsToWs1[i].first != lsToWs2[i].first) {
          return false;
        }
        if (!checkMatricesIdentical(lsToWs1[i].second, lsToWs2[i].second,
                                    tolerance)) {
          return false;
        }
        if (!checkMatricesIdentical(vsToWs1[i].second, vsToWs2[i].second,
                                    tolerance)) {
          return false;        
        }
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
  typedef MatrixCurve::SampleVec::const_iterator SampleIter;

  // Build the voxel to world space transforms ---
  M44d lsToVs;
  getLocalToVoxelMatrix(lsToVs);
  M44d vsToLs = lsToVs.inverse();
  // Loop over all samples in lsToWs, append vsToLs and create new curve
  // Also handle the special case where lsToWs has no samples. In that
  // case m_vsToWsCurve still has to have one sample.
  const MatrixCurve::SampleVec &lsToWs = m_lsToWsCurve.samples();
  m_vsToWsCurve.clear();
  for (SampleIter i = lsToWs.begin(), end = lsToWs.end(); i != end; i++) {
    m_vsToWsCurve.addSample(i->first, vsToLs * i->second);
  }

  // See if the curve has more than just a single sample
  m_isTimeVarying = m_lsToWsCurve.numSamples() > 1;

  // Sample the time-varying transforms at time=0.0
  m_lsToWs = m_lsToWsCurve.linear(0.0);
  m_wsToLs = m_lsToWs.inverse();
  m_vsToWs = vsToLs * m_lsToWs;
  m_wsToVs = m_vsToWs.inverse();

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
// FrustumFieldMapping
//----------------------------------------------------------------------------//

FrustumFieldMapping::FrustumFieldMapping()
  : FieldMapping(),
    m_zDistribution(PerspectiveDistribution),
    m_defaultState(true)
{ 
  reset();
}

//----------------------------------------------------------------------------//

FrustumFieldMapping::FrustumFieldMapping(const Box3i &extents)
  : FieldMapping(extents)
{ 
  reset();
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::setTransforms(const M44d &ssToWs, const M44d &csToWs)
{
  setTransforms(0.0, ssToWs, csToWs);
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::setTransforms(float t, 
                                        const M44d &ssToWs, const M44d &csToWs)
{
  if (m_defaultState) {
    clearCurves();
    m_defaultState = false;
  }

  // Construct local-to-world transform from ssToWs
  M44d lsToSs, scale, translation;
  scale.setScale(V3d(2.0, 2.0, 1.0));
  translation.setTranslation(V3d(-1.0, -1.0, 0.0));
  lsToSs = scale * translation;
  M44d lpsToWs = lsToSs * ssToWs;

  // Add samples to Curves
  m_ssToWsCurve.addSample(t, ssToWs);
  m_lpsToWsCurve.addSample(t, lpsToWs);
  m_csToWsCurve.addSample(t, csToWs);

  // Compute near and far planes ---

  // Because the frustum may be skewed we can't just measure distance from
  // the apex of the frustum to the world-space center point of the frustum.
  // Instead, we transform into camera space and measure z depth there.

  V3d lsNearP(0.5, 0.5, 0.0), lsFarP(0.5, 0.5, 1.0);
  V3d wsNearP, wsFarP, csNearP, csFarP;

  lpsToWs.multVecMatrix(lsNearP, wsNearP);
  lpsToWs.multVecMatrix(lsFarP, wsFarP);

  M44d wsToCs = csToWs.inverse();
  wsToCs.multVecMatrix(wsNearP, csNearP);
  wsToCs.multVecMatrix(wsFarP, csFarP);

  double near = -csNearP.z;
  double far = -csFarP.z;

  // Catch NaN here
  if (isnan(near) || isnan(far)) {
    throw BadPerspectiveMatrix("FrustumFieldMapping::setTransforms "
                               "received bad screen-to-world matrix");
  }

  m_nearCurve.addSample(t, near);
  m_farCurve.addSample(t, far);

  computeVoxelSize();
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::reset()
{
  // Default camera to world ---

  M44d csToWs;
  csToWs.makeIdentity();

  // Default screen to world ---

  double near = 1;
  double far = 2;
  double fovRadians = 45.0 * M_PI / 180.0;
  double invTan = 1.0 / std::tan(fovRadians / 2.0);
  double imageAspectRatio = 1.0;

  M44d perspective(1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, (far) / (far - near),          1,
                   0, 0, (- far * near) / (far - near), 0);

  M44d fov;
  fov.setScale(V3d(invTan / imageAspectRatio, invTan, 1.0));
  
  M44d flipZ;
  flipZ.setScale(V3d(1.0, 1.0, -1.0));
  
  M44d csToSs = flipZ * perspective * fov;

  M44d standardSsToWs = csToSs.inverse() * csToWs;

  // Set default state ---

  clearCurves();
  setTransforms(standardSsToWs, csToWs);

  m_defaultState = true;

  computeVoxelSize();
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::extentsChanged()
{ 
  computeVoxelSize();
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::worldToVoxel(const V3d &wsP, V3d &vsP) const
{
  worldToVoxel(wsP, vsP, 0.0);
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::worldToVoxel(const V3d &wsP, V3d &vsP, float time) const
{
  V3d lsP;
  worldToLocal(wsP, lsP, time);
  localToVoxel(lsP, vsP);
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::voxelToWorld(const V3d &vsP, V3d &wsP) const
{
  voxelToWorld(vsP, wsP, 0.0);
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::voxelToWorld(const V3d &vsP, V3d &wsP, float time) const
{
  V3d lsP;
  voxelToLocal(vsP, lsP);
  localToWorld(lsP, wsP, time);
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::worldToLocal(const V3d &wsP, V3d &lsP) const
{
  worldToLocal(wsP, lsP, 0.0);
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::worldToLocal(const V3d &wsP, V3d &lsP, float time) const
{
  switch (m_zDistribution) {
  case UniformDistribution:
    {
      // First transform to local perspective space
      V3d lpsP;
      m_lpsToWsCurve.linear(time).inverse().multVecMatrix(wsP, lpsP);
      // Also transform to camera space
      V3d csP;
      m_csToWsCurve.linear(time).inverse().multVecMatrix(wsP, csP);
      // Interpolate near and far plane at current time
      double near = m_nearCurve.linear(time);
      double far = m_farCurve.linear(time);
      // Use perspective-space X/Y and normalized depth for Z.
      lsP = V3d(lpsP.x, lpsP.y, FIELD3D_LERPFACTOR(-csP.z, near, far));
      break;
    }
  case PerspectiveDistribution:
  default:
    {
      m_lpsToWsCurve.linear(time).inverse().multVecMatrix(wsP, lsP);
      break;
    }
  }
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::localToWorld(const V3d &lsP, V3d &wsP) const
{
  localToWorld(lsP, wsP, 0.0);
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::localToWorld(const V3d &lsP, V3d &wsP, float time) const
{
  switch (m_zDistribution) {
  case UniformDistribution:
    {
      // Interpolate near and far plane at current time
      double near = m_nearCurve.linear(time);
      double far = m_farCurve.linear(time);
      // In this case, local space is -not- equal to local perspective space
      // Determine distance from camera
      double wsDepthFromCam = FIELD3D_LERP(near, far, lsP.z);
      // Transform point right in front of camera, X units away into world space
      V3d lpsCenterP, wsCenterP, csCenterP(0.0, 0.0, -wsDepthFromCam);
      m_csToWsCurve.linear(time).multVecMatrix(csCenterP, wsCenterP);
      // Transform center point into screen space so we know what depth 
      // (in screen space) the voxel would live at -if- it were in local
      // perspective space.
      m_lpsToWsCurve.linear(time).inverse().multVecMatrix(wsCenterP, lpsCenterP);
      // Now we create a local perspective coordinate that can be transformed
      // using m_lpsToWsCurve
      V3d lpsP(lsP.x, lsP.y, lpsCenterP.z);
      // Now we can use m_lpsToWsCurve to transform the actual voxel location
      m_lpsToWsCurve.linear(time).multVecMatrix(lpsP, wsP);
      break;
    }
  case PerspectiveDistribution:
  default:
    {
      // In this case, local space and local perspective space are the same.
      m_lpsToWsCurve.linear(time).multVecMatrix(lsP, wsP);
      break;
    }
  }
}

//----------------------------------------------------------------------------//

std::string FrustumFieldMapping::className() const
{
  return std::string(classType());
}

//----------------------------------------------------------------------------//

bool FrustumFieldMapping::isIdentical(FieldMapping::Ptr other, 
                                     double tolerance) const
{
  typedef MatrixFieldMapping::MatrixCurve::SampleVec SampleVec;

  if (other->className() != k_frustumMappingName) {
    return false;
  } else {

    FrustumFieldMapping::Ptr fm = 
      FIELD_DYNAMIC_CAST<FrustumFieldMapping>(other);

    if (fm) {

      const SampleVec lpsToWs1 = m_lpsToWsCurve.samples();
      const SampleVec lpsToWs2 = fm->m_lpsToWsCurve.samples();
      const SampleVec csToWs1 = m_csToWsCurve.samples();
      const SampleVec csToWs2 = fm->m_csToWsCurve.samples();

      size_t numSamples = lpsToWs1.size();
      
      // Check that slice distributions match
      if (m_zDistribution != fm->m_zDistribution) {
        return false;
      }

      // First check if time sample counts differ
      // lpsToWs and csToWs are guaranteed to have same sample count.
      if (lpsToWs1.size() != lpsToWs2.size()) {
        return false;
      }
      
      // Then check if all time samples match, then check localToWorld 
      // and voxelToWorld matrices
      for (size_t i = 0; i < numSamples; ++i) {
        if (lpsToWs1[i].first != lpsToWs2[i].first) {
          return false;
        }
        if (!checkMatricesIdentical(lpsToWs1[i].second, lpsToWs2[i].second,
                                    tolerance)) {
          return false;
        }
        if (!checkMatricesIdentical(csToWs1[i].second, csToWs2[i].second,
                                    tolerance)) {
          return false;        
        }
      }
      
      return true;

    } else {
      return false;
    }
  }
  return false;
}

//----------------------------------------------------------------------------//

V3d FrustumFieldMapping::wsVoxelSize(int /*i*/, int /*j*/, int k) const
{
  k = std::min(std::max(k, static_cast<int>(m_origin.z)), 
               static_cast<int>(m_origin.z + m_res.z - 1));
  return m_wsVoxelSize[k - static_cast<int>(m_origin.z)];
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::computeVoxelSize() 
{
  // Precompute the voxel size ---

  m_wsVoxelSize.resize(static_cast<int>(m_res.z),V3d(0.0));

  int i = m_origin.x + m_res.x / 2;
  int j = m_origin.y + m_res.y / 2;

  // Do all z slices except last
  int zMin = static_cast<int>(m_origin.z);
  int zMax = static_cast<int>(m_origin.z + m_res.z - 1);

  for (int k = zMin, idx = 0; k < zMax; ++k, ++idx) {
    V3d wsP, wsPx, wsPy, wsPz;
    V3d vsP = discToCont(V3i(i, j, k));
    V3d vsPx = discToCont(V3i(i + 1, j, k));
    V3d vsPy = discToCont(V3i(i, j + 1, k));
    V3d vsPz = discToCont(V3i(i, j, k + 1));
    voxelToWorld(vsP, wsP);
    voxelToWorld(vsPx, wsPx);
    voxelToWorld(vsPy, wsPy);
    voxelToWorld(vsPz, wsPz);
    m_wsVoxelSize[idx] = V3d((wsPx - wsP).length(), 
                             (wsPy - wsP).length(),
                             (wsPz - wsP).length());
  }

  // Duplicate last value since there are no further slices to differentiate
  // against
  if (m_res.z >= 2) {
    m_wsVoxelSize[m_res.z - 1] = m_wsVoxelSize[m_res.z - 2];
  }

}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::getLocalToVoxelMatrix(M44d &result)
{
  // Local to voxel is a scale by the resolution of the field, offset
  // to the origin of the extents
  M44d scaling, translation;
  scaling.setScale(m_res);
  translation.setTranslation(m_origin);
  result = scaling * translation;
}

//----------------------------------------------------------------------------//

void FrustumFieldMapping::clearCurves()
{
  m_lpsToWsCurve.clear();
  m_csToWsCurve.clear();
  m_nearCurve.clear();
  m_farCurve.clear();
}

//----------------------------------------------------------------------------//

FieldMapping::Ptr FrustumFieldMapping::clone() const
{
  return Ptr(new FrustumFieldMapping(*this));
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
