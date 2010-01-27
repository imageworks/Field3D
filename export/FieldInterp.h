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

/*! \file FieldInterp.h
  \ingroup field
  \brief Contains the FieldInterp base class and some standard interpolation
  classes.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_FieldInterp_H_
#define _INCLUDED_Field3D_FieldInterp_H_

#include <boost/shared_ptr.hpp>

#include "Field.h"
#include "DenseField.h"
#include "MACField.h"
#include "ProceduralField.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// FieldInterp
//----------------------------------------------------------------------------//

/*! \class FieldInterp
  \ingroup field
  \brief Base class for interpolators. 
  \note An interpolator always interpolates in voxel space.
*/

template <class Data_T>
class FieldInterp
{
 public:
  typedef boost::shared_ptr<FieldInterp> Ptr;
  virtual ~FieldInterp() 
  { }
  virtual Data_T sample(const Field<Data_T> &data, const V3d &vsP) const = 0;
};

//----------------------------------------------------------------------------//
// LinearFieldInterp
//----------------------------------------------------------------------------//

/* \class LinearFieldInterp
   \ingroup field
   \brief Basic linear interpolator using voxel access through Field base class
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class LinearFieldInterp : public FieldInterp<Data_T>
{
 public:
  typedef boost::shared_ptr<LinearFieldInterp> Ptr;
  virtual Data_T sample(const Field<Data_T> &data, const V3d &vsP) const;
};

//----------------------------------------------------------------------------//
// CubicFieldInterp
//----------------------------------------------------------------------------//

/* \class CubicFieldInterp
   \ingroup field
   \brief Basic cubic interpolator using voxel access through Field base class
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class CubicFieldInterp : public FieldInterp<Data_T>
{
 public:
  typedef boost::shared_ptr<CubicFieldInterp> Ptr;
  virtual Data_T sample(const Field<Data_T> &data, const V3d &vsP) const;
};

//----------------------------------------------------------------------------//
// LinearGenericFieldInterp
//----------------------------------------------------------------------------//

/* \class LinearGenericFieldInterp
   \ingroup field
   \brief Linear interpolator optimized for fields with a fastValue function 
*/

//----------------------------------------------------------------------------//

template <class Field_T>
class LinearGenericFieldInterp
{
 public:
  typedef boost::shared_ptr<LinearGenericFieldInterp> Ptr;
  typename Field_T::value_type sample(const Field_T &data, const V3d &vsP) const;
};

//----------------------------------------------------------------------------//
// LinearMACFieldInterp
//----------------------------------------------------------------------------//

/* \class LinearMACFieldInterp
   \ingroup field
   \brief Linear interpolator optimized for the MAC fields
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class LinearMACFieldInterp
{
 public:

  typedef boost::shared_ptr<LinearMACFieldInterp> Ptr;

  Data_T sample(const MACField<Data_T> &data, const V3d &vsP) const;
};

//----------------------------------------------------------------------------//
// CubicGenericFieldInterp
//----------------------------------------------------------------------------//

/* \class CubicGenericFieldInterp
   \ingroup field
   \brief Cubic interpolator optimized for fields with a fastValue function 
*/

//----------------------------------------------------------------------------//

template <class Field_T>
class CubicGenericFieldInterp
{
public:
  typedef boost::shared_ptr<CubicGenericFieldInterp> Ptr;
  typename Field_T::value_type sample(const Field_T &data, const V3d &vsP) const;
};

//----------------------------------------------------------------------------//
// CubicMACFieldInterp
//----------------------------------------------------------------------------//

/* \class CubicMACFieldInterp
   \ingroup field
   \brief Linear interpolator optimized for MAC fields
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class CubicMACFieldInterp
{
public:
  typedef boost::shared_ptr<CubicMACFieldInterp> Ptr;
  Data_T sample(const MACField<Data_T> &data, const V3d &vsP) const;
};

//----------------------------------------------------------------------------//
// ProceduralFieldLookup
//----------------------------------------------------------------------------//

/* \class ProceduralFieldLookup
   \ingroup field
   \brief "Interpolator" for procedural fields - point samples instead of
   interpolating.
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class ProceduralFieldLookup
{
public:
  typedef boost::shared_ptr<ProceduralFieldLookup> Ptr;
  Data_T sample(const ProceduralField<Data_T> &data,
                const V3d &vsP) const;
};

//----------------------------------------------------------------------------//
// Interpolation functions
//----------------------------------------------------------------------------//

//! Helper function for interpolating in world space.
template <class Data_T>
Data_T wsSample(const typename Field<Data_T>::Ptr f, 
                const FieldInterp<Data_T> &interp, 
                const V3d &wsP)
{
  V3d vsP;
  f->mapping()->worldToVoxel(wsP, vsP);
  return interp.sample(*f, vsP);
}

//----------------------------------------------------------------------------//
// Interpolation helper functions
//----------------------------------------------------------------------------//

//! Checks whether the point is within the given field
bool isPointInField(const FieldRes::Ptr f, const V3d &wsP);

//----------------------------------------------------------------------------//

//! Checks whether the floating - point voxel coordinate is within the
//! given (floating point) data window
bool isLegalVoxelCoord(const V3d &vsP, const Box3d &vsDataWindow);

//----------------------------------------------------------------------------//
// Math functions
//----------------------------------------------------------------------------//

//! Scalar times Vec3 multiplication. Makes the interpolation calls
//! cleaner.
template <class S, class T>
FIELD3D_VEC3_T<T> operator * (S s, const FIELD3D_VEC3_T<T> vec);

//----------------------------------------------------------------------------//
// Interpolants
//----------------------------------------------------------------------------//

//! Monotonic cubic interpolation
//! References:
//! http://en.wikipedia.org/wiki/Monotone_cubic_interpolation
//! http://en.wikipedia.org/wiki/Cubic_Hermite_spline
template <class Data_T>
Data_T monotonicCubicInterpolant(const Data_T &f1, const Data_T &f2, 
                                 const Data_T &f3, const Data_T &f4, 
                                 double t);

//----------------------------------------------------------------------------//

//! Monotonic cubic interpolation on 3 - vectors
//! References:
//! http://en.wikipedia.org/wiki/Monotone_cubic_interpolation
//! http://en.wikipedia.org/wiki/Cubic_Hermite_spline
template <class Data_T>
Data_T monotonicCubicInterpolantVec(const Data_T &f1, const Data_T &f2, 
                                    const Data_T &f3, const Data_T &f4, 
                                    double t);

//----------------------------------------------------------------------------//
// Implementations
//----------------------------------------------------------------------------//

template <class Data_T>
Data_T LinearFieldInterp<Data_T>::sample(const Field<Data_T> &data, 
                                         const V3d &vsP) const
{
  // Voxel centers are at .5 coordinates
  // NOTE: Don't use contToDisc for this, we're looking for sample
  // point locations, not coordinate shifts.
  FIELD3D_VEC3_T<double> p(vsP - FIELD3D_VEC3_T<double>(0.5));

  // Lower left corner
  V3i c1(static_cast<int>(floor(p.x)), 
         static_cast<int>(floor(p.y)), 
         static_cast<int>(floor(p.z)));
  // Upper right corner
  V3i c2(c1 + V3i(1));
  // C1 fractions
  FIELD3D_VEC3_T<double> f1(static_cast<FIELD3D_VEC3_T<double> >(c2) - p);
  // C2 fraction
  FIELD3D_VEC3_T<double> f2(static_cast<FIELD3D_VEC3_T<double> >(1.0) - f1);

  // Clamp the indexing coordinates
  if (true) {
    const Box3i &dataWindow = data.dataWindow();        
    c1.x = std::max(dataWindow.min.x, std::min(c1.x, dataWindow.max.x));
    c2.x = std::max(dataWindow.min.x, std::min(c2.x, dataWindow.max.x));
    c1.y = std::max(dataWindow.min.y, std::min(c1.y, dataWindow.max.y));
    c2.y = std::max(dataWindow.min.y, std::min(c2.y, dataWindow.max.y));
    c1.z = std::max(dataWindow.min.z, std::min(c1.z, dataWindow.max.z));
    c2.z = std::max(dataWindow.min.z, std::min(c2.z, dataWindow.max.z));
  }
    
  return static_cast<Data_T>
    (f1.x * (f1.y * (f1.z * data.value(c1.x, c1.y, c1.z) +
                     f2.z * data.value(c1.x, c1.y, c2.z)) +
             f2.y * (f1.z * data.value(c1.x, c2.y, c1.z) + 
                     f2.z * data.value(c1.x, c2.y, c2.z))) +
     f2.x * (f1.y * (f1.z * data.value(c2.x, c1.y, c1.z) + 
                     f2.z * data.value(c2.x, c1.y, c2.z)) +
             f2.y * (f1.z * data.value(c2.x, c2.y, c1.z) + 
                     f2.z * data.value(c2.x, c2.y, c2.z))));

}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T CubicFieldInterp<Data_T>::sample(const Field<Data_T> &data, 
                                        const V3d &vsP) const
{
  // Voxel centers are at .5 coordinates
  // NOTE: Don't use contToDisc for this, we're looking for sample
  // point locations, not coordinate shifts.
  FIELD3D_VEC3_T<double> p(vsP - FIELD3D_VEC3_T<double>(0.5));

  // Lower left corner
  V3i c(static_cast<int>(floor(p.x)), 
        static_cast<int>(floor(p.y)), 
        static_cast<int>(floor(p.z)));

  // Fractions
  FIELD3D_VEC3_T<double> t(p - static_cast<FIELD3D_VEC3_T<double> >(c));

  const Box3i &dataWindow = data.dataWindow();

  // Clamp the coordinates
  int im, jm, km;
  im = std::max(dataWindow.min.x, std::min(c.x, dataWindow.max.x));
  jm = std::max(dataWindow.min.y, std::min(c.y, dataWindow.max.y));
  km = std::max(dataWindow.min.z, std::min(c.z, dataWindow.max.z));
  int im_1, jm_1, km_1;
  im_1 = std::max(dataWindow.min.x, std::min(im - 1, dataWindow.max.x));
  jm_1 = std::max(dataWindow.min.y, std::min(jm - 1, dataWindow.max.y));
  km_1 = std::max(dataWindow.min.z, std::min(km - 1, dataWindow.max.z));
  int im1, jm1, km1;
  im1 = std::max(dataWindow.min.x, std::min(im + 1, dataWindow.max.x));
  jm1 = std::max(dataWindow.min.y, std::min(jm + 1, dataWindow.max.y));
  km1 = std::max(dataWindow.min.z, std::min(km + 1, dataWindow.max.z));
  int im2, jm2, km2;
  im2 = std::max(dataWindow.min.x, std::min(im + 2, dataWindow.max.x));
  jm2 = std::max(dataWindow.min.y, std::min(jm + 2, dataWindow.max.y));
  km2 = std::max(dataWindow.min.z, std::min(km + 2, dataWindow.max.z));

  // interpolate 16 lines in z:
  Data_T z11 = monotonicCubicInterpolant(data.value(im_1, jm_1, km_1), 
                                         data.value(im_1, jm_1, km), 
                                         data.value(im_1, jm_1, km1), 
                                         data.value(im_1, jm_1, km2), t.z);
  Data_T z12 = monotonicCubicInterpolant(data.value(im_1, jm, km_1), 
                                         data.value(im_1, jm, km), 
                                         data.value(im_1, jm, km1), 
                                         data.value(im_1, jm, km2), t.z);
  Data_T z13 = monotonicCubicInterpolant(data.value(im_1, jm1, km_1), 
                                         data.value(im_1, jm1, km), 
                                         data.value(im_1, jm1, km1), 
                                         data.value(im_1, jm1, km2), t.z);
  Data_T z14 = monotonicCubicInterpolant(data.value(im_1, jm2, km_1), 
                                         data.value(im_1, jm2, km), 
                                         data.value(im_1, jm2, km1), 
                                         data.value(im_1, jm2, km2), t.z);

  Data_T z21 = monotonicCubicInterpolant(data.value(im, jm_1, km_1), 
                                         data.value(im, jm_1, km), 
                                         data.value(im, jm_1, km1), 
                                         data.value(im, jm_1, km2), t.z);
  Data_T z22 = monotonicCubicInterpolant(data.value(im, jm, km_1), 
                                         data.value(im, jm, km), 
                                         data.value(im, jm, km1), 
                                         data.value(im, jm, km2), t.z);
  Data_T z23 = monotonicCubicInterpolant(data.value(im, jm1, km_1), 
                                         data.value(im, jm1, km), 
                                         data.value(im, jm1, km1), 
                                         data.value(im, jm1, km2), t.z);
  Data_T z24 = monotonicCubicInterpolant(data.value(im, jm2, km_1), 
                                         data.value(im, jm2, km), 
                                         data.value(im, jm2, km1), 
                                         data.value(im, jm2, km2), t.z);

  Data_T z31 = monotonicCubicInterpolant(data.value(im1, jm_1, km_1), 
                                         data.value(im1, jm_1, km), 
                                         data.value(im1, jm_1, km1), 
                                         data.value(im1, jm_1, km2), t.z);
  Data_T z32 = monotonicCubicInterpolant(data.value(im1, jm, km_1), 
                                         data.value(im1, jm, km), 
                                         data.value(im1, jm, km1), 
                                         data.value(im1, jm, km2), t.z);
  Data_T z33 = monotonicCubicInterpolant(data.value(im1, jm1, km_1), 
                                         data.value(im1, jm1, km), 
                                         data.value(im1, jm1, km1), 
                                         data.value(im1, jm1, km2), t.z);
  Data_T z34 = monotonicCubicInterpolant(data.value(im1, jm2, km_1), 
                                         data.value(im1, jm2, km), 
                                         data.value(im1, jm2, km1), 
                                         data.value(im1, jm2, km2), t.z);

  Data_T z41 = monotonicCubicInterpolant(data.value(im2, jm_1, km_1), 
                                         data.value(im2, jm_1, km), 
                                         data.value(im2, jm_1, km1), 
                                         data.value(im2, jm_1, km2), t.z);
  Data_T z42 = monotonicCubicInterpolant(data.value(im2, jm, km_1), 
                                         data.value(im2, jm, km), 
                                         data.value(im2, jm, km1), 
                                         data.value(im2, jm, km2), t.z);
  Data_T z43 = monotonicCubicInterpolant(data.value(im2, jm1, km_1), 
                                         data.value(im2, jm1, km), 
                                         data.value(im2, jm1, km1), 
                                         data.value(im2, jm1, km2), t.z);
  Data_T z44 = monotonicCubicInterpolant(data.value(im2, jm2, km_1), 
                                         data.value(im2, jm2, km), 
                                         data.value(im2, jm2, km1), 
                                         data.value(im2, jm2, km2), t.z);

  Data_T y1 = monotonicCubicInterpolant(z11, z12, z13, z14, t.y);
  Data_T y2 = monotonicCubicInterpolant(z21, z22, z23, z24, t.y);
  Data_T y3 = monotonicCubicInterpolant(z31, z32, z33, z34, t.y);
  Data_T y4 = monotonicCubicInterpolant(z41, z42, z43, z44, t.y);
                   
  Data_T z0 = monotonicCubicInterpolant(y1, y2, y3, y4, t.x);

  return z0;
}

//----------------------------------------------------------------------------//

template <class Field_T>
typename Field_T::value_type
LinearGenericFieldInterp<Field_T>::sample(const Field_T &data, 
                                          const V3d &vsP) const
{
  typedef typename Field_T::value_type Data_T;

  // Pixel centers are at .5 coordinates
  // NOTE: Don't use contToDisc for this, we're looking for sample
  // point locations, not coordinate shifts.
  FIELD3D_VEC3_T<double> p(vsP - FIELD3D_VEC3_T<double>(0.5));

  // Lower left corner
  V3i c1(static_cast<int>(floor(p.x)), 
         static_cast<int>(floor(p.y)), 
         static_cast<int>(floor(p.z)));
  // Upper right corner
  V3i c2(c1 + V3i(1));
  // C1 fractions
  FIELD3D_VEC3_T<double> f1(static_cast<FIELD3D_VEC3_T<double> >(c2) - p);
  // C2 fraction
  FIELD3D_VEC3_T<double> f2(static_cast<FIELD3D_VEC3_T<double> >(1.0) - f1);

  const Box3i &dataWindow = data.dataWindow();        

  // Clamp the coordinates
  c1.x = std::min(dataWindow.max.x, std::max(dataWindow.min.x, c1.x));
  c1.y = std::min(dataWindow.max.y, std::max(dataWindow.min.y, c1.y));
  c1.z = std::min(dataWindow.max.z, std::max(dataWindow.min.z, c1.z));
  c2.x = std::min(dataWindow.max.x, std::max(dataWindow.min.x, c2.x));
  c2.y = std::min(dataWindow.max.y, std::max(dataWindow.min.y, c2.y));
  c2.z = std::min(dataWindow.max.z, std::max(dataWindow.min.z, c2.z));

  return static_cast<Data_T>
    (f1.x * (f1.y * (f1.z * data.fastValue(c1.x, c1.y, c1.z) +
                     f2.z * data.fastValue(c1.x, c1.y, c2.z)) +
             f2.y * (f1.z * data.fastValue(c1.x, c2.y, c1.z) + 
                     f2.z * data.fastValue(c1.x, c2.y, c2.z))) +
     f2.x * (f1.y * (f1.z * data.fastValue(c2.x, c1.y, c1.z) + 
                     f2.z * data.fastValue(c2.x, c1.y, c2.z)) +
             f2.y * (f1.z * data.fastValue(c2.x, c2.y, c1.z) + 
                     f2.z * data.fastValue(c2.x, c2.y, c2.z))));
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T LinearMACFieldInterp<Data_T>::sample(const MACField<Data_T> &data, 
                                            const V3d &vsP) const
{
  // Pixel centers are at .5 coordinates
  // NOTE: Don't use contToDisc for this, we're looking for sample
  // point locations, not coordinate shifts.

  const Box3i &dataWindow = data.dataWindow();      

  Data_T ret;

  FIELD3D_VEC3_T<double> p(vsP.x , vsP.y - 0.5, vsP.z - 0.5);

  // X component ---

  // Lower left corner
  V3i c1(static_cast<int>(floor(p.x)), 
         static_cast<int>(floor(p.y)), 
         static_cast<int>(floor(p.z)));
    
  // Upper right corner
  V3i c2(c1 + V3i(1));

  // C1 fractions
  FIELD3D_VEC3_T<double> f1(static_cast<FIELD3D_VEC3_T<double> >(c2) - p);
  // C2 fraction
  FIELD3D_VEC3_T<double> f2(static_cast<FIELD3D_VEC3_T<double> >(1.0) - f1);

  // Clamp the coordinates
  c1.x = std::min(dataWindow.max.x + 1, std::max(dataWindow.min.x, c1.x));
  c1.y = std::min(dataWindow.max.y, std::max(dataWindow.min.y, c1.y));
  c1.z = std::min(dataWindow.max.z, std::max(dataWindow.min.z, c1.z));
  c2.x = std::min(dataWindow.max.x + 1, std::max(dataWindow.min.x, c2.x));
  c2.y = std::min(dataWindow.max.y, std::max(dataWindow.min.y, c2.y));
  c2.z = std::min(dataWindow.max.z, std::max(dataWindow.min.z, c2.z));

  ret.x = (f1.x * (f1.y * (f1.z * data.u(c1.x, c1.y, c1.z) +
                           f2.z * data.u(c1.x, c1.y, c2.z)) +
                   f2.y * (f1.z * data.u(c1.x, c2.y, c1.z) + 
                           f2.z * data.u(c1.x, c2.y, c2.z))) +
           f2.x * (f1.y * (f1.z * data.u(c2.x, c1.y, c1.z) + 
                           f2.z * data.u(c2.x, c1.y, c2.z)) +
                   f2.y * (f1.z * data.u(c2.x, c2.y, c1.z) + 
                           f2.z * data.u(c2.x, c2.y, c2.z))));

  // Y component ---

  p.setValue(vsP.x - 0.5, vsP.y , vsP.z - 0.5);

  // Lower left corner
  c1.x = static_cast<int>(floor(p.x ));
  c1.y = static_cast<int>(floor(p.y )); 
  c1.z = static_cast<int>(floor(p.z ));
    
  // Upper right corner
  c2.x = c1.x + 1;
  c2.y = c1.y + 1;
  c2.z = c1.z + 1;

  // C1 fractions
  f1.setValue(static_cast<FIELD3D_VEC3_T<double> >(c2) - p);
  // C2 fraction
  f2.setValue(static_cast<FIELD3D_VEC3_T<double> >(1.0) - f1);

  // Clamp the coordinates
  c1.x = std::min(dataWindow.max.x, std::max(dataWindow.min.x, c1.x));
  c1.y = std::min(dataWindow.max.y + 1, std::max(dataWindow.min.y, c1.y));
  c1.z = std::min(dataWindow.max.z, std::max(dataWindow.min.z, c1.z));
  c2.x = std::min(dataWindow.max.x, std::max(dataWindow.min.x, c2.x));
  c2.y = std::min(dataWindow.max.y + 1, std::max(dataWindow.min.y, c2.y));
  c2.z = std::min(dataWindow.max.z, std::max(dataWindow.min.z, c2.z));

  ret.y = (f1.x * (f1.y * (f1.z * data.v(c1.x, c1.y, c1.z) +
                           f2.z * data.v(c1.x, c1.y, c2.z)) +
                   f2.y * (f1.z * data.v(c1.x, c2.y, c1.z) + 
                           f2.z * data.v(c1.x, c2.y, c2.z))) +
           f2.x * (f1.y * (f1.z * data.v(c2.x, c1.y, c1.z) + 
                           f2.z * data.v(c2.x, c1.y, c2.z)) +
                   f2.y * (f1.z * data.v(c2.x, c2.y, c1.z) + 
                           f2.z * data.v(c2.x, c2.y, c2.z))));

  // Z component ---

  p.setValue(vsP.x - 0.5 , vsP.y - 0.5, vsP.z);

  // Lower left corner
  c1.x = static_cast<int>(floor(p.x ));
  c1.y = static_cast<int>(floor(p.y )); 
  c1.z = static_cast<int>(floor(p.z ));
    
  // Upper right corner
  c2.x = c1.x + 1;
  c2.y = c1.y + 1;
  c2.z = c1.z + 1;

  // C1 fractions
  f1.setValue(static_cast<FIELD3D_VEC3_T<double> >(c2) - p);
  // C2 fraction
  f2.setValue(static_cast<FIELD3D_VEC3_T<double> >(1.0) - f1);

  // Clamp the coordinates
  c1.x = std::min(dataWindow.max.x, std::max(dataWindow.min.x, c1.x));
  c1.y = std::min(dataWindow.max.y, std::max(dataWindow.min.y, c1.y));
  c1.z = std::min(dataWindow.max.z + 1, std::max(dataWindow.min.z, c1.z));
  c2.x = std::min(dataWindow.max.x, std::max(dataWindow.min.x, c2.x));
  c2.y = std::min(dataWindow.max.y, std::max(dataWindow.min.y, c2.y));
  c2.z = std::min(dataWindow.max.z + 1, std::max(dataWindow.min.z, c2.z));

  ret.z = (f1.x * (f1.y * (f1.z * data.w(c1.x, c1.y, c1.z) +
                           f2.z * data.w(c1.x, c1.y, c2.z)) +
                   f2.y * (f1.z * data.w(c1.x, c2.y, c1.z) + 
                           f2.z * data.w(c1.x, c2.y, c2.z))) +
           f2.x * (f1.y * (f1.z * data.w(c2.x, c1.y, c1.z) + 
                           f2.z * data.w(c2.x, c1.y, c2.z)) +
                   f2.y * (f1.z * data.w(c2.x, c2.y, c1.z) + 
                           f2.z * data.w(c2.x, c2.y, c2.z))));

  return ret;
}

//----------------------------------------------------------------------------//

template <class Field_T>
typename Field_T::value_type
CubicGenericFieldInterp<Field_T>::sample(const Field_T &data, 
                                         const V3d &vsP) const
{
  typedef typename Field_T::value_type Data_T;

  // Pixel centers are at .5 coordinates
  // NOTE: Don't use contToDisc for this, we're looking for sample
  // point locations, not coordinate shifts.
  FIELD3D_VEC3_T<double> p(vsP - FIELD3D_VEC3_T<double>(0.5));

  const Box3i &dataWindow = data.dataWindow();

  // Lower left corner
  V3i c(static_cast<int>(floor(p.x)), 
        static_cast<int>(floor(p.y)), 
        static_cast<int>(floor(p.z)));

  // Fractions
  FIELD3D_VEC3_T<double> t(p - static_cast<FIELD3D_VEC3_T<double> >(c));

  // Clamp the coordinates
  int im, jm, km;
  im = std::max(dataWindow.min.x, std::min(c.x, dataWindow.max.x));
  jm = std::max(dataWindow.min.y, std::min(c.y, dataWindow.max.y));
  km = std::max(dataWindow.min.z, std::min(c.z, dataWindow.max.z));
  int im_1, jm_1, km_1;
  im_1 = std::max(dataWindow.min.x, std::min(im - 1, dataWindow.max.x));
  jm_1 = std::max(dataWindow.min.y, std::min(jm - 1, dataWindow.max.y));
  km_1 = std::max(dataWindow.min.z, std::min(km - 1, dataWindow.max.z));
  int im1, jm1, km1;
  im1 = std::max(dataWindow.min.x, std::min(im + 1, dataWindow.max.x));
  jm1 = std::max(dataWindow.min.y, std::min(jm + 1, dataWindow.max.y));
  km1 = std::max(dataWindow.min.z, std::min(km + 1, dataWindow.max.z));
  int im2, jm2, km2;
  im2 = std::max(dataWindow.min.x, std::min(im + 2, dataWindow.max.x));
  jm2 = std::max(dataWindow.min.y, std::min(jm + 2, dataWindow.max.y));
  km2 = std::max(dataWindow.min.z, std::min(km + 2, dataWindow.max.z));

  Data_T z11 = monotonicCubicInterpolant(data.fastValue(im_1, jm_1, km_1), 
                                         data.fastValue(im_1, jm_1, km), 
                                         data.fastValue(im_1, jm_1, km1), 
                                         data.fastValue(im_1, jm_1, km2), t.z);
  Data_T z12 = monotonicCubicInterpolant(data.fastValue(im_1, jm, km_1), 
                                         data.fastValue(im_1, jm, km), 
                                         data.fastValue(im_1, jm, km1), 
                                         data.fastValue(im_1, jm, km2), t.z);
  Data_T z13 = monotonicCubicInterpolant(data.fastValue(im_1, jm1, km_1), 
                                         data.fastValue(im_1, jm1, km), 
                                         data.fastValue(im_1, jm1, km1), 
                                         data.fastValue(im_1, jm1, km2), t.z);
  Data_T z14 = monotonicCubicInterpolant(data.fastValue(im_1, jm2, km_1), 
                                         data.fastValue(im_1, jm2, km), 
                                         data.fastValue(im_1, jm2, km1), 
                                         data.fastValue(im_1, jm2, km2), t.z);

  Data_T z21 = monotonicCubicInterpolant(data.fastValue(im, jm_1, km_1), 
                                         data.fastValue(im, jm_1, km), 
                                         data.fastValue(im, jm_1, km1), 
                                         data.fastValue(im, jm_1, km2), t.z);
  Data_T z22 = monotonicCubicInterpolant(data.fastValue(im, jm, km_1), 
                                         data.fastValue(im, jm, km), 
                                         data.fastValue(im, jm, km1), 
                                         data.fastValue(im, jm, km2), t.z);
  Data_T z23 = monotonicCubicInterpolant(data.fastValue(im, jm1, km_1), 
                                         data.fastValue(im, jm1, km), 
                                         data.fastValue(im, jm1, km1), 
                                         data.fastValue(im, jm1, km2), t.z);
  Data_T z24 = monotonicCubicInterpolant(data.fastValue(im, jm2, km_1), 
                                         data.fastValue(im, jm2, km), 
                                         data.fastValue(im, jm2, km1), 
                                         data.fastValue(im, jm2, km2), t.z);

  Data_T z31 = monotonicCubicInterpolant(data.fastValue(im1, jm_1, km_1), 
                                         data.fastValue(im1, jm_1, km), 
                                         data.fastValue(im1, jm_1, km1), 
                                         data.fastValue(im1, jm_1, km2), t.z);
  Data_T z32 = monotonicCubicInterpolant(data.fastValue(im1, jm, km_1), 
                                         data.fastValue(im1, jm, km), 
                                         data.fastValue(im1, jm, km1), 
                                         data.fastValue(im1, jm, km2), t.z);
  Data_T z33 = monotonicCubicInterpolant(data.fastValue(im1, jm1, km_1), 
                                         data.fastValue(im1, jm1, km), 
                                         data.fastValue(im1, jm1, km1), 
                                         data.fastValue(im1, jm1, km2), t.z);
  Data_T z34 = monotonicCubicInterpolant(data.fastValue(im1, jm2, km_1), 
                                         data.fastValue(im1, jm2, km), 
                                         data.fastValue(im1, jm2, km1), 
                                         data.fastValue(im1, jm2, km2), t.z);

  Data_T z41 = monotonicCubicInterpolant(data.fastValue(im2, jm_1, km_1), 
                                         data.fastValue(im2, jm_1, km), 
                                         data.fastValue(im2, jm_1, km1), 
                                         data.fastValue(im2, jm_1, km2), t.z);
  Data_T z42 = monotonicCubicInterpolant(data.fastValue(im2, jm, km_1), 
                                         data.fastValue(im2, jm, km), 
                                         data.fastValue(im2, jm, km1), 
                                         data.fastValue(im2, jm, km2), t.z);
  Data_T z43 = monotonicCubicInterpolant(data.fastValue(im2, jm1, km_1), 
                                         data.fastValue(im2, jm1, km), 
                                         data.fastValue(im2, jm1, km1), 
                                         data.fastValue(im2, jm1, km2), t.z);
  Data_T z44 = monotonicCubicInterpolant(data.fastValue(im2, jm2, km_1), 
                                         data.fastValue(im2, jm2, km), 
                                         data.fastValue(im2, jm2, km1), 
                                         data.fastValue(im2, jm2, km2), t.z);

  Data_T y1 = monotonicCubicInterpolant(z11, z12, z13, z14, t.y);
  Data_T y2 = monotonicCubicInterpolant(z21, z22, z23, z24, t.y);
  Data_T y3 = monotonicCubicInterpolant(z31, z32, z33, z34, t.y);
  Data_T y4 = monotonicCubicInterpolant(z41, z42, z43, z44, t.y);
                   
  Data_T z0 = monotonicCubicInterpolant(y1, y2, y3, y4, t.x);

  return z0;
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T CubicMACFieldInterp<Data_T>::sample(const MACField<Data_T> &data, 
                                           const V3d &vsP) const
{
  typedef typename Data_T::BaseType T;

  const Box3i &dataWindow = data.dataWindow();      

  // Pixel centers are at .5 coordinates
  // NOTE: Don't use contToDisc for this, we're looking for sample
  // point locations, not coordinate shifts.

  Data_T ret;

  // X component ---

  FIELD3D_VEC3_T<double> p(vsP.x , vsP.y - 0.5, vsP.z - 0.5);

  // Lower left corner
  V3i c(static_cast<int>(floor(p.x)), 
        static_cast<int>(floor(p.y)), 
        static_cast<int>(floor(p.z)));
    
  FIELD3D_VEC3_T<double> t(p - static_cast<FIELD3D_VEC3_T<double> >(c));
    
  {                   
    // Clamp the coordinates
    int im, jm, km;
    im = std::max(dataWindow.min.x, std::min(c.x, dataWindow.max.x + 1));
    jm = std::max(dataWindow.min.y, std::min(c.y, dataWindow.max.y));
    km = std::max(dataWindow.min.z, std::min(c.z, dataWindow.max.z));
    int im_1, jm_1, km_1;
    im_1 = std::max(dataWindow.min.x, std::min(im - 1, dataWindow.max.x + 1));
    jm_1 = std::max(dataWindow.min.y, std::min(jm - 1, dataWindow.max.y));
    km_1 = std::max(dataWindow.min.z, std::min(km - 1, dataWindow.max.z));
    int im1, jm1, km1;
    im1 = std::max(dataWindow.min.x, std::min(im + 1, dataWindow.max.x + 1));
    jm1 = std::max(dataWindow.min.y, std::min(jm + 1, dataWindow.max.y));
    km1 = std::max(dataWindow.min.z, std::min(km + 1, dataWindow.max.z));
    int im2, jm2, km2;
    im2 = std::max(dataWindow.min.x, std::min(im + 2, dataWindow.max.x + 1));
    jm2 = std::max(dataWindow.min.y, std::min(jm + 2, dataWindow.max.y));
    km2 = std::max(dataWindow.min.z, std::min(km + 2, dataWindow.max.z));

    T z11 = monotonicCubicInterpolant(data.u(im_1, jm_1, km_1), 
                                      data.u(im_1, jm_1, km), 
                                      data.u(im_1, jm_1, km1), 
                                      data.u(im_1, jm_1, km2), t.z);
    T z12 = monotonicCubicInterpolant(data.u(im_1, jm, km_1), 
                                      data.u(im_1, jm, km), 
                                      data.u(im_1, jm, km1), 
                                      data.u(im_1, jm, km2), t.z);
    T z13 = monotonicCubicInterpolant(data.u(im_1, jm1, km_1), 
                                      data.u(im_1, jm1, km), 
                                      data.u(im_1, jm1, km1), 
                                      data.u(im_1, jm1, km2), t.z);
    T z14 = monotonicCubicInterpolant(data.u(im_1, jm2, km_1), 
                                      data.u(im_1, jm2, km), 
                                      data.u(im_1, jm2, km1), 
                                      data.u(im_1, jm2, km2), t.z);

    T z21 = monotonicCubicInterpolant(data.u(im, jm_1, km_1), 
                                      data.u(im, jm_1, km), 
                                      data.u(im, jm_1, km1), 
                                      data.u(im, jm_1, km2), t.z);
    T z22 = monotonicCubicInterpolant(data.u(im, jm, km_1), 
                                      data.u(im, jm, km), 
                                      data.u(im, jm, km1), 
                                      data.u(im, jm, km2), t.z);
    T z23 = monotonicCubicInterpolant(data.u(im, jm1, km_1), 
                                      data.u(im, jm1, km), 
                                      data.u(im, jm1, km1), 
                                      data.u(im, jm1, km2), t.z);
    T z24 = monotonicCubicInterpolant(data.u(im, jm2, km_1), 
                                      data.u(im, jm2, km), 
                                      data.u(im, jm2, km1), 
                                      data.u(im, jm2, km2), t.z);

    T z31 = monotonicCubicInterpolant(data.u(im1, jm_1, km_1), 
                                      data.u(im1, jm_1, km), 
                                      data.u(im1, jm_1, km1), 
                                      data.u(im1, jm_1, km2), t.z);
    T z32 = monotonicCubicInterpolant(data.u(im1, jm, km_1), 
                                      data.u(im1, jm, km), 
                                      data.u(im1, jm, km1), 
                                      data.u(im1, jm, km2), t.z);
    T z33 = monotonicCubicInterpolant(data.u(im1, jm1, km_1), 
                                      data.u(im1, jm1, km), 
                                      data.u(im1, jm1, km1), 
                                      data.u(im1, jm1, km2), t.z);
    T z34 = monotonicCubicInterpolant(data.u(im1, jm2, km_1), 
                                      data.u(im1, jm2, km), 
                                      data.u(im1, jm2, km1), 
                                      data.u(im1, jm2, km2), t.z);

    T z41 = monotonicCubicInterpolant(data.u(im2, jm_1, km_1), 
                                      data.u(im2, jm_1, km), 
                                      data.u(im2, jm_1, km1), 
                                      data.u(im2, jm_1, km2), t.z);
    T z42 = monotonicCubicInterpolant(data.u(im2, jm, km_1), 
                                      data.u(im2, jm, km), 
                                      data.u(im2, jm, km1), 
                                      data.u(im2, jm, km2), t.z);
    T z43 = monotonicCubicInterpolant(data.u(im2, jm1, km_1), 
                                      data.u(im2, jm1, km), 
                                      data.u(im2, jm1, km1), 
                                      data.u(im2, jm1, km2), t.z);
    T z44 = monotonicCubicInterpolant(data.u(im2, jm2, km_1), 
                                      data.u(im2, jm2, km), 
                                      data.u(im2, jm2, km1), 
                                      data.u(im2, jm2, km2), t.z);

    T y1 = monotonicCubicInterpolant(z11, z12, z13, z14, t.y);
    T y2 = monotonicCubicInterpolant(z21, z22, z23, z24, t.y);
    T y3 = monotonicCubicInterpolant(z31, z32, z33, z34, t.y);
    T y4 = monotonicCubicInterpolant(z41, z42, z43, z44, t.y);
                   
    ret.x = monotonicCubicInterpolant(y1, y2, y3, y4, t.x);
  }


  // Y component ---

  p.setValue(vsP.x - 0.5, vsP.y , vsP.z - 0.5);

  // Lower left corner
  c.x = static_cast<int>(floor(p.x));
  c.y = static_cast<int>(floor(p.y)); 
  c.z = static_cast<int>(floor(p.z));
    
  t.setValue(p - static_cast<FIELD3D_VEC3_T<double> >(c));
  {                   
    // Clamp the coordinates
    int im, jm, km;
    im = std::max(dataWindow.min.x, std::min(c.x, dataWindow.max.x));
    jm = std::max(dataWindow.min.y, std::min(c.y, dataWindow.max.y + 1));
    km = std::max(dataWindow.min.z, std::min(c.z, dataWindow.max.z));
    int im_1, jm_1, km_1;
    im_1 = std::max(dataWindow.min.x, std::min(im - 1, dataWindow.max.x));
    jm_1 = std::max(dataWindow.min.y, std::min(jm - 1, dataWindow.max.y + 1));
    km_1 = std::max(dataWindow.min.z, std::min(km - 1, dataWindow.max.z));
    int im1, jm1, km1;
    im1 = std::max(dataWindow.min.x, std::min(im + 1, dataWindow.max.x));
    jm1 = std::max(dataWindow.min.y, std::min(jm + 1, dataWindow.max.y + 1));
    km1 = std::max(dataWindow.min.z, std::min(km + 1, dataWindow.max.z));
    int im2, jm2, km2;
    im2 = std::max(dataWindow.min.x, std::min(im + 2, dataWindow.max.x));
    jm2 = std::max(dataWindow.min.y, std::min(jm + 2, dataWindow.max.y + 1));
    km2 = std::max(dataWindow.min.z, std::min(km + 2, dataWindow.max.z));

    T z11 = monotonicCubicInterpolant(data.v(im_1, jm_1, km_1), 
                                      data.v(im_1, jm_1, km), 
                                      data.v(im_1, jm_1, km1), 
                                      data.v(im_1, jm_1, km2), t.z);
    T z12 = monotonicCubicInterpolant(data.v(im_1, jm, km_1), 
                                      data.v(im_1, jm, km), 
                                      data.v(im_1, jm, km1), 
                                      data.v(im_1, jm, km2), t.z);
    T z13 = monotonicCubicInterpolant(data.v(im_1, jm1, km_1), 
                                      data.v(im_1, jm1, km), 
                                      data.v(im_1, jm1, km1), 
                                      data.v(im_1, jm1, km2), t.z);
    T z14 = monotonicCubicInterpolant(data.v(im_1, jm2, km_1), 
                                      data.v(im_1, jm2, km), 
                                      data.v(im_1, jm2, km1), 
                                      data.v(im_1, jm2, km2), t.z);

    T z21 = monotonicCubicInterpolant(data.v(im, jm_1, km_1), 
                                      data.v(im, jm_1, km), 
                                      data.v(im, jm_1, km1), 
                                      data.v(im, jm_1, km2), t.z);
    T z22 = monotonicCubicInterpolant(data.v(im, jm, km_1), 
                                      data.v(im, jm, km), 
                                      data.v(im, jm, km1), 
                                      data.v(im, jm, km2), t.z);
    T z23 = monotonicCubicInterpolant(data.v(im, jm1, km_1), 
                                      data.v(im, jm1, km), 
                                      data.v(im, jm1, km1), 
                                      data.v(im, jm1, km2), t.z);
    T z24 = monotonicCubicInterpolant(data.v(im, jm2, km_1), 
                                      data.v(im, jm2, km), 
                                      data.v(im, jm2, km1), 
                                      data.v(im, jm2, km2), t.z);

    T z31 = monotonicCubicInterpolant(data.v(im1, jm_1, km_1), 
                                      data.v(im1, jm_1, km), 
                                      data.v(im1, jm_1, km1), 
                                      data.v(im1, jm_1, km2), t.z);
    T z32 = monotonicCubicInterpolant(data.v(im1, jm, km_1), 
                                      data.v(im1, jm, km), 
                                      data.v(im1, jm, km1), 
                                      data.v(im1, jm, km2), t.z);
    T z33 = monotonicCubicInterpolant(data.v(im1, jm1, km_1), 
                                      data.v(im1, jm1, km), 
                                      data.v(im1, jm1, km1), 
                                      data.v(im1, jm1, km2), t.z);
    T z34 = monotonicCubicInterpolant(data.v(im1, jm2, km_1), 
                                      data.v(im1, jm2, km), 
                                      data.v(im1, jm2, km1), 
                                      data.v(im1, jm2, km2), t.z);

    T z41 = monotonicCubicInterpolant(data.v(im2, jm_1, km_1), 
                                      data.v(im2, jm_1, km), 
                                      data.v(im2, jm_1, km1), 
                                      data.v(im2, jm_1, km2), t.z);
    T z42 = monotonicCubicInterpolant(data.v(im2, jm, km_1), 
                                      data.v(im2, jm, km), 
                                      data.v(im2, jm, km1), 
                                      data.v(im2, jm, km2), t.z);
    T z43 = monotonicCubicInterpolant(data.v(im2, jm1, km_1), 
                                      data.v(im2, jm1, km), 
                                      data.v(im2, jm1, km1), 
                                      data.v(im2, jm1, km2), t.z);
    T z44 = monotonicCubicInterpolant(data.v(im2, jm2, km_1), 
                                      data.v(im2, jm2, km), 
                                      data.v(im2, jm2, km1), 
                                      data.v(im2, jm2, km2), t.z);

    T y1 = monotonicCubicInterpolant(z11, z12, z13, z14, t.y);
    T y2 = monotonicCubicInterpolant(z21, z22, z23, z24, t.y);
    T y3 = monotonicCubicInterpolant(z31, z32, z33, z34, t.y);
    T y4 = monotonicCubicInterpolant(z41, z42, z43, z44, t.y);
                   
    ret.y = monotonicCubicInterpolant(y1, y2, y3, y4, t.x);
  }

  // Z component ---

  p.setValue(vsP.x - 0.5 , vsP.y - 0.5, vsP.z);

  // Lower left corner
  c.x = static_cast<int>(floor(p.x));
  c.y = static_cast<int>(floor(p.y)); 
  c.z = static_cast<int>(floor(p.z));

  t.setValue(p - static_cast<FIELD3D_VEC3_T<double> >(c));
  {                   
    // Clamp the coordinates
    int im, jm, km;
    im = std::max(dataWindow.min.x, std::min(c.x, dataWindow.max.x));
    jm = std::max(dataWindow.min.y, std::min(c.y, dataWindow.max.y));
    km = std::max(dataWindow.min.z, std::min(c.z, dataWindow.max.z + 1));
    int im_1, jm_1, km_1;
    im_1 = std::max(dataWindow.min.x, std::min(im - 1, dataWindow.max.x));
    jm_1 = std::max(dataWindow.min.y, std::min(jm - 1, dataWindow.max.y));
    km_1 = std::max(dataWindow.min.z, std::min(km - 1, dataWindow.max.z + 1));
    int im1, jm1, km1;
    im1 = std::max(dataWindow.min.x, std::min(im + 1, dataWindow.max.x));
    jm1 = std::max(dataWindow.min.y, std::min(jm + 1, dataWindow.max.y));
    km1 = std::max(dataWindow.min.z, std::min(km + 1, dataWindow.max.z + 1));
    int im2, jm2, km2;
    im2 = std::max(dataWindow.min.x, std::min(im + 2, dataWindow.max.x));
    jm2 = std::max(dataWindow.min.y, std::min(jm + 2, dataWindow.max.y));
    km2 = std::max(dataWindow.min.z, std::min(km + 2, dataWindow.max.z + 1));

    T z11 = monotonicCubicInterpolant(data.w(im_1, jm_1, km_1), 
                                      data.w(im_1, jm_1, km), 
                                      data.w(im_1, jm_1, km1), 
                                      data.w(im_1, jm_1, km2), t.z);
    T z12 = monotonicCubicInterpolant(data.w(im_1, jm, km_1), 
                                      data.w(im_1, jm, km), 
                                      data.w(im_1, jm, km1), 
                                      data.w(im_1, jm, km2), t.z);
    T z13 = monotonicCubicInterpolant(data.w(im_1, jm1, km_1), 
                                      data.w(im_1, jm1, km), 
                                      data.w(im_1, jm1, km1), 
                                      data.w(im_1, jm1, km2), t.z);
    T z14 = monotonicCubicInterpolant(data.w(im_1, jm2, km_1), 
                                      data.w(im_1, jm2, km), 
                                      data.w(im_1, jm2, km1), 
                                      data.w(im_1, jm2, km2), t.z);

    T z21 = monotonicCubicInterpolant(data.w(im, jm_1, km_1), 
                                      data.w(im, jm_1, km), 
                                      data.w(im, jm_1, km1), 
                                      data.w(im, jm_1, km2), t.z);
    T z22 = monotonicCubicInterpolant(data.w(im, jm, km_1), 
                                      data.w(im, jm, km), 
                                      data.w(im, jm, km1), 
                                      data.w(im, jm, km2), t.z);
    T z23 = monotonicCubicInterpolant(data.w(im, jm1, km_1), 
                                      data.w(im, jm1, km), 
                                      data.w(im, jm1, km1), 
                                      data.w(im, jm1, km2), t.z);
    T z24 = monotonicCubicInterpolant(data.w(im, jm2, km_1), 
                                      data.w(im, jm2, km), 
                                      data.w(im, jm2, km1), 
                                      data.w(im, jm2, km2), t.z);

    T z31 = monotonicCubicInterpolant(data.w(im1, jm_1, km_1), 
                                      data.w(im1, jm_1, km), 
                                      data.w(im1, jm_1, km1), 
                                      data.w(im1, jm_1, km2), t.z);
    T z32 = monotonicCubicInterpolant(data.w(im1, jm, km_1), 
                                      data.w(im1, jm, km), 
                                      data.w(im1, jm, km1), 
                                      data.w(im1, jm, km2), t.z);
    T z33 = monotonicCubicInterpolant(data.w(im1, jm1, km_1), 
                                      data.w(im1, jm1, km), 
                                      data.w(im1, jm1, km1), 
                                      data.w(im1, jm1, km2), t.z);
    T z34 = monotonicCubicInterpolant(data.w(im1, jm2, km_1), 
                                      data.w(im1, jm2, km), 
                                      data.w(im1, jm2, km1), 
                                      data.w(im1, jm2, km2), t.z);

    T z41 = monotonicCubicInterpolant(data.w(im2, jm_1, km_1), 
                                      data.w(im2, jm_1, km), 
                                      data.w(im2, jm_1, km1), 
                                      data.w(im2, jm_1, km2), t.z);
    T z42 = monotonicCubicInterpolant(data.w(im2, jm, km_1), 
                                      data.w(im2, jm, km), 
                                      data.w(im2, jm, km1), 
                                      data.w(im2, jm, km2), t.z);
    T z43 = monotonicCubicInterpolant(data.w(im2, jm1, km_1), 
                                      data.w(im2, jm1, km), 
                                      data.w(im2, jm1, km1), 
                                      data.w(im2, jm1, km2), t.z);
    T z44 = monotonicCubicInterpolant(data.w(im2, jm2, km_1), 
                                      data.w(im2, jm2, km), 
                                      data.w(im2, jm2, km1), 
                                      data.w(im2, jm2, km2), t.z);

    T y1 = monotonicCubicInterpolant(z11, z12, z13, z14, t.y);
    T y2 = monotonicCubicInterpolant(z21, z22, z23, z24, t.y);
    T y3 = monotonicCubicInterpolant(z31, z32, z33, z34, t.y);
    T y4 = monotonicCubicInterpolant(z41, z42, z43, z44, t.y);
                   
    ret.z = monotonicCubicInterpolant(y1, y2, y3, y4, t.x);
  }

  return ret;
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T 
ProceduralFieldLookup<Data_T>::sample(const ProceduralField<Data_T> &data,
                                      const V3d &vsP) const 
{
  V3d voxelScale = V3d(1.0) / data.dataResolution();
  V3d lsP = vsP * voxelScale;
  return data.lsSample(lsP);
}

//----------------------------------------------------------------------------//

template <class S, class T>
FIELD3D_VEC3_T<T> operator * (S s, const FIELD3D_VEC3_T<T> vec)
{
  return FIELD3D_VEC3_T<T>(vec.x * s, vec.y * s, vec.z * s);
}

//----------------------------------------------------------------------------//

template<class T>
T monotonicCubicInterpolant(const T &f1, const T &f2, const T &f3, const T &f4, 
                            double t)
{
  T d_k = T(.5) * (f3 - f1);
  T d_k1 = T(.5) * (f4 - f2);
  T delta_k = f3 - f2;

  if (delta_k == static_cast<T>(0)) {
    d_k = static_cast<T>(0);
    d_k1 = static_cast<T>(0);
  }

  T a0 = f2;
  T a1 = d_k;
  T a2 = (T(3) * delta_k) - (T(2) * d_k) - d_k1;
  T a3 = d_k + d_k1 - (T(2) * delta_k);

  T t1 = t;
  T t2 = t1 * t1;
  T t3 = t2 * t1;

  return a3 * t3 + a2 * t2 + a1 * t1 + a0;
}

//----------------------------------------------------------------------------//

//! Monotonic cubic interpolation on 3-vectors
// References:
// http://en.wikipedia.org/wiki/Monotone_cubic_interpolation
// http://en.wikipedia.org/wiki/Cubic_Hermite_spline
template <class Data_T>
Data_T monotonicCubicInterpolantVec(const Data_T &f1, const Data_T &f2, 
                                    const Data_T &f3, const Data_T &f4, 
                                    double t)
{
  typedef typename Data_T::BaseType T;

  Data_T d_k     = T(.5) * (f3 - f1);
  Data_T d_k1    = T(.5) * (f4 - f2);
  Data_T delta_k = f3 - f2;

  for (int i = 0; i < 3; i++) {
    if (delta_k[i] == static_cast<T>(0)) {
      d_k[i] = static_cast<T>(0);
      d_k1[i]= static_cast<T>(0);
    }
  }

  Data_T a0 = f2;
  Data_T a1 = d_k;
  Data_T a2 = (delta_k * T(3)) - (d_k * T(2)) - d_k1;
  Data_T a3 = d_k + d_k1 - (delta_k * T(2));

  T t1 = t;
  T t2 = t1 * t1;
  T t3 = t2 * t1;

  return a3 * t3 + a2 * t2 + a1 * t1 + a0;
}

//----------------------------------------------------------------------------//
// Template specializations
//----------------------------------------------------------------------------//

template<>
inline
V3h monotonicCubicInterpolant<V3h>(const V3h &f1, const V3h &f2, 
                                   const V3h &f3, const V3h &f4, double t)
{
  return monotonicCubicInterpolantVec(f1, f2, f3, f4, t);
}

//----------------------------------------------------------------------------//

template<>
inline
V3f monotonicCubicInterpolant<V3f>(const V3f &f1, const V3f &f2, 
                                   const V3f &f3, const V3f &f4, double t)
{
  return monotonicCubicInterpolantVec(f1, f2, f3, f4, t);
}

//----------------------------------------------------------------------------//

template<>
inline
V3d monotonicCubicInterpolant<V3d>(const V3d &f1, const V3d &f2, 
                                   const V3d &f3, const V3d &f4, double t)
{
  return monotonicCubicInterpolantVec(f1, f2, f3, f4, t);
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
