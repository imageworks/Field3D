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

/*! \file FieldMapping.h
  \ingroup field
  \brief Contains the FieldMapping base class and the NullFieldMapping and
  MatrixFieldMapping subclasses.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_FieldMapping_H_
#define _INCLUDED_Field3D_FieldMapping_H_

#include <vector>
#include <algorithm>

#include "Curve.h"
#include "Exception.h"
#include "RefCount.h"
#include "Types.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// FieldMapping
//----------------------------------------------------------------------------//

/*! \class FieldMapping
  \ingroup field
  \brief Base class for mapping between world-, local- and voxel coordinates.

  Refer to \ref using_mappings for examples of how to use this in your code.

  Local coordinates (ls) are defined as [0,1] over the FieldData object's
  -extents- (not data window). Thus, if the extents.min isn't at origin, 
  the coordinate system stays the same as if it was. 

  Voxel coordinates (vs) are defined as [0,size-1] over the FieldData object's
  -extents- (not data window).

  \note The center of a voxel at (i,j) in integer coordinates is (i+0.5,j+0.5) 
  in continuous coordinates.
*/

//----------------------------------------------------------------------------//

class FIELD3D_API FieldMapping : public RefBase
{
 public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::intrusive_ptr<FieldMapping> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef FieldMapping class_type;
  DEFINE_FIELD_RTTI_ABSTRACT_CLASS;
  
  static const char* classType()
  {
    return "FieldMapping";
  }

  // Ctors, dtor ---------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  //! Constructor
  FieldMapping();
  //! Construct with known extents
  FieldMapping(const Box3i &extents);
  //! Destructor
  virtual ~FieldMapping();

  //! \}

  // Main methods --------------------------------------------------------------

  //! This sets the field extents information to use for defining the
  //! local coordinate space.
  //! \note You need to call this again if you change the resolution of the
  //! field after creation. We grab the extents information in this call because
  //! it would be too slow to do so in every transformation.
  void setExtents(const Box3i &extents);

  //! Returns the origin
  const V3d& origin() const
  { return m_origin; }
  //! Returns the resolution
  const V3d& resolution() const
  { return m_res; }
  
  // To be implemented by subclasses -------------------------------------------

  //! \name To be implemented by subclasses
  //! \{

  //! Returns a pointer to a copy of the mapping, pure virtual so ensure
  //! derived classes properly implement it
  virtual Ptr clone() const = 0;

  //! Transform from world space position into voxel space
  virtual void worldToVoxel(const V3d &wsP, V3d &vsP) const = 0;
  virtual void worldToVoxel(const V3d &wsP, V3d &vsP, float time) const = 0;
  //! Transform from voxel space position into world space
  virtual void voxelToWorld(const V3d &vsP, V3d &wsP) const = 0;
  virtual void voxelToWorld(const V3d &vsP, V3d &wsP, float time) const = 0;
  //! Transform from world space position into local space
  virtual void worldToLocal(const V3d &wsP, V3d &lsP) const = 0;
  virtual void worldToLocal(const V3d &wsP, V3d &lsP, float time) const = 0;
  //! Transform from local space position into world space
  virtual void localToWorld(const V3d &lsP, V3d &wsP) const = 0;
  virtual void localToWorld(const V3d &lsP, V3d &wsP, float time) const = 0;

  //! Returns world-space size of a voxel at the specified coordinate
  virtual V3d wsVoxelSize(int i, int j, int k) const = 0;

  //! Implement this if the subclass needs to update itself when the 
  //! resolution changes.
  virtual void extentsChanged()
  { /* Empty */ }
  
  //! Returns the FieldMapping type name. Used when writing/reading from disk
  virtual std::string className() const = 0;

  //! Whether the mapping is identical to another mapping
  virtual bool isIdentical(FieldMapping::Ptr other, 
                           double tolerance = 0.0) const = 0;

  //! \}

  // Transform calls -----------------------------------------------------------

  //! \name Transforms implemented in this class
  //! \{

  //! Transform from local space to voxel space. This is just a multiplication
  //! by the resolution of the Field that we're mapping.
  void localToVoxel(const V3d &lsP, V3d &vsP) const;
  //! Inverse of localToVoxel.
  void voxelToLocal(const V3d &vsP, V3d &lsP) const;

  //! \}
  
protected:

  //! The integer voxel-space origin of the underlying Field object.
  //! Is equal to field.extents.min
  V3d m_origin;
  //! The integer voxel-space resolution of the underlying Field object.
  //! Is equal to field.extents.max - field.extents.min + 1
  V3d m_res;

private:

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef RefBase base;  

};

//----------------------------------------------------------------------------//
// NullFieldMapping
//----------------------------------------------------------------------------//

/*! \class NullFieldMapping
  \ingroup field
  \brief Trivial class, world space is equal to local space, i.e. the field
  is contained in the unit cube [0..1] in all axes.

  Refer to \ref using_mappings for examples of how to use this in your code.
*/

//----------------------------------------------------------------------------//

class FIELD3D_API NullFieldMapping : public FieldMapping
{
public:

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef
  typedef boost::intrusive_ptr<NullFieldMapping> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef NullFieldMapping class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;
  
  static const char* classType()
  {
    return "NullFieldMapping";
  }

  // Ctors, dtor ---------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  NullFieldMapping()
    : FieldMapping()
  { /* Empty */ }
  NullFieldMapping(const Box3i &extents)
    : FieldMapping(extents)
  { /* Empty */ }

  //! \}

  // From FieldMapping ---------------------------------------------------------

  //! \name From FieldMapping
  //! \{

  virtual void worldToVoxel(const V3d &wsP, V3d &vsP) const 
  { localToVoxel(wsP, vsP); }
  virtual void worldToVoxel(const V3d &wsP, V3d &vsP, float /*time*/) const 
  { localToVoxel(wsP, vsP); }

  virtual void voxelToWorld(const V3d &vsP, V3d &wsP) const 
  { voxelToLocal(vsP, wsP); }
  virtual void voxelToWorld(const V3d &vsP, V3d &wsP, float /*time*/) const 
  { voxelToLocal(vsP, wsP); }

  virtual void worldToLocal(const V3d &wsP, V3d &lsP) const 
  { lsP = wsP; }
  virtual void worldToLocal(const V3d &wsP, V3d &lsP, float /*time*/) const 
  { lsP = wsP; }

  virtual void localToWorld(const V3d &lsP, V3d &wsP) const 
  { wsP = lsP; }
  virtual void localToWorld(const V3d &lsP, V3d &wsP, float /*time*/) const 
  { wsP = lsP; }

  virtual std::string className() const;

  virtual bool isIdentical(FieldMapping::Ptr other, 
                           double tolerance = 0.0) const;

  virtual V3d wsVoxelSize(int /*i*/, int /*j*/, int /*k*/) const
  { return V3d(1.0 / m_res.x, 1.0 / m_res.y, 1.0 / m_res.z); }

  virtual FieldMapping::Ptr clone() const;

  //! \}
  
private:

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef FieldMapping base;  

};

//----------------------------------------------------------------------------//
// MatrixFieldMapping
//----------------------------------------------------------------------------//

/*! \class MatrixFieldMapping
  \ingroup field
  \brief Represents the mapping of a field by a matrix transform

  Refer to \ref using_mappings for examples of how to use this in your code.

  \note Regarding time-varying matrices. If setLocalToWorld(M44d) is called,
  an underlying Curve object is created with just one sample at time=0.0.

  \todo Add calls for easily specifying the transform given grid size,
  offset, rotation, etc.
*/

//----------------------------------------------------------------------------//

class FIELD3D_API MatrixFieldMapping : public FieldMapping
{
public:

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef
  typedef boost::intrusive_ptr<MatrixFieldMapping> Ptr;
  //! Time-varying matrix
  typedef Curve<Imath::M44d> MatrixCurve;

  // RTTI replacement ----------------------------------------------------------

  typedef MatrixFieldMapping class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;
  
  static const char* classType ()
  {
    return "MatrixFieldMapping";
  }

  // Ctors, dtor ---------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  MatrixFieldMapping();
  MatrixFieldMapping(const Box3i &extents);

  //! \}

  // Main methods --------------------------------------------------------------
  
  //! Sets the local to world transform. All other matrices will be updated
  //! based on this.
  //! \note This resets the Curve to contain just one sample at time=0.0
  void setLocalToWorld(const M44d &lsToWs);
  //! Sets the local to world transform at a given time.
  void setLocalToWorld(float t, const M44d &lsToWs);

  //! Returns a reference to the local to world transform. 
  //! \note This assumes the query to be at time=0.0
  const M44d& localToWorld() const
  { return m_lsToWs; }

  //! Returns a reference to the world to voxel space transform. 
  //! \note This assumes the query to be at time=0.0
  const M44d& worldToVoxel() const
  { return m_wsToVs; }

  //! Returns a reference to the voxel to world space transform. 
  //! \note This assumes the query to be at time=0.0
  const M44d& voxelToWorld() const
  { return m_vsToWs; }

  //! Returns a vector of all motion samples for local to world transform.
  const MatrixCurve::SampleVec& localToWorldSamples() const
  { return m_lsToWsCurve.samples(); } 

  //! Sets the transform to identity. This makes it functionally equivalent to
  //! a NullFieldMapping.
  void makeIdentity();

  // From FieldMapping ---------------------------------------------------------

  //! \name From FieldMapping
  //! \{

  virtual void worldToVoxel(const V3d &wsP, V3d &vsP) const 
  { m_wsToVs.multVecMatrix(wsP, vsP); }
  virtual void worldToVoxel(const V3d &wsP, V3d &vsP, float time) const 
  { 
    if (!m_isTimeVarying) {
      m_wsToVs.multVecMatrix(wsP, vsP);
    } else {
      M44d wsToVs = m_vsToWsCurve.linear(time).inverse();
      wsToVs.multVecMatrix(wsP, vsP);
    }
  }

  virtual void voxelToWorld(const V3d &vsP, V3d &wsP) const 
  { m_vsToWs.multVecMatrix(vsP, wsP); }
  virtual void voxelToWorld(const V3d &vsP, V3d &wsP, float time) const 
  { 
    if (!m_isTimeVarying) {
      m_vsToWs.multVecMatrix(vsP, wsP); 
    } else {
      M44d vsToWs = m_vsToWsCurve.linear(time);
      vsToWs.multVecMatrix(vsP, wsP);
    }
  }

  virtual void worldToLocal(const V3d &wsP, V3d &lsP) const 
  { m_wsToLs.multVecMatrix(wsP, lsP); }
  virtual void worldToLocal(const V3d &wsP, V3d &lsP,
                            float time) const 
  { 
    if (!m_isTimeVarying) {
      m_wsToLs.multVecMatrix(wsP, lsP); 
    } else {
      M44d wsToLs = m_lsToWsCurve.linear(time).inverse();
      wsToLs.multVecMatrix(wsP, lsP);
    }
  }

  virtual void localToWorld(const V3d &lsP, V3d &wsP) const 
  { m_lsToWs.multVecMatrix(lsP, wsP); }
  virtual void localToWorld(const V3d &lsP, V3d &wsP, float time) const 
  { 
    if (!m_isTimeVarying) {
      m_lsToWs.multVecMatrix(lsP, wsP); 
    } else {
      M44d lsToWs = m_lsToWsCurve.linear(time);
      lsToWs.multVecMatrix(lsP, wsP);
    }
  }

  //! \todo Generalize and make time-dependent.
  void worldToVoxelDir(const V3d &wsV, V3d &vsV) const 
  { m_wsToVs.multDirMatrix(wsV, vsV); }

  //! \todo Generalize and make time-dependent.
  void voxelToWorldDir(const V3d &vsV, V3d &wsV) const 
  { m_vsToWs.multDirMatrix(vsV, wsV); }

  //! \todo Generalize and make time-dependent.
  void worldToLocalDir(const V3d &wsV, V3d &lsV) const 
  { m_wsToLs.multDirMatrix(wsV, lsV); }

  //! \todo Generalize and make time-dependent.
  void localToWorldDir(const V3d &lsV, V3d &wsV) const 
  { m_lsToWs.multDirMatrix(lsV, wsV); }

  virtual void extentsChanged();

  virtual std::string className() const;

  virtual bool isIdentical(FieldMapping::Ptr other, 
                           double tolerance = 0.0) const;

  virtual V3d wsVoxelSize(int /*i*/, int /*j*/, int /*k*/) const
  { return m_wsVoxelSize; }

  virtual FieldMapping::Ptr clone() const;

  //! \}
  
private:

  //! Updates the local to world transformation matrix
  void updateTransform();

  //! \todo Unit test this
  void getLocalToVoxelMatrix(M44d &result);

  // Data members -------------------------------------------------------------

  //! Local space to world space
  //! \note This is used only when m_lsToWsCurve has zero or one samples.
  M44d m_lsToWs;
  //! World space to local space
  //! \note This is used only when m_lsToWsCurve has zero or one samples.
  M44d m_wsToLs;
  //! Voxel space to world space
  //! \note This is used only when m_lsToWsCurve has zero or one samples.
  M44d m_vsToWs;
  //! World space to voxel space
  //! \note This is used only when m_lsToWsCurve has zero or one samples.
  M44d m_wsToVs;

  //! Time-varying local to world space transform
  MatrixCurve m_lsToWsCurve;
  //! Time-varying voxel to world space transform
  MatrixCurve m_vsToWsCurve;

  //! Stores whether the curve has more than one time sample.
  //! \note This is set by updateTransform().
  bool m_isTimeVarying;

  //! Precomputed world-space voxel size. Calculations may assume orthogonal
  //! transformation for efficiency
  V3d m_wsVoxelSize;

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef FieldMapping base;  
};

//----------------------------------------------------------------------------//
// FrustumFieldMapping
//----------------------------------------------------------------------------//

/*! \class FrustumFieldMapping
  \ingroup field
  \brief Represents the mapping of a field by a perspective transform

  Refer to \ref using_mappings for examples of how to use this in your code.

  Frustum mappings can use two approaches in determining the distribution
  of "Z slices". By transforming from world space into screen space and using
  the Z component in perspective space, the slices in Z will be distributed
  in world space accordingly. It is also possible to use a uniform distribution
  of Z slices by specifying a near and far clip plane and normalizing the
  camera-space Z distance between those. 

  \note Screen space is defined left-handed as [-1.0,1.0] in all three 
  dimensions

  \note Camera space is defined right-handed with the camera looking down
  negative Z.

  \todo Define local perspective space

  \note Regarding time-varying matrices. If setTransforms() is called,
  an underlying Curve object is created with just one sample at time=0.0.
*/

//----------------------------------------------------------------------------//

class FIELD3D_API FrustumFieldMapping : public FieldMapping
{
public:

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef
  typedef boost::intrusive_ptr<FrustumFieldMapping> Ptr;
  //! Time-varying matrix
  typedef Curve<Imath::M44d> MatrixCurve;
  //! Time-varying float
  typedef Curve<double> FloatCurve;

  // Exceptions ----------------------------------------------------------------

  DECLARE_FIELD3D_GENERIC_EXCEPTION(BadPerspectiveMatrix, Exc::Exception)

  // Enums ---------------------------------------------------------------------

  //! Enumerates the Z slice distribution. .f3d files will store values as
  //! an int, so be very careful not to change the order of these.
  enum ZDistribution {
    PerspectiveDistribution,
    UniformDistribution
  };

  // RTTI replacement ----------------------------------------------------------

  typedef FrustumFieldMapping class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;
  
  static const char* classType ()
  {
    return "FrustumFieldMapping";
  }

  // Ctors, dtor ---------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  FrustumFieldMapping();
  FrustumFieldMapping(const Box3i &extents);

  //! \}

  // Main methods --------------------------------------------------------------
  
  //! Sets the screenToWorld and cameraToWorld transforms. 
  //! All other internal matrices will be updated based on these.
  //! \note This resets the transform Curve instances to contain just one
  //! sample at time=0.0
  //! \param ssToWs See class documentation for definition.
  //! \param csToWs See class documentation for definition.
  void setTransforms(const M44d &ssToWs, const M44d &csToWs);
  //! Sets time-varying screenToWorld and cameraToWorld transforms.
  //! All other internal matrices will be updated based on these.
  //! \param ssToWs See class documentation for definition.
  //! \param csToWs See class documentation for definition.
  void setTransforms(float t, const M44d &ssToWs, const M44d &csToWs);

  //! Sets the z slice distribution
  void setZDistribution(ZDistribution dist)
  { m_zDistribution = dist; }
  //! Returns the z slice distribution
  ZDistribution zDistribution() const
  { return m_zDistribution; }

  //! Returns a reference to the screen to world space transform. 
  //! \note This assumes the query to be at time=0.0
  const M44d screenToWorld() const
  { return m_ssToWsCurve.linear(0.0); }

  //! Returns a reference to the camera to world space transform. 
  //! \note This assumes the query to be at time=0.0
  const M44d cameraToWorld() const
  { return m_csToWsCurve.linear(0.0); }

  //! Returns a vector of all motion samples for screen to world transform.
  const MatrixCurve::SampleVec& screenToWorldSamples() const
  { return m_ssToWsCurve.samples(); } 

  //! Returns a vector of all motion samples for camera to world transform.
  const MatrixCurve::SampleVec& cameraToWorldSamples() const
  { return m_csToWsCurve.samples(); } 

  //! Returns a vector of all motion samples for near plane.
  const FloatCurve::SampleVec& nearPlaneSamples() const
  { return m_nearCurve.samples(); } 

  //! Returns a vector of all motion samples for far plane.
  const FloatCurve::SampleVec& farPlaneSamples() const
  { return m_farCurve.samples(); } 

  //! Returns the near plane
  double nearPlane() const 
  { return m_nearCurve.linear(0.0); }

  //! Returns the far plane
  double farPlane() const
  { return m_farCurve.linear(0.0); }

  //! Resets the transform. Makes a perspective transform at the origin,
  //! looking down the negative Z axis with a 45 degree FOV and square 
  //! projection.
  void reset();

  // From FieldMapping ---------------------------------------------------------

  //! \name From FieldMapping
  //! \{

  virtual void worldToVoxel(const V3d &wsP, V3d &vsP) const;
  virtual void worldToVoxel(const V3d &wsP, V3d &vsP, float time) const;

  virtual void voxelToWorld(const V3d &vsP, V3d &wsP) const;
  virtual void voxelToWorld(const V3d &vsP, V3d &wsP, float time) const;

  virtual void worldToLocal(const V3d &wsP, V3d &lsP) const;
  virtual void worldToLocal(const V3d &wsP, V3d &lsP, float time) const;

  virtual void localToWorld(const V3d &lsP, V3d &wsP) const;
  virtual void localToWorld(const V3d &lsP, V3d &wsP, float time) const;

  virtual void extentsChanged();

  virtual std::string className() const;

  virtual bool isIdentical(FieldMapping::Ptr other, 
                           double tolerance = 0.0) const;

  virtual V3d wsVoxelSize(int i, int j, int k) const;

  virtual FieldMapping::Ptr clone() const;

  //! \}
  
private:

  //! Updates the local to world transformation matrix
  void computeVoxelSize();

  //! \todo Unit test this
  void getLocalToVoxelMatrix(M44d &result);

  //! Clears all Curve data members. Used by setTransforms() to prepare
  //! for the first sample to be added.
  void clearCurves();

  // Data members -------------------------------------------------------------

  //! Slice distribution type
  ZDistribution m_zDistribution;

  //! Time-varying local perspective to world space transform
  //! This is not used in calculations, but rather as the public interface
  //! to the class.
  MatrixCurve m_ssToWsCurve;
  //! Time-varying camera to world space transform
  MatrixCurve m_csToWsCurve;
  //! Time-varying local perspective to world space transform.
  //! Computed from m_ssToWsCurve
  MatrixCurve m_lpsToWsCurve;
  //! Time-varying near plane. Computed from m_lpsToWsCurve
  FloatCurve m_nearCurve;
  //! Time-varying far plane. Computed from m_lpsToWsCurve
  FloatCurve m_farCurve;

  //! Precomputed world-space voxel size. Calculations may assume orthogonal
  //! transformation for efficiency
  std::vector<V3d> m_wsVoxelSize;

  //! Boolean to tell us if the mapping is in its 'default' state.
  //! This is needed because the class has a default configuration where
  //! there is a single sample in all the curves. Once a new transform is
  //! set through setTransforms(), the default samples must be cleared.
  bool m_defaultState;

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef FieldMapping base;

};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
