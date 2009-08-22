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

#include <boost/intrusive_ptr.hpp>

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

class FieldMapping
{
 public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::intrusive_ptr<FieldMapping> Ptr;

  // Ctors, dtor ---------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  FieldMapping();
  FieldMapping(const Box3i &extents);
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

  // Reference counting --------------------------------------------------------

  size_t refcnt(void) 
  { return m_counter; }

  void ref(void) 
  { m_counter++; }

  void unref(void) 
  { m_counter--; }
  
  // To be implemented by subclasses -------------------------------------------

  //! \name To be implemented by subclasses
  //! \{

  //! Returns a pointer to a copy of the mapping, pure virtual so ensure
  //! derived classes properly implement it
  virtual Ptr clone() const = 0;

  //! Transform from world space position into voxel space
  virtual void worldToVoxel(const V3d &wsP, V3d &vsP) const = 0;
  virtual void worldToVoxel(const V3d &wsP, V3d &vsP, float time) const = 0;
  //! Transforms multiple positions at once. Mainly used to avoid the
  //! overhad of virtual calls when transforming large quantities of points
  //! \note This would ideally be templated on the storage container, but since
  //! we can't have templated virtual calls, we only support std::vector for now
  virtual void worldToVoxel(std::vector<V3d>::const_iterator wsP, 
                            std::vector<V3d>::const_iterator end, 
                            std::vector<V3d>::iterator vsP) const = 0;
  //! Transform from voxel space position into world space
  virtual void voxelToWorld(const V3d &vsP, V3d &wsP) const = 0;
  //! Transform from world space position into local space
  virtual void worldToLocal(const V3d &wsP, V3d &lsP) const = 0;
  virtual void worldToLocal(const V3d &wsP, V3d &lsP, float time) const = 0;
  //! Transforms multiple positions at once. Mainly used to avoid the
  //! overhad of virtual calls when transforming large quantities of points
  //! \note This would ideally be templated on the storage container, but since
  //! we can't have templated virtual calls, we only support std::vector for now
  virtual void worldToLocal(std::vector<V3d>::const_iterator wsP, 
                            std::vector<V3d>::const_iterator end, 
                            std::vector<V3d>::iterator lsP) const = 0;
  //! Transform from local space position into world space
  virtual void localToWorld(const V3d &lsP, V3d &wsP) const = 0;

  //! Returns world-space size of a voxel at the specified coordinate
  virtual V3d wsVoxelSize(int i, int j, int k) const = 0;

  //! Implement this if the subclass needs to update itself when the 
  //! resolution changes.
  virtual void extentsChanged()
  { /* Empty */ }
  
  //! Returns the FieldMapping type name. Used when writing/reading from disk
  virtual std::string typeName() const = 0;
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
  void localToVoxel(const V3d &lsP, V3d &vsP, float time) const;
  void localToVoxel(std::vector<V3d>::const_iterator lsP, 
                    std::vector<V3d>::const_iterator end, 
                    std::vector<V3d>::iterator vsP) const;
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

  //! Reference counter
  mutable int m_counter;
};

//----------------------------------------------------------------------------//
// Intrusive Pointer reference counting 
//----------------------------------------------------------------------------//

inline void
intrusive_ptr_add_ref(FieldMapping* r)
{
  r->ref();
}

inline void
intrusive_ptr_release(FieldMapping* r)
{
  r->unref();

  if (r->refcnt() == 0)
    delete r;
}

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

class NullFieldMapping : public FieldMapping
{
public:

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef
  typedef boost::intrusive_ptr<NullFieldMapping> Ptr;
  
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
  virtual void worldToVoxel(const V3d &wsP, V3d &vsP,
                            float time) const 
  { localToVoxel(wsP, vsP, time); }
  virtual void worldToVoxel(std::vector<V3d>::const_iterator wsP,
                            std::vector<V3d>::const_iterator end,
                            std::vector<V3d>::iterator vsP) const 
  { localToVoxel(wsP, end, vsP); }
  virtual void voxelToWorld(const V3d &vsP, V3d &wsP) const 
  { voxelToLocal(vsP, wsP); }
  virtual void worldToLocal(const V3d &wsP, V3d &lsP) const 
  { lsP = wsP; }
  virtual void worldToLocal(const V3d &wsP, V3d &lsP,
                            float /*time*/) const 
  { lsP = wsP; }
  virtual void worldToLocal(std::vector<V3d>::const_iterator wsP,
                            std::vector<V3d>::const_iterator end,
                            std::vector<V3d>::iterator lsP) const 
  { std::copy(wsP, end, lsP); }
  virtual void localToWorld(const V3d &lsP, V3d &wsP) const 
  { wsP = lsP; }
  virtual std::string typeName() const;
  virtual bool isIdentical(FieldMapping::Ptr other, 
                           double tolerance = 0.0) const;
  virtual V3d wsVoxelSize(int /*i*/, int /*j*/, int /*k*/) const
  { return V3d(1.0 / m_res.x, 1.0 / m_res.y, 1.0 / m_res.z); }

  virtual FieldMapping::Ptr clone() const;

  //! \}

};

//----------------------------------------------------------------------------//
// MatrixFieldMapping
//----------------------------------------------------------------------------//

/*! \class MatrixFieldMapping
  \ingroup field
  \brief Represents the mapping of a field by a matrix transform

  Refer to \ref using_mappings for examples of how to use this in your code.

  \todo Add calls for easily specifying the transform given grid size,
  offset, rotation, etc.
*/

//----------------------------------------------------------------------------//

class MatrixFieldMapping : public FieldMapping
{
public:

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef
  typedef boost::intrusive_ptr<MatrixFieldMapping> Ptr;

  // Ctors, dtor ---------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  MatrixFieldMapping();
  MatrixFieldMapping(const Box3i &extents);

  //! \}

  // Main methods --------------------------------------------------------------
  
  //! Sets the local to world transform. All other matrices will be updated
  //! based on this.
  void setLocalToWorld(const M44d &lsToWs);

  //! Returns a reference to the local to world transform. 
  const M44d& localToWorld() const
  { return m_lsToWs; }

  //! Returns a reference to the world to voxel space transform. 
  const M44d& worldToVoxel() const
  { return m_wsToVs; }

  //! Sets the transform to identity. This makes it functionally equivalend to
  //! a NullFieldMapping.
  void makeIndentity();

  // From FieldMapping ---------------------------------------------------------

  //! \name From FieldMapping
  //! \{

  virtual void worldToVoxel(const V3d &wsP, V3d &vsP) const 
  { m_wsToVs.multVecMatrix(wsP, vsP); }

  //! \todo Make MatrixFieldMapping support time-varying matrices
  virtual void worldToVoxel(const V3d &wsP, V3d &vsP,
                            float /*time*/) const 
  { m_wsToVs.multVecMatrix(wsP, vsP); }

  virtual void worldToVoxel(std::vector<V3d>::const_iterator wsP, 
                            std::vector<V3d>::const_iterator end, 
                            std::vector<V3d>::iterator vsP) const 
  { 
    for (; wsP != end; ++wsP, ++vsP) 
      m_wsToVs.multVecMatrix(*wsP, *vsP);
  }

  virtual void voxelToWorld(const V3d &vsP, V3d &wsP) const 
  { m_vsToWs.multVecMatrix(vsP, wsP); }

  virtual void worldToLocal(const V3d &wsP, V3d &lsP) const 
  { m_wsToLs.multVecMatrix(wsP, lsP); }

  //! \todo Make MatrixFieldMapping support time-varying matrices
  virtual void worldToLocal(const V3d &wsP, V3d &lsP,
                            float /*time*/) const 
  { m_wsToLs.multVecMatrix(wsP, lsP); }

  virtual void worldToLocal(std::vector<V3d>::const_iterator wsP, 
                            std::vector<V3d>::const_iterator end, 
                            std::vector<V3d>::iterator lsP) const 
  { 
    for (; wsP != end; ++wsP, ++lsP) 
      m_wsToLs.multVecMatrix(*wsP, *lsP);
  }

  virtual void localToWorld(const V3d &lsP, V3d &wsP) const 
  { m_lsToWs.multVecMatrix(lsP, wsP); }

  void worldToVoxelDir(const V3d &wsV, V3d &vsV) const 
  { m_wsToVs.multDirMatrix(wsV, vsV); }

  void voxelToWorldDir(const V3d &vsV, V3d &wsV) const 
  { m_vsToWs.multDirMatrix(vsV, wsV); }

  void worldToLocalDir(const V3d &wsV, V3d &lsV) const 
  { m_wsToLs.multDirMatrix(wsV, lsV); }

  void localToWorldDir(const V3d &lsV, V3d &wsV) const 
  { m_lsToWs.multDirMatrix(lsV, wsV); }

  virtual void extentsChanged();

  virtual std::string typeName() const;
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

  //! Local space to world space
  M44d m_lsToWs;
  //! World space to local space
  M44d m_wsToLs;
  //! Voxel space to world space
  M44d m_vsToWs;
  //! World space to voxel space
  M44d m_wsToVs;
  //! Precomputed world-space voxel size. Calculations may assume orthogonal
  //! transformation for efficiency
  V3d m_wsVoxelSize;
};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
