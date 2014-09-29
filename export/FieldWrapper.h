//----------------------------------------------------------------------------//

#ifndef __F3DUTIL_FIELDWRAPPER_H__
#define __F3DUTIL_FIELDWRAPPER_H__

//------------------------------------------------------------------------------

// Field3D includes
#include <lava/field3d15/DenseField.h>
#include <lava/field3d15/Field3DFile.h>
#include <lava/field3d15/FieldInterp.h>
#include <lava/field3d15/InitIO.h>
#include <lava/field3d15/MIPField.h>
#include <lava/field3d15/MIPUtil.h>
#include <lava/field3d15/SparseField.h>

// Project includes
#include "FieldSampler.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//------------------------------------------------------------------------------
// FieldWrapper
//------------------------------------------------------------------------------

//! This class wraps up a single field to make its interpolator and its
//! mapping easily accessible. The 'Vec' typedef gives access to a std::vector.
template <typename Field_T>
struct FieldWrapper
{
  typedef Field_T                   field_type;
  typedef std::vector<FieldWrapper> Vec;

  FieldWrapper(const typename Field_T::Ptr f)
    : field(f.get()), 
      fieldPtr(f), 
      mapping(f->mapping().get()), 
      vsBounds(continuousBounds(f->dataWindow()))
  { }

  typename Field_T::LinearInterp  interp;
  const Field_T                  *field;
  typename Field_T::Ptr           fieldPtr;
  const Field3D::FieldMapping    *mapping;
  Box3d                           vsBounds;
};

//------------------------------------------------------------------------------
// MIPFieldWrapper
//------------------------------------------------------------------------------

//! This class wraps up a single MIP field to make its interpolator and its
//! mapping easily accessible. The 'Vec' typedef gives access to a std::vector.
template <typename Field_T>
struct MIPFieldWrapper
{
  typedef Field_T                        field_type;
  typedef std::vector<MIPFieldWrapper>   Vec;
  typedef typename Field_T::LinearInterp LinearInterp;

  MIPFieldWrapper(const typename Field_T::Ptr f)
    : interpPtr(new LinearInterp(*f)), 
      field(f.get()), 
      fieldPtr(f), 
      mapping(f->mapping().get()), 
      vsBounds(continuousBounds(f->dataWindow()))
  { 
    interp = interpPtr.get();
  }

  boost::shared_ptr<LinearInterp>  interpPtr;
  LinearInterp                    *interp;
  const Field_T                   *field;
  typename Field_T::Ptr            fieldPtr;
  const Field3D::FieldMapping     *mapping;
  Box3d                            vsBounds;
};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//------------------------------------------------------------------------------

#endif // include guard

//------------------------------------------------------------------------------
