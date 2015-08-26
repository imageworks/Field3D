//----------------------------------------------------------------------------//

#ifndef __F3DUTIL_FIELDWRAPPER_H__
#define __F3DUTIL_FIELDWRAPPER_H__

//------------------------------------------------------------------------------

// Library includes
#include <OpenEXR/ImathMatrixAlgo.h>

// Project includes
#include "DenseField.h"
#include "Field3DFile.h"
#include "FieldInterp.h"
#include "InitIO.h"
#include "MIPField.h"
#include "MIPUtil.h"
#include "SparseField.h"
#include "FieldSampler.h"
#include "FieldMapping.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//------------------------------------------------------------------------------
// ValueRemapOp
//------------------------------------------------------------------------------

//! The ValueRemapOp class is used when shader-like calculations need to
//! be applied to individual fields that are part of a FieldGroup.
//! Use FieldGroup::setValueRemap() to set the current op before loading the
//! corresponding fields. Then, once lookups take place, the operator is 
//! called upon to remap the resulting values.
//! \note The class is not templated, and it needs to handle both scalar
//! and vector values.
class ValueRemapOp
{
public:
  
  // Typedefs ---

  typedef boost::shared_ptr<ValueRemapOp> Ptr;

  // To be implemented by subclasses ---

  //! Remaps a float value
  virtual float remap(const float value) const = 0;
  //! Remaps a V3f value
  virtual V3f   remap(const V3f &value)  const = 0;

};

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
      vsBounds(continuousBounds(f->dataWindow())),
      worldScale(1.0), 
      doOsToWs(false),
      doWsBoundsOptimization(false),
      valueRemapOp(NULL)
  { }

  void setOsToWs(const M44d &i_osToWs)
  {
    osToWs   = i_osToWs;
    wsToOs   = osToWs.inverse();
    // Compute world scale
    V3d ws(1.0);
    if (!Imath::extractScaling(osToWs, ws, false)) {
      Msg::print("WARNING: FieldGroup/FieldWrapper: "
                 "Couldn't extract world scale from object-to-world "
                 "transform. Defaulting to 1.0.");
    }
    worldScale = std::max(std::max(ws.x, ws.y), ws.z);
    // Set boolean
    doOsToWs = true;

    // Update wsBounds
    if (doWsBoundsOptimization) {
      setWsBoundsOptimization(doWsBoundsOptimization);
    }
  }

  void setWsBoundsOptimization(const bool doWsBoundsOptimization_)
  {
    if (!doWsBoundsOptimization_)
      return;
    // wsBounds can be set only if mapping is a matrix
    const MatrixFieldMapping *mtx_mapping = 
      dynamic_cast<const MatrixFieldMapping*>(mapping);
    if (mtx_mapping) {
      const float time = 0;
      M44d vsToWs;
      if (doOsToWs) {
        wsToVs = wsToOs * mtx_mapping->worldToVoxel(time);
        vsToWs = wsToVs.inverse();
      } else {
        wsToVs = mtx_mapping->worldToVoxel(time);
        vsToWs = wsToVs.inverse();
      }
      const Imath::Box3d wsBounds_d = Imath::transform(vsBounds,
                                                       vsToWs);
      wsBounds = Imath::Box3f(wsBounds_d.min, wsBounds_d.max);
      doWsBoundsOptimization = true;
    }
  }

  void setValueRemapOp(ValueRemapOp::Ptr op)
  {
    valueRemapOpPtr = op;
    valueRemapOp    = valueRemapOpPtr.get();
  }

  typename Field_T::LinearInterp  interp;
  const Field_T                  *field;
  typename Field_T::Ptr           fieldPtr;
  const Field3D::FieldMapping    *mapping;
  Box3d                           vsBounds;
  //! Optionally, enable doOsToWs to apply a world to object transform before
  //! lookups.
  M44d                            osToWs, wsToOs;
  double                          worldScale;
  bool                            doOsToWs;
  //! Optionally, enable wsBounds optimization to use a world axis
  //! aligned bounding box in lookups.
  M44d                            wsToVs;
  Imath::Box3f                    wsBounds;
  bool                            doWsBoundsOptimization;
  //! Optionally, set a ValueRemapOp to remap values
  ValueRemapOp::Ptr               valueRemapOpPtr;
  const ValueRemapOp             *valueRemapOp;
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
      vsBounds(continuousBounds(f->dataWindow())),
      worldScale(1.0), 
      doOsToWs(false),
      valueRemapOp(NULL)
  { 
    interp = interpPtr.get();
  }

  void setOsToWs(const M44d &i_osToWs)
  {
    osToWs   = i_osToWs;
    wsToOs   = osToWs.inverse();
    // Compute world scale
    V3d ws(1.0);
    if (!Imath::extractScaling(osToWs, ws, false)) {
      Msg::print("WARNING: FieldGroup/FieldWrapper: "
                 "Couldn't extract world scale from object-to-world "
                 "transform. Defaulting to 1.0.");
    }
    worldScale = std::max(std::max(ws.x, ws.y), ws.z);
    // Set boolean
    doOsToWs = true;

    // Update wsBounds
    if (doWsBoundsOptimization) {
      setWsBoundsOptimization(doWsBoundsOptimization);
    }
  }

  void setWsBoundsOptimization(const bool doWsBoundsOptimization_)
  {
    if (!doWsBoundsOptimization_)
      return;
    // wsBounds can be set only if mapping is a matrix
    const MatrixFieldMapping *mtx_mapping = 
      dynamic_cast<const MatrixFieldMapping*>(mapping);
    if (mtx_mapping) {
      const float time = 0;
      M44d vsToWs;
      if (doOsToWs) {
        wsToVs = wsToOs * mtx_mapping->worldToVoxel(time);
        vsToWs = wsToVs.inverse();
      } else {
        wsToVs = mtx_mapping->worldToVoxel(time);
        vsToWs = wsToVs.inverse();
      }
      const Imath::Box3d wsBounds_d = Imath::transform(vsBounds,
                                                       vsToWs);
      wsBounds = Imath::Box3f(wsBounds_d.min, wsBounds_d.max);
      doWsBoundsOptimization = true;
    }
  }

  void setValueRemapOp(ValueRemapOp::Ptr op)
  {
    valueRemapOpPtr = op;
    valueRemapOp    = valueRemapOpPtr.get();
  }

  boost::shared_ptr<LinearInterp>  interpPtr;
  LinearInterp                    *interp;
  const Field_T                   *field;
  typename Field_T::Ptr            fieldPtr;
  const Field3D::FieldMapping     *mapping;
  Box3d                            vsBounds;
  //! Optionally, enable doOsToWs to apply a world to object transform before
  //! lookups.
  M44d                             osToWs, wsToOs;
  double                           worldScale;
  bool                             doOsToWs;
  //! Optionally, enable wsBounds optimization to use a world axis
  //! aligned bounding box in lookups.
  M44d                             wsToVs;
  Imath::Box3f                     wsBounds;
  bool                             doWsBoundsOptimization;
  //! Optionally, set a ValueRemapOp to remap values
  ValueRemapOp::Ptr                valueRemapOpPtr;
  const ValueRemapOp              *valueRemapOp;
};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//------------------------------------------------------------------------------

#endif // include guard

//------------------------------------------------------------------------------
