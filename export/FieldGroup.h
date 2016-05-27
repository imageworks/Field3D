//----------------------------------------------------------------------------//

#ifndef __F3DUTIL_FIELDGROUP_H__
#define __F3DUTIL_FIELDGROUP_H__

//------------------------------------------------------------------------------

// Boost includes
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/fusion/mpl.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/as_vector.hpp>

// Field3D includes
#include "DenseField.h"
#include "Field3DFile.h"
#include "FieldInterp.h"
#include "FieldWrapper.h"
#include "InitIO.h"
#include "MIPField.h"
#include "MIPUtil.h"
#include "MinMaxUtil.h"
#include "SparseField.h"
#include "TemporalField.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//------------------------------------------------------------------------------
// MPL stuff 
//------------------------------------------------------------------------------

namespace mpl    = boost::mpl;
namespace ph     = mpl::placeholders;
namespace fusion = boost::fusion;
namespace f_ro   = boost::fusion::result_of;

typedef mpl::vector<Field3D::half, float, double>             ScalarTypes;
typedef mpl::vector<Field3D::V3h, Field3D::V3f, Field3D::V3d> VectorTypes;

//------------------------------------------------------------------------------
// Detail namespace
//------------------------------------------------------------------------------

namespace detail {

//------------------------------------------------------------------------------

static const char* k_minSuffix = "_min";
static const char* k_maxSuffix = "_max";

//------------------------------------------------------------------------------

//! MPL utility
template <typename T>
struct MakeDense
{
  typedef typename FieldWrapper<Field3D::DenseField<T> >::Vec type;
};

//------------------------------------------------------------------------------

//! MPL utility
template <typename T>
struct MakeSparse
{
  typedef typename FieldWrapper<Field3D::SparseField<T> >::Vec type;
};

//------------------------------------------------------------------------------

//! MPL utility
template <typename T>
struct MakeTemporal
{
  typedef typename FieldWrapper<Field3D::TemporalField<T> >::Vec type;
};

//------------------------------------------------------------------------------

//! MPL utility
template <typename T>
struct MakeMIPDense
{
  typedef typename 
  MIPFieldWrapper<Field3D::MIPField<Field3D::DenseField<T> > >::Vec type;
};

//------------------------------------------------------------------------------

//! MPL utility
template <typename T>
struct MakeMIPSparse
{
  typedef typename 
  MIPFieldWrapper<Field3D::MIPField<Field3D::SparseField<T> > >::Vec type;
};

//------------------------------------------------------------------------------

//! MPL utility
template <typename T>
struct MakeMIPTemporal
{
  typedef typename 
  MIPFieldWrapper<Field3D::MIPField<Field3D::TemporalField<T> > >::Vec type;
};

//------------------------------------------------------------------------------

template <int Dims_T>
struct LoadFields;

struct LoadFieldsParams
{
  LoadFieldsParams(Field3D::Field3DInputFile &a_in, 
                   const std::string         &a_name,
                   const std::string         &a_attribute, 
                   Field3D::FieldRes::Vec    &a_results,
                   Field3D::FieldRes::Vec    &a_minResults, 
                   Field3D::FieldRes::Vec    &a_maxResults)
    : in(a_in), 
      name(a_name), 
      attribute(a_attribute), 
      results(a_results),
      minResults(a_minResults), 
      maxResults(a_maxResults)
  { }
  Field3D::Field3DInputFile &in;
  const std::string         &name;
  const std::string         &attribute;
  Field3D::FieldRes::Vec    &results;
  Field3D::FieldRes::Vec    &minResults;
  Field3D::FieldRes::Vec    &maxResults;
};

template <>
struct LoadFields<1>
{
  // Ctor
  LoadFields(LoadFieldsParams &params)
    : m_p(params)
  { }
  // Functor
  template <typename T>
  void operator()(T)
  {
    // Load all fields of type T
    typename Field3D::Field<T>::Vec fields = 
      m_p.in.readScalarLayers<T>(m_p.name, m_p.attribute);
    // Add the fields to the result
    BOOST_FOREACH (const typename Field3D::Field<T>::Ptr &ptr, fields) {
      m_p.results.push_back(ptr);
    }
    // Load 'min' fields
    typename Field3D::Field<T>::Vec minFields = 
      m_p.in.readScalarLayers<T>(m_p.name, m_p.attribute + k_minSuffix);
    // Add the fields to the result
    BOOST_FOREACH (const typename Field3D::Field<T>::Ptr &ptr, minFields) {
      m_p.minResults.push_back(ptr);
    }
    // Load 'max' fields
    typename Field3D::Field<T>::Vec maxFields = 
      m_p.in.readScalarLayers<T>(m_p.name, m_p.attribute + k_maxSuffix);
    // Add the fields to the result
    BOOST_FOREACH (const typename Field3D::Field<T>::Ptr &ptr, maxFields) {
      m_p.maxResults.push_back(ptr);
    }
  }
  // Data members
  LoadFieldsParams &m_p;
};

template <>
struct LoadFields<3>
{
  // Ctor
  LoadFields(LoadFieldsParams &params)
    : m_p(params)
  { }
  // Functor
  template <typename Vec_T>
  void operator()(Vec_T)
  {
    typedef typename Vec_T::BaseType T;

    // Load all fields of type T
    typename Field3D::Field<Vec_T>::Vec fields = 
      m_p.in.readVectorLayers<T>(m_p.name, m_p.attribute);
    // Add the fields to the result
    BOOST_FOREACH (const typename Field3D::Field<Vec_T>::Ptr &ptr, fields) {
      m_p.results.push_back(ptr);
    }
    // Load 'min' fields
    typename Field3D::Field<Vec_T>::Vec minFields = 
      m_p.in.readVectorLayers<T>(m_p.name, m_p.attribute + k_minSuffix);
    // Add the fields to the result
    BOOST_FOREACH (const typename Field3D::Field<Vec_T>::Ptr &ptr, minFields) {
      m_p.minResults.push_back(ptr);
    }
    // Load 'max' fields
    typename Field3D::Field<Vec_T>::Vec maxFields = 
      m_p.in.readVectorLayers<T>(m_p.name, m_p.attribute + k_maxSuffix);
    // Add the fields to the result
    BOOST_FOREACH (const typename Field3D::Field<Vec_T>::Ptr &ptr, maxFields) {
      m_p.maxResults.push_back(ptr);
    }
  }  
  // Data members
  LoadFieldsParams &m_p;
};

//------------------------------------------------------------------------------

inline bool 
intersect(const Ray3d &ray, const Box3d &box, double &outT0, double &outT1)
{
  double tNear = -std::numeric_limits<double>::max();
  double tFar = std::numeric_limits<double>::max();
  const double epsilon = std::numeric_limits<double>::epsilon() * 10.0;
  
  for (size_t dim = 0; dim < 3; ++dim) {
    double t0, t1;
    if (std::abs(ray.dir[dim]) < epsilon) {
      // Ray is parallel, check if inside slab
      if (ray.pos[dim] < box.min[dim] || ray.pos[dim] > box.max[dim]) {
        return false;
      }
    }
    t0 = (box.min[dim] - ray.pos[dim]) / ray.dir[dim];
    t1 = (box.max[dim] - ray.pos[dim]) / ray.dir[dim];
    if (t0 > t1) {
      std::swap(t0, t1);
    }
    tNear = std::max(tNear, t0);
    tFar = std::min(tFar, t1);
    if (tNear > tFar) {
      return false;
    }
    if (tFar < 0.0) {
      return false;
    }
  }
  outT0 = tNear;
  outT1 = tFar;
  return true;
}

//------------------------------------------------------------------------------

} // namespace detail

//------------------------------------------------------------------------------
// FieldGroup
//------------------------------------------------------------------------------

/*! \class FieldGroup
  The FieldGroup is a convenient way to access a collection of heterogeneous 
  fields as one. It will accept any combination of known data structures
  and template types and efficiently evaluates each one with the optimal
  interpolator, etc.
  
  FieldGroup also provides efficient min/max queries:
  If FieldGroup::load() is called, min/max representations of the attributes
  are read from disk, if available. Otherwise, min/max representations can
  be constructed by calling FieldGroup::makeMinMax().

  The class can also be used to provide basic instancing. By calling
  setTransform() prior to setup() and load(), an object transform may be
  applied to each set of fields.
 */

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup
{
  // MPL Typedefs --------------------------------------------------------------
  
  // The list of basic types to support.
  typedef BaseTypeList_T MPLBaseTypes;

  // Instantiate FieldWrapper<Field_T> for each family with each basic type
  typedef typename mpl::transform<
    MPLBaseTypes, 
    detail::MakeDense<ph::_1> >::type       MPLDenseTypes;
  typedef typename mpl::transform<
    MPLBaseTypes, 
    detail::MakeSparse<ph::_1> >::type      MPLSparseTypes;
  typedef typename mpl::transform<
    MPLBaseTypes, 
    detail::MakeTemporal<ph::_1> >::type    MPLTemporalTypes;
  typedef typename mpl::transform<
    MPLBaseTypes, 
    detail::MakeMIPDense<ph::_1> >::type    MPLMIPDenseTypes;
  typedef typename mpl::transform<
    MPLBaseTypes, 
    detail::MakeMIPSparse<ph::_1> >::type   MPLMIPSparseTypes;
  typedef typename mpl::transform<
    MPLBaseTypes, 
    detail::MakeMIPTemporal<ph::_1> >::type MPLMIPTemporalTypes;

  // Map MPL types to boost fusion types
  typedef typename f_ro::as_vector<MPLDenseTypes>::type       DenseTypes;
  typedef typename f_ro::as_vector<MPLSparseTypes>::type      SparseTypes;
  typedef typename f_ro::as_vector<MPLTemporalTypes>::type    TemporalTypes;
  typedef typename f_ro::as_vector<MPLMIPDenseTypes>::type    MIPDenseTypes;
  typedef typename f_ro::as_vector<MPLMIPSparseTypes>::type   MIPSparseTypes;
  typedef typename f_ro::as_vector<MPLMIPTemporalTypes>::type MIPTemporalTypes;

  // Typedefs ------------------------------------------------------------------

  typedef FieldRes::Vec FieldsVec;

  // Enums ---------------------------------------------------------------------
  
  enum CompositeOp
  {
    Add = 0,
    Avg
  };

  // Constants -----------------------------------------------------------------

  //! Used by load() to indicate missing file
  static const int k_missingFile = -1;

  // Ctors ---------------------------------------------------------------------

  //! Default constructor, does nothing
  FieldGroup();
  //! Construct from a set of fields
  FieldGroup(const Field3D::FieldRes::Vec &fields);

  // Main methods --------------------------------------------------------------

  //! Adds a single field to the group
  void             setup(const Field3D::FieldRes::Ptr field);
  //! Initializes the FieldGroup from a set of fields.
  void             setup(const Field3D::FieldRes::Vec &fields);
  //! Initializes the FieldGroup from a set of fields with pre-computed
  //! min/max representations
  void             setup(const Field3D::FieldRes::Vec &fields,
                         const Field3D::FieldRes::Vec &minFields,
                         const Field3D::FieldRes::Vec &maxFields);
  //! Loads all fields from a given file and optional attribute pattern
  //! \returns Number of fields loaded, or a negative number if 
  //! the file failed to open.
  int              load(const std::string &filename, 
                        const std::string &attribute);
  //! Make min/max representations of the fields in the group
  void             makeMinMax(const float resMult);
  //! The number of fields in the group
  size_t           size() const;
  //! The number of MIP fields in the group
  size_t           sizeMIP() const;
  //! The number of temporal fields in the group
  size_t           sizeTemporal() const;

  // Sampling methods ----------------------------------------------------------

  //! Unified sampling of the group's fields. Will handle both MIP and non-MIP
  //! data with optional compositing functor
  void             sample(const V3d &wsP, 
                          const float wsSpotSize, 
                          const float time,
                          float *result, 
                          const CompositeOp compOp = Add) const;
  //! Unified sampling of the group's fields. Will handle both MIP and non-MIP
  //! data with optional compositing functor
  void             sample(const size_t n, 
                          const float *wsP, 
                          const float *wsSpotSize, 
                          const float *time, 
                          float *result, 
                          const float *active = NULL,
                          const CompositeOp compOp = Add) const;
  //! Unified sampling of all fields using stochastic interpolation
  void             sampleStochastic(const size_t n, 
                                    const float *wsP, 
                                    const float *wsSpotSize, 
                                    const float *time, 
                                    const float *xiX, 
                                    const float *xiY, 
                                    const float *xiZ,
                                    const float *xiSpotSize, 
                                    const float *xiTime, 
                                    float *result, 
                                    const float *active = NULL,
                                    const CompositeOp compOp = Add) const;

  // Deprecated sampling methods -----------------------------------------------

#if 1

  //! Samples the group of fields at the given point. This call will not
  //! include MIP fields, which require a spot diameter.
  //! \warning To be deprecated in favor of sample(wsP, wsSpotSize, time, ...)
  void             sample(const V3d &vsP, 
                          float *result, 
                          bool isVs) const;
  //! Samples all the MIP fields in the group at the given point and
  //! spot diameter.
  //! \warning To be deprecated in favor of sample(wsP, wsSpotSize, time, ...)
  void             sampleMIP(const V3d &vsP, 
                             const float wsSpotSize, 
                             float *result, 
                             bool isVs) const;
  //! Samples the fields in the group.
  //! \warning To be deprecated in favor of sample(wsP, wsSpotSize, time, ...)
  void             sampleMultiple(const size_t n, 
                                  const float *wsP, 
                                  float *result,
                                  const float *active = NULL) const;
  //! Samples all the MIP fields in the group.
  //! \warning To be deprecated in favor of sample(wsP, wsSpotSize, time, ...)
  void             sampleMIPMultiple(const size_t n, 
                                     const float *wsP, 
                                     const float *wsSpotSize, 
                                     float *result, 
                                     const float *active = NULL) const;

#endif

  // Info methods --------------------------------------------------------------

  //! Returns the bounds of the group
  Box3d            wsBounds() const;
  //! Whether the given point intersects any of the fields in the FieldGroup
  bool             intersects(const V3d &wsP) const;
  //! Whether the given point intersects any of the fields in the FieldGroup
  void             intersectsMultiple(const size_t n, 
                                      const float *wsP, 
                                      bool *result) const;
  //! Gets the intersection intervals between the ray and the fields
  bool             getIntersections(const Ray3d &ray, 
                                    IntervalVec &intervals) const;
  //! Returns the min/max range within a given bounding box.
  void             getMinMax(const Box3d &wsBounds, 
                             float *min, 
                             float *max) const;
  //! Smallest voxel size 
  float            minWsVoxelSize() const
  { return m_minWsVoxelSize; }
  //! Whether the FieldGroup has a pre-filtered min/max representation
  bool             hasPrefiltMinMax() const
  { return m_hasPrefiltMinMax; }
  //! Returns the memory use in bytes for the fields in the group
  long long int    memSize() const;
  //! Returns a vector of FieldRes::Ptrs to the fields in the group
  const FieldsVec& fields() const
  { return m_allFields; }

  // Per-instance methods ------------------------------------------------------

  //! Sets the current object to world transform. This will be used for
  //! subsequent setup() and load() calls. Primarily used when the FieldGroup
  //! is employed for instancing of multiple fields.
  void             setOsToWs(const Imath::M44d &osToWs);
  //! Enable world axis aligned bounding box in lookups. This will be
  //! used for subsequent setup() and load() calls. Primarily used
  //! when the FieldGroup is employed for instancing of multiple
  //! fields.
  void             setWsBoundsOptimization(const bool doWsBoundsOptimization);
  //! Sets the current ValueRemap operator. This will be used for
  //! subsequent setup() and load() calls. Primarily used when the FieldGroup
  //! is employed for instancing of multiple fields. By default, no value
  //! remapping takes place.
  //! \note It is ok to pass in a null pointer to disable value remapping.
  void             setValueRemapOp(ValueRemapOp::Ptr op);

protected:

  // Utility methods -----------------------------------------------------------

  //! Set up the min/max MIP representations
  void setupMinMax(const FieldRes::Vec &minFields,
                   const FieldRes::Vec &maxFields);

  // Data members --------------------------------------------------------------
  
  DenseTypes       m_dense;
  SparseTypes      m_sparse;
  TemporalTypes    m_temporal;
  MIPDenseTypes    m_mipDense, m_mipDenseMin, m_mipDenseMax;
  MIPSparseTypes   m_mipSparse, m_mipSparseMin, m_mipSparseMax;
  MIPTemporalTypes m_mipTemporal;

  //! Whether pre-filtered min/max are present
  bool m_hasPrefiltMinMax;

  //! Cached min voxel size
  float m_minWsVoxelSize;

  //! Current object to world transform
  M44d m_osToWs;

  //! Enable world space bounds optimization
  bool m_doWsBoundsOptimization;

  //! Current value remap op. Defaults to null pointer
  ValueRemapOp::Ptr m_valueRemapOp;

  //! Stores all the fields owned by the FieldGroup
  FieldRes::Vec  m_allFields;
  //! Stores all the auxiliary fields owned by the FieldGroup
  FieldRes::Vec  m_auxFields;
  
  // Functors ------------------------------------------------------------------

  struct GrabFields;
  struct DoWsBoundsOptimization;
  struct CountFields;
  struct MakeMinMax;
  struct MakeMinMaxMIP;
  struct Sample;
  struct SampleMIP;
  struct SampleMultiple;
  struct SampleMIPMultiple;
  struct SampleStochastic;
  struct SampleMIPStochastic;
  struct SampleTemporal;
  struct SampleMIPTemporal;
  struct SampleTemporalMultiple;
  struct SampleMIPTemporalMultiple;
  struct SampleTemporalStochastic;
  struct SampleMIPTemporalStochastic;
  struct GetWsBounds;
  struct GetIntersections;
  struct GetMinMax;
  struct GetMinMaxMIP;
  struct GetMinMaxPrefilt;
  struct GetMinMaxTemporal;
  struct GetMinMaxMIPTemporal;
  struct MinWsVoxelSize;
  struct PointIsect;
  struct PointIsectMultiple;
  struct MemSize;

};

//------------------------------------------------------------------------------

typedef FieldGroup<ScalarTypes, 1> ScalarFieldGroup;
typedef FieldGroup<VectorTypes, 3> VectorFieldGroup;

//------------------------------------------------------------------------------
// Template implementations
//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
FieldGroup<BaseTypeList_T, Dims_T>::FieldGroup()
  : m_hasPrefiltMinMax(false), 
    m_minWsVoxelSize(std::numeric_limits<float>::max()),
    m_doWsBoundsOptimization(false)
{ }

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
FieldGroup<BaseTypeList_T, Dims_T>::FieldGroup
(const Field3D::FieldRes::Vec &fields)
  : m_hasPrefiltMinMax(false), 
    m_minWsVoxelSize(std::numeric_limits<float>::max()),
    m_doWsBoundsOptimization(false)
{
  // Perform setup
  setup(fields);
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::setOsToWs(const Imath::M44d &osToWs)
{
  m_osToWs = osToWs;
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::setWsBoundsOptimization(const bool enabled)
{
  m_doWsBoundsOptimization = enabled;

  if (m_doWsBoundsOptimization) {
    DoWsBoundsOptimization op(m_doWsBoundsOptimization);
    fusion::for_each(m_dense, op);
    fusion::for_each(m_sparse, op);
    fusion::for_each(m_temporal, op);
    fusion::for_each(m_mipDense, op);
    fusion::for_each(m_mipSparse, op);
    fusion::for_each(m_mipTemporal, op);
  }
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::setValueRemapOp(ValueRemapOp::Ptr op)
{
  m_valueRemapOp = op;
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::setup(const Field3D::FieldRes::Ptr field)
{
  FieldRes::Vec fields, minFields, maxFields;
  fields.push_back(field);
  // Perform setup
  setup(fields, minFields, maxFields);
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::setup(const Field3D::FieldRes::Vec &fields)
{
  FieldRes::Vec minFields, maxFields;
  // Perform setup
  setup(fields, minFields, maxFields);
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::setup
(const Field3D::FieldRes::Vec &fields,
 const Field3D::FieldRes::Vec &minFields,
 const Field3D::FieldRes::Vec &maxFields)
{
  // Record fields in m_allFields
  m_allFields = fields;

  // Pick out primary fields
  for (size_t i = 0, end = fields.size(); i < end; ++i) {
    GrabFields op(fields[i], m_osToWs, m_valueRemapOp, 
                  m_doWsBoundsOptimization);
    fusion::for_each(m_dense, op);
    fusion::for_each(m_sparse, op);
    fusion::for_each(m_mipDense, op);
    fusion::for_each(m_mipSparse, op);
    fusion::for_each(m_temporal, op);
    fusion::for_each(m_mipTemporal, op);
  }

  // Get min voxel size ---

  MinWsVoxelSize op(m_minWsVoxelSize);

  fusion::for_each(this->m_dense, op);
  fusion::for_each(this->m_sparse, op);
  fusion::for_each(this->m_mipDense, op);
  fusion::for_each(this->m_mipSparse, op);    
  fusion::for_each(m_temporal, op);
  fusion::for_each(m_mipTemporal, op);

  // Pick out min/max fields
  setupMinMax(minFields, maxFields);
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void
FieldGroup<BaseTypeList_T, Dims_T>::setupMinMax
(const Field3D::FieldRes::Vec &minFields,
 const Field3D::FieldRes::Vec &maxFields)
{
  // Record minFields and maxFields as auxiliary fields
  m_auxFields.insert(m_auxFields.end(), minFields.begin(), minFields.end());
  m_auxFields.insert(m_auxFields.end(), maxFields.begin(), maxFields.end());

  // Pick out min fields
  for (size_t i = 0, end = minFields.size(); i < end; ++i) {
    GrabFields op(minFields[i], m_osToWs, m_valueRemapOp, 
                  m_doWsBoundsOptimization);
    fusion::for_each(m_mipDenseMin, op);
    fusion::for_each(m_mipSparseMin, op);
  }
  // Pick out max fields
  for (size_t i = 0, end = maxFields.size(); i < end; ++i) {
    GrabFields op(maxFields[i], m_osToWs, m_valueRemapOp, 
                  m_doWsBoundsOptimization);
    fusion::for_each(m_mipDenseMax, op);
    fusion::for_each(m_mipSparseMax, op);
  }
  // Check if we have pre-filtered fields
  CountFields countMinOp, countMaxOp;
  fusion::for_each(m_mipDenseMin, countMinOp);
  fusion::for_each(m_mipDenseMax, countMaxOp);
  fusion::for_each(m_mipSparseMin, countMinOp);
  fusion::for_each(m_mipSparseMax, countMaxOp);
  if (countMinOp.count > 0 && countMaxOp.count > 0) {
    m_hasPrefiltMinMax = true;
  }
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
int
FieldGroup<BaseTypeList_T, Dims_T>::load
(const std::string &filename, const std::string &attribute)
{
  using namespace Field3D;

  // Storage for the primary fields
  FieldRes::Vec results;
  // Storage for the auxiliary fields
  FieldRes::Vec minResults, maxResults;

  // Track number of fields in group before loading.
  const size_t sizeBeforeLoading = size();

  // Open each file ---

  std::vector<std::string> filenames;
  filenames.push_back(filename);

  BOOST_FOREACH (const std::string fn, filenames) {

    Field3DInputFile in;
    if (!in.open(fn)) {
      return k_missingFile;
    }

    // Use partition names to determine if fields should be loaded
    std::vector<std::string> names;
    in.getPartitionNames(names);

    BOOST_FOREACH (const std::string &name, names) {
      detail::LoadFieldsParams params(in, name, attribute, results, 
                                      minResults, maxResults);
      detail::LoadFields<Dims_T> op(params);
      mpl::for_each<BaseTypeList_T>(op);
    }

  }

  // Set up from fields
  setup(results, minResults, maxResults);

  // Done. Return the number of fields that were loaded.
  return size() - sizeBeforeLoading;
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void
FieldGroup<BaseTypeList_T, Dims_T>::makeMinMax
(const float resMult)
{
  // Storage for the auxiliary fields
  FieldRes::Vec minFields, maxFields;

  MakeMinMax op(minFields, maxFields, resMult);
  fusion::for_each(m_dense, op);
  fusion::for_each(m_sparse, op);

  MakeMinMaxMIP opMIP(minFields, maxFields, resMult);
  fusion::for_each(m_mipDense, opMIP);
  fusion::for_each(m_mipSparse, opMIP);

  setupMinMax(minFields, maxFields);
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
size_t 
FieldGroup<BaseTypeList_T, Dims_T>::size() const
{
  CountFields op;
  fusion::for_each(m_dense, op);
  fusion::for_each(m_sparse, op);
  fusion::for_each(m_mipDense, op);
  fusion::for_each(m_mipSparse, op);
  fusion::for_each(m_temporal, op);
  fusion::for_each(m_mipTemporal, op);
  return op.count;
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
size_t 
FieldGroup<BaseTypeList_T, Dims_T>::sizeMIP() const
{
  CountFields op;
  fusion::for_each(m_mipDense, op);
  fusion::for_each(m_mipSparse, op);
  fusion::for_each(m_mipTemporal, op);
  return op.count;
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
size_t 
FieldGroup<BaseTypeList_T, Dims_T>::sizeTemporal() const
{
  CountFields op;
  fusion::for_each(m_temporal, op);
  return op.count;
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::sample(const V3d &wsP, 
                                           const float wsSpotSize, 
                                           const float time,
                                           float *result, 
                                           const CompositeOp compOp) const
{
  size_t numHits = 0;

  // Handle ordinary fields
  Sample op(wsP, result, numHits);
  fusion::for_each(m_dense, op);
  fusion::for_each(m_sparse, op);

  // Handle MIP fields
  SampleMIP mipOp(wsP, wsSpotSize, result, numHits);
  fusion::for_each(m_mipDense, mipOp);
  fusion::for_each(m_mipSparse, mipOp);

  // Handle Temporal fields
  SampleTemporal temporalOp(wsP, time, result, numHits);
  fusion::for_each(m_temporal, temporalOp);

  // Handle Temporal MIP fields
  SampleMIPTemporal mipTemporalOp(wsP, wsSpotSize, time, result, numHits);
  fusion::for_each(m_mipTemporal, mipTemporalOp);

  // Check composite op
  if (compOp == Add) {
    // Nothing
  } else {
    if (numHits > 1) {
      for (size_t i = 0; i < Dims_T; ++i) {
        result[i] /= static_cast<float>(numHits);
      }
    }
  }
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void FieldGroup<BaseTypeList_T, Dims_T>::
sample(const size_t n, 
       const float *wsP, 
       const float *wsSpotSize, 
       const float *time, 
       float *result, 
       const float *active,
       const CompositeOp compOp) const
{
  // Initialize numHits array
  size_t numHits[n];
  std::fill_n(numHits, n, 0);

  // Handle ordinary fields
  SampleMultiple op(n, wsP, result, numHits, active);
  fusion::for_each(m_dense, op);
  fusion::for_each(m_sparse, op);

  // Handle MIP fields
  SampleMIPMultiple mipOp(n, wsP, wsSpotSize, result, numHits, active);
  fusion::for_each(m_mipDense, mipOp);
  fusion::for_each(m_mipSparse, mipOp);

  // Handle Temporal fields
  SampleTemporalMultiple temporalOp(n, wsP, time, result, numHits, active);
  fusion::for_each(m_temporal, temporalOp);

  // Handle Temporal MIP fields
  SampleMIPTemporalMultiple mipTemporalOp(n, wsP, wsSpotSize, time, 
                                          result, numHits, active);
  fusion::for_each(m_temporal, mipTemporalOp);

  // Check composite op
  if (compOp == Add) {
    // Nothing
  } else {
    for (int i = 0; i < n; ++i) {
      if (numHits[i]) {
        for (size_t c = 0; c < Dims_T; ++c) {
          result[i * Dims_T + c] /= static_cast<float>(numHits[i]);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void FieldGroup<BaseTypeList_T, Dims_T>::
sampleStochastic(const size_t n, 
                 const float *wsP, 
                 const float *wsSpotSize, 
                 const float *time, 
                 const float *xiX, 
                 const float *xiY, 
                 const float *xiZ,
                 const float *xiSpotSize, 
                 const float *xiTime, 
                 float *result, 
                 const float *active,
                 const CompositeOp compOp) const
{
  // Initialize numHits array
  size_t numHits[n];
  std::fill_n(numHits, n, 0);

  // Handle ordinary fields
  if (xiX && xiY && xiZ) {
    SampleStochastic op(n, SampleStochasticArgs(wsP, xiX, xiY, xiZ, result, 
                                                numHits, active));
    fusion::for_each(m_dense, op);
    fusion::for_each(m_sparse, op);
  } else {
    SampleMultiple op(n, wsP, result, numHits, active);
    fusion::for_each(m_dense, op);
    fusion::for_each(m_sparse, op);
  }

  // Handle MIP fields
  //   ... SampleMIPStochastic works two ways - either stochastic in
  //   ... all dimensions or only in LOD choice.
  if ((xiSpotSize && !xiX && !xiY && !xiZ) || 
      (xiSpotSize && xiX && xiY && xiZ)) {
    SampleMIPStochastic mipOp(n, SampleMIPStochasticArgs(wsP, wsSpotSize, 
                                                         xiX, xiY, xiZ, 
                                                         xiSpotSize, result, 
                                                         numHits, active));
    fusion::for_each(m_mipDense, mipOp);
    fusion::for_each(m_mipSparse, mipOp);
  } else {
    SampleMIPMultiple mipOp(n, wsP, wsSpotSize, result, numHits, active);
    fusion::for_each(m_mipDense, mipOp);
    fusion::for_each(m_mipSparse, mipOp);
  }

  // Handle temporal fields
  if (xiTime && xiSpotSize && xiX && xiY && xiZ) {
    SampleTemporalStochasticArgs tArgs(wsP, wsSpotSize, time, xiX, xiY, xiZ,
                                       xiSpotSize, xiTime, result, numHits, 
                                       active);
    SampleTemporalStochastic temporalOp(n, tArgs);
    fusion::for_each(m_temporal, temporalOp);
    SampleMIPTemporalStochastic mipTemporalOp(n, tArgs);
    fusion::for_each(m_temporal, mipTemporalOp);
  } else {
    SampleMIPTemporalMultiple mipTemporalOp(n, wsP, time, result, active);
    fusion::for_each(m_temporal, mipTemporalOp);
  }

  // Check composite op
  if (compOp == Add) {
    // Nothing
  } else {
    for (int i = 0; i < n; ++i) {
      if (numHits[i]) {
        for (size_t c = 0; c < Dims_T; ++c) {
          result[i * Dims_T + c] /= static_cast<float>(numHits[i]);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------

#if 1

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::sample(const V3d &vsP, 
                                           float *result, 
                                           bool /* isVs */) const
{
  size_t numHits = 0;

  Sample op(vsP, result, numHits);
  fusion::for_each(m_dense, op);
  fusion::for_each(m_sparse, op);
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void
FieldGroup<BaseTypeList_T, Dims_T>::sampleMultiple(const size_t n,
                                                   const float *wsP,
                                                   float *result,
                                                   const float *active) const
{
  size_t numHits[n];
  std::fill_n(numHits, n, 0);

  SampleMultiple op(n, wsP, result, numHits, active);
  fusion::for_each(m_dense, op);
  fusion::for_each(m_sparse, op);
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::sampleMIP(const V3d &vsP, 
                                              const float wsSpotSize, 
                                              float *result, 
                                              bool /* isVs */) const
{
  size_t numHits = 0;

  SampleMIP op(vsP, wsSpotSize, result, numHits);
  fusion::for_each(m_mipDense, op);
  fusion::for_each(m_mipSparse, op);
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void
FieldGroup<BaseTypeList_T, Dims_T>::sampleMIPMultiple(const size_t n,
                                                      const float *wsP,
                                                      const float *wsSpotSize, 
                                                      float *result,
                                                      const float *active) const
{
  size_t numHits[n];
  std::fill_n(numHits, n, 0);

  SampleMIPMultiple op(n, wsP, wsSpotSize, result, numHits, active);
  fusion::for_each(m_mipDense, op);
  fusion::for_each(m_mipSparse, op);
}

#endif

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
Box3d 
FieldGroup<BaseTypeList_T, Dims_T>::wsBounds() const
{
  Box3d wsBounds;
  GetWsBounds op(wsBounds);
  fusion::for_each(m_dense, op);
  fusion::for_each(m_sparse, op);
  fusion::for_each(m_mipDense, op);
  fusion::for_each(m_mipSparse, op);
  fusion::for_each(m_temporal, op);
  fusion::for_each(m_mipTemporal, op);
  return wsBounds;
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
bool
FieldGroup<BaseTypeList_T, Dims_T>::intersects(const V3d &wsP) const
{
  bool doesIntersect = false;

  PointIsect op(wsP, doesIntersect);
  fusion::for_each(m_dense, op);
  fusion::for_each(m_sparse, op);
  fusion::for_each(m_mipDense, op);
  fusion::for_each(m_mipSparse, op);
  fusion::for_each(m_temporal, op);
  fusion::for_each(m_mipTemporal, op);
  return op.result();
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void
FieldGroup<BaseTypeList_T, Dims_T>::intersectsMultiple(const size_t n,
                                                       const float *wsP,
                                                       bool *result) const

{
  PointIsectMultiple op(n, wsP, result);
  fusion::for_each(m_dense, op);
  fusion::for_each(m_sparse, op);
  fusion::for_each(m_mipDense, op);
  fusion::for_each(m_mipSparse, op);
  fusion::for_each(m_temporal, op);
  fusion::for_each(m_mipTemporal, op);
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
bool 
FieldGroup<BaseTypeList_T, Dims_T>::getIntersections
(const Ray3d &ray, IntervalVec &intervals) const
{
  GetIntersections op(ray, intervals);
  fusion::for_each(m_dense, op);
  fusion::for_each(m_sparse, op);
  fusion::for_each(m_mipDense, op);
  fusion::for_each(m_mipSparse, op);
  fusion::for_each(m_temporal, op);
  fusion::for_each(m_mipTemporal, op);
  return intervals.size() > 0;
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::getMinMax(const Box3d &wsBounds, 
                                              float *min, 
                                              float *max) const
{
  // Check whether query region is large enough to warrant using prefiltered
  // minmax data
#if 1
  const double mult       = 2.0;
  const double querySize  = wsBounds.size().length();
  const bool   usePrefilt = querySize > this->m_minWsVoxelSize * mult;
#else
  const bool   usePrefilt = true;
#endif

  if (usePrefilt && m_hasPrefiltMinMax) {
    // Pre-filtered types
    GetMinMaxPrefilt opMin(wsBounds, min, GetMinMaxPrefilt::Min);
    GetMinMaxPrefilt opMax(wsBounds, max, GetMinMaxPrefilt::Max);
    fusion::for_each(m_mipDenseMin, opMin);
    fusion::for_each(m_mipSparseMin, opMin);    
    fusion::for_each(m_mipDenseMax, opMax);
    fusion::for_each(m_mipSparseMax, opMax);
  } else {
    // Non-prefiltered types
    GetMinMax op(wsBounds, min, max);
    fusion::for_each(m_dense, op);
    fusion::for_each(m_sparse, op);
    // Non-prefiltered MIP types
    GetMinMaxMIP opMIP(wsBounds, min, max);
    fusion::for_each(m_mipDense, opMIP);
    fusion::for_each(m_mipSparse, opMIP);
    // Non-prefiltered Temporal types
    GetMinMaxTemporal opTemporal(wsBounds, min, max);
    fusion::for_each(m_temporal, opTemporal);
    // Non-prefiltered Temporal MIP types
    GetMinMaxMIPTemporal opMipTemporal(wsBounds, min, max);
    fusion::for_each(m_mipTemporal, opMipTemporal);
  }
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
long long int
FieldGroup<BaseTypeList_T, Dims_T>::memSize() const
{
  long long int result = 0;
  MemSize op(result);
  fusion::for_each(m_dense, op);
  fusion::for_each(m_sparse, op);
  fusion::for_each(m_mipDense, op);
  fusion::for_each(m_mipSparse, op);
  fusion::for_each(m_temporal, op);
  fusion::for_each(m_mipTemporal, op);
  return result;
}

//------------------------------------------------------------------------------
// Functor implementations 
//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::GrabFields
{
  //! Ctor
  GrabFields(Field3D::FieldRes::Ptr f, 
             const M44d &osToWs,
             ValueRemapOp::Ptr op,
             const bool doWsBoundsOptimization)
    : m_field(f), m_osToWs(osToWs), m_op(op),
      m_doWsBoundsOptimization(doWsBoundsOptimization)
  { }
  //! Functor
  template <typename WrapperVec_T>
  void operator()(WrapperVec_T &vec) const
  { 
    // Typedefs
    typedef typename WrapperVec_T::value_type Wrapper_T;
    typedef typename Wrapper_T::field_type    Field_T;
    typedef typename Field_T::Ptr             FieldPtr;
    // Grab field if type matches
    if (FieldPtr f = 
        Field3D::field_dynamic_cast<Field_T>(m_field)) {
      // Add to FieldWrapper vector
      vec.push_back(f);
      // Grab just-inserted entry
      Wrapper_T &entry = vec.back();
      // Set up transform
      M44d id;
      if (m_osToWs != id) {
        entry.setOsToWs(m_osToWs);
      }

      // Set toggle to use world axis aligned bounding boxes in
      // lookups
      if (m_doWsBoundsOptimization) {
        entry.setWsBoundsOptimization(m_doWsBoundsOptimization);
      }

      // Set up value remap op
      if (m_op) {
        entry.setValueRemapOp(m_op);
      }
    }
  }
private:
  //! The field to work on. Will be matched against the type of operator().
  Field3D::FieldRes::Ptr m_field;
  //! Object to world transform
  M44d                   m_osToWs;
  //! Value remap operator
  ValueRemapOp::Ptr      m_op;
  //! Enable world space bounds optimization
  bool                   m_doWsBoundsOptimization;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::DoWsBoundsOptimization
{
  //! Ctor
  DoWsBoundsOptimization(const bool doWsBoundsOptimization)
    : m_doWsBoundsOptimization(doWsBoundsOptimization)
  { }
  //! Functor
  template <typename WrapperVec_T>
  void operator()(WrapperVec_T &vec) const
  {
    for (size_t i = 0, end = vec.size(); i < end; ++i) {
      vec[i].setWsBoundsOptimization(m_doWsBoundsOptimization);
    }
  }
  //! Enable world space bounds optimization
  bool                   m_doWsBoundsOptimization;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::CountFields
{
  //! Ctor
  CountFields()
    : count(0)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { count += vec.size(); }
  // Data members
  mutable int count;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::MakeMinMax
{
  //! Ctor
  MakeMinMax(Field3D::FieldRes::Vec &minFields,
             Field3D::FieldRes::Vec &maxFields,
             const float resMult)
    : m_minFields(minFields),
      m_maxFields(maxFields),
      m_resMult(resMult),
      m_numThreads(Field3D::numIOThreads())
  { }
  //! Functor
  template <typename WrapperVec_T>
  void operator()(const WrapperVec_T &vec)
  {
    // Typedefs
    typedef typename WrapperVec_T::value_type     Wrapper_T;
    typedef typename Wrapper_T::field_type        Field_T;
    typedef typename Field3D::MIPField<Field_T>   MIPField_T;
    typedef typename Field_T::value_type          Value_T;
    typedef typename Field3D::Field<Value_T>::Ptr FieldPtr;

    std::pair<FieldPtr, FieldPtr> result;
    for (size_t i = 0, end = vec.size(); i < end; ++i) {
      const Field_T &f = *(vec[i].field);
      result = Field3D::makeMinMax<MIPField_T>(f, m_resMult, m_numThreads);
      m_minFields.push_back(result.first);
      m_maxFields.push_back(result.second);
    }
  }
  // Data members
  Field3D::FieldRes::Vec &m_minFields;
  Field3D::FieldRes::Vec &m_maxFields;
  const float m_resMult;
  const size_t m_numThreads;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::MakeMinMaxMIP
{
  //! Ctor
  MakeMinMaxMIP(Field3D::FieldRes::Vec &minFields,
                Field3D::FieldRes::Vec &maxFields,
                const float resMult)
    : m_minFields(minFields),
      m_maxFields(maxFields),
      m_resMult(resMult),
      m_numThreads(Field3D::numIOThreads())
  { }
  //! Functor
  template <typename WrapperVec_T>
  void operator()(const WrapperVec_T &vec)
  {
    // Typedefs
    typedef typename WrapperVec_T::value_type     Wrapper_T;
    typedef typename Wrapper_T::field_type        MIPField_T;
    typedef typename MIPField_T::NestedType       Field_T;
    typedef typename Field_T::value_type          Value_T;
    typedef typename Field3D::Field<Value_T>::Ptr FieldPtr;

    std::pair<FieldPtr, FieldPtr> result;
    for (size_t i = 0, end = vec.size(); i < end; ++i) {
      const Field_T &f = *(vec[i].field->concreteMipLevel(0));
      result = Field3D::makeMinMax<MIPField_T>(f, m_resMult, m_numThreads);
      m_minFields.push_back(result.first);
      m_maxFields.push_back(result.second);
    }
  }
  // Data members
  Field3D::FieldRes::Vec &m_minFields;
  Field3D::FieldRes::Vec &m_maxFields;
  const float m_resMult;
  const size_t m_numThreads;
};
  
//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::Sample
{
  //! Ctor
  Sample(const V3d &p, float *result, size_t &numHits)
    : m_p(p), m_result(result), m_numHits(numHits)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sample(vec, m_p, m_result, m_numHits); 
  }
  // Data members
  const V3d &m_p;
  float     *m_result;
  size_t    &m_numHits;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::SampleMIP
{
  //! Ctor
  SampleMIP(const V3d &p, const float wsSpotSize, float *result, 
            size_t &numHits)
    : m_p(p), m_wsSpotSize(wsSpotSize), m_result(result), m_numHits(numHits)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sampleMIP(vec, m_p, m_wsSpotSize, m_result, 
                                       m_numHits); 
  }
  // Data members
  const V3d   &m_p;
  const float  m_wsSpotSize;
  float       *m_result;
  size_t      &m_numHits;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::SampleMultiple
{
  //! Ctor
  SampleMultiple(const size_t n, const float *p, float *result,
                 size_t *numHits, const float *active = NULL)
    : m_n(n), m_p(p), m_result(result), m_numHits(numHits), m_active(active)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sampleMultiple(vec, m_n, m_p, m_result, 
                                            m_numHits, m_active); 
  }
  // Data members
  const int    m_n;
  const float *m_p;
  float       *m_result;
  size_t      *m_numHits;
  const float *m_active;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::SampleMIPMultiple
{
  //! Ctor
  SampleMIPMultiple(const size_t n, const float *p, const float *wsSpotSize,
                    float *result, size_t *numHits, const float *active = NULL)
    : m_n(n), m_p(p), m_wsSpotSize(wsSpotSize), m_result(result),
      m_numHits(numHits), m_active(active)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sampleMIPMultiple(vec, m_n, m_p, m_wsSpotSize,
                                               m_result, m_numHits, m_active); 
  }
  // Data members
  const int    m_n;
  const float *m_p;
  const float *m_wsSpotSize;
  float       *m_result;
  size_t      *m_numHits;
  const float *m_active;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::SampleStochastic
{
  //! Ctor
  SampleStochastic(const size_t n, const SampleStochasticArgs &args)
    : m_n(n), m_args(args)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sampleStochastic(vec, m_n, m_args);
  }
  // Data members
  const int                   m_n;
  const SampleStochasticArgs &m_args;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::SampleMIPStochastic
{
  //! Ctor
  SampleMIPStochastic(const size_t n, const SampleMIPStochasticArgs &args)
    : m_n(n), m_args(args)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sampleMIPStochastic(vec, m_n, m_args);
  }
  // Data members
  const int                      m_n;
  const SampleMIPStochasticArgs &m_args;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::SampleTemporal
{
  //! Ctor
  SampleTemporal(const V3d &p, const float t, float *result, size_t &numHits)
    : m_p(p), m_t(t), m_result(result), m_numHits(numHits)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sampleTemporal(vec, m_p, m_t, m_result, m_numHits);
  }
  // Data members
  const V3d &m_p;
  const float m_t;
  float *m_result;
  size_t &m_numHits;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::SampleMIPTemporal
{
  //! Ctor
  SampleMIPTemporal(const V3d &p, const float wsSpotSize, const float t, 
                    float *result, size_t &numHits)
    : m_p(p), m_wsSpotSize(wsSpotSize), m_t(t), 
      m_result(result), m_numHits(numHits)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sampleMIPTemporal(vec, m_p, m_wsSpotSize, m_t, 
                                               m_result, m_numHits);
  }
  // Data members
  const V3d &m_p;
  const float m_wsSpotSize;
  const float m_t;
  float *m_result;
  size_t &m_numHits;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::SampleTemporalMultiple
{
  //! Ctor
  SampleTemporalMultiple(const size_t n,
                         const float *p,
                         const float *t,
                         float *result,
                         size_t *numHits, 
                         const float *active)
    : m_n(n), m_p(p), m_t(t), m_result(result), m_numHits(numHits),
      m_active(active)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sampleTemporalMultiple(vec, m_n, m_p, m_t,
                                                    m_result, m_numHits, 
                                                    m_active); 
  }
  // Data members
  const size_t m_n;
  const float *m_p;
  const float *m_t;
  float *m_result;
  size_t *m_numHits;
  const float *m_active;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::SampleMIPTemporalMultiple
{
  //! Ctor
  SampleMIPTemporalMultiple(const size_t n,
                            const float *p,
                            const float *t,
                            float *result,
                            size_t *numHits, 
                            const float *active)
    : m_n(n), m_p(p), m_t(t), m_result(result), m_numHits(numHits),
      m_active(active)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sampleMIPTemporalMultiple(vec, m_n, m_p, m_t,
                                                       m_result, m_numHits, 
                                                       m_active); 
  }
  // Data members
  const size_t m_n;
  const float *m_p;
  const float *m_t;
  float *m_result;
  size_t *m_numHits;
  const float *m_active;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::SampleTemporalStochastic
{
  //! Ctor
  SampleTemporalStochastic(const size_t n, 
                           const SampleTemporalStochasticArgs &args)
    : m_n(n), m_args(args)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sampleTemporalStochastic(vec, m_n, m_args);
  }
  // Data members
  const int                           m_n;
  const SampleTemporalStochasticArgs &m_args;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::SampleMIPTemporalStochastic
{
  //! Ctor
  SampleMIPTemporalStochastic(const size_t n, 
                              const SampleTemporalStochasticArgs &args)
    : m_n(n), m_args(args)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sampleMIPTemporalStochastic(vec, m_n, m_args);
  }
  // Data members
  const int                           m_n;
  const SampleTemporalStochasticArgs &m_args;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::GetWsBounds
{
  //! Ctor
  GetWsBounds(Box3d &wsBounds)
    : m_wsBounds(wsBounds)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    for (size_t field = 0, end = vec.size(); field < end; ++field) {
      // Pointer to mapping
      const FieldMapping* mapping = vec[field].mapping;
      if (mapping) {
        // Corner vertices in local space
        std::vector<V3d> lsP = unitCornerPoints();
        // Transform to world space and pad resulting bounds
        for (size_t i = 0; i < 8; ++i) {
          V3d wsP;
          if (vec[field].doOsToWs) {
            V3d osP;
            mapping->localToWorld(lsP[i], osP);
            vec[field].osToWs.multVecMatrix(osP, wsP);
          } else {
            mapping->localToWorld(lsP[i], wsP);
          }
          m_wsBounds.extendBy(wsP);
        }
      }
    }
  }
  // Data members
  Box3d &m_wsBounds;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::GetIntersections
{
  //! Ctor
  GetIntersections(const Ray3d &wsRay, IntervalVec &intervals)
    : m_wsRay(wsRay), m_intervals(intervals)
  { 

  }
  //! Intersect matrix mapping
  void intersectMatrixMapping(const Ray3d &wsRay,
                              const MatrixFieldMapping *mtx,
                              const float worldScale) const
  {
    using std::min;

    const float time = 0.0f;
    
    // Transform ray to local space for intersection test
    Ray3d lsRay;
    mtx->worldToLocal(wsRay.pos, lsRay.pos, time);
    mtx->worldToLocalDir(wsRay.dir, lsRay.dir);
    // Use unit bounding box to intersect against
    Box3d lsBBox(V3d(0.0), V3d(1.0));
    // Calculate intersection points
    double t0, t1;
    // Add the interval if the ray intersects the box
    if (detail::intersect(lsRay, lsBBox, t0, t1)) {
      const V3d    wsVoxelSize = mtx->wsVoxelSize(0, 0, 0);
      const double minLen      = min(min(wsVoxelSize.x, wsVoxelSize.y),
                                     wsVoxelSize.z);
      m_intervals.push_back(Interval(t0, t1, minLen * worldScale));
    } 
  }
  //! Intersect frustum mapping
  void intersectFrustumMapping(const Ray3d &wsRay,
                               const FrustumFieldMapping *mtx,
                               const float worldScale) const
  {
    using std::min;

    typedef std::vector<V3d> PointVec;
    
    const float time = 0.0f;

    // Get the eight corners of the local space bounding box
    Box3d lsBounds(V3d(0.0), V3d(1.0));
    PointVec lsCorners = cornerPoints(lsBounds);
    // Get the world space positions of the eight corners of the frustum
    PointVec wsCorners(lsCorners.size());
    for (PointVec::iterator lsP = lsCorners.begin(), wsP = wsCorners.begin(),
           end = lsCorners.end(); lsP != end; ++lsP, ++wsP) {
      mtx->localToWorld(*lsP, *wsP, time);
    }

    // Construct plane for each face of frustum
    Plane3d planes[6];
    planes[0] = Plane3d(wsCorners[4], wsCorners[0], wsCorners[6]);
    planes[1] = Plane3d(wsCorners[1], wsCorners[5], wsCorners[3]);
    planes[2] = Plane3d(wsCorners[4], wsCorners[5], wsCorners[0]);
    planes[3] = Plane3d(wsCorners[2], wsCorners[3], wsCorners[6]);
    planes[4] = Plane3d(wsCorners[0], wsCorners[1], wsCorners[2]);
    planes[5] = Plane3d(wsCorners[5], wsCorners[4], wsCorners[7]);

    // Intersect ray against planes
    double t0 = -std::numeric_limits<double>::max();
    double t1 =  std::numeric_limits<double>::max();
    for (int i = 0; i < 6; ++i) {
      double t;
      const Plane3d &p = planes[i];
      if (p.intersectT(wsRay, t)) {
        if (wsRay.dir.dot(p.normal) > 0.0) {
          // Non-opposing plane
          t1 = std::min(t1, t);
        } else {
          // Opposing plane
          t0 = std::max(t0, t);
        }
      }
    }
    if (t0 < t1) {
      t0 = std::max(t0, 0.0);
      const V3d    wsVoxelSize = mtx->wsVoxelSize(0, 0, 0);
      const double minLen      = min(min(wsVoxelSize.x, wsVoxelSize.y),
                                     wsVoxelSize.z);
      m_intervals.push_back(Interval(t0, t1, minLen * worldScale));
    }
  }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    // Intersect the ray against all the fields
    for (size_t field = 0, end = vec.size(); field < end; ++field) {
      // Check object space transform
      Ray3d wsRay = m_wsRay;
      if (vec[field].doOsToWs) {
        vec[field].wsToOs.multVecMatrix(m_wsRay.pos, wsRay.pos);
        vec[field].wsToOs.multDirMatrix(m_wsRay.dir, wsRay.dir);
      }
      // Pointer to mapping
      const FieldMapping* m = vec[field].mapping;
      // Check matrix mapping
      if (const MatrixFieldMapping *mtx = 
          dynamic_cast<const MatrixFieldMapping*>(m)) {
        intersectMatrixMapping(wsRay, mtx, vec[field].worldScale);
      }
      // Check frustum mapping
      if (const FrustumFieldMapping *f = 
          dynamic_cast<const FrustumFieldMapping*>(m)) {
        intersectFrustumMapping(wsRay, f, vec[field].worldScale);
      }
    }
  }
  // Data members
  const Ray3d &m_wsRay;
  IntervalVec &m_intervals;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::GetMinMax
{
  //! Ctor
  GetMinMax(const Box3d &wsBounds, float *min, float *max)
    : m_wsBounds(wsBounds), m_min(min), m_max(max)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::getMinMax(vec, m_wsBounds, m_min, m_max);
  }
  // Data members
  const Box3d &m_wsBounds;
  float *m_min;
  float *m_max;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::GetMinMaxMIP
{
  //! Ctor
  GetMinMaxMIP(const Box3d &wsBounds, float *min, float *max)
    : m_wsBounds(wsBounds), m_min(min), m_max(max)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::getMinMaxMIP(vec, m_wsBounds, m_min, m_max);
  }
  // Data members
  const Box3d &m_wsBounds;
  float *m_min;
  float *m_max;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::GetMinMaxPrefilt
{
  enum MinMaxMode {
    Min,
    Max
  };
  //! Ctor
  GetMinMaxPrefilt(const Box3d &wsBounds, float *result, MinMaxMode mode)
    : m_wsBounds(wsBounds), m_result(result), m_mode(mode)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    if (m_mode == Min) {
      FieldSampler<T, Dims_T>::getMinMaxPrefilt(vec, m_wsBounds, m_result, 
                                                FieldSampler<T, Dims_T>::Min);
    } else {
      FieldSampler<T, Dims_T>::getMinMaxPrefilt(vec, m_wsBounds, m_result, 
                                                FieldSampler<T, Dims_T>::Max);
    }
  }
  // Data members
  const Box3d &m_wsBounds;
  float       *m_result;
  MinMaxMode   m_mode;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::GetMinMaxTemporal
{
  //! Ctor
  GetMinMaxTemporal(const Box3d &wsBounds, float *min, float *max)
    : m_wsBounds(wsBounds), m_min(min), m_max(max)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::getMinMaxTemporal(vec, m_wsBounds, 
                                               m_min, m_max);
  }
  // Data members
  const Box3d &m_wsBounds;
  float *m_min;
  float *m_max;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::GetMinMaxMIPTemporal
{
  //! Ctor
  GetMinMaxMIPTemporal(const Box3d &wsBounds, float *min, float *max)
    : m_wsBounds(wsBounds), m_min(min), m_max(max)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::getMinMaxTemporal(vec, m_wsBounds, 
                                               m_min, m_max);
  }
  // Data members
  const Box3d &m_wsBounds;
  float *m_min;
  float *m_max;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::MinWsVoxelSize
{
  //! Ctor
  MinWsVoxelSize(float &sizeMin)
    : m_sizeMin(sizeMin)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::getMinWsVoxelSize(vec, m_sizeMin);
  }
  // Data members
  float &m_sizeMin;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::MemSize
{
  //! Ctor
  MemSize(long long int &memSize)
    : m_memSize(&memSize)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    for (size_t field = 0, end = vec.size(); field < end; ++field) {
      *m_memSize += vec[field].field->memSize();
    }
  }
  //! Result
  long long int result() const
  { return m_memSize; }
  // Data members
  long long int *m_memSize;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::PointIsect
{
  //! Ctor
  PointIsect(const V3d &wsP, bool &doesIntersect)
    : m_wsP(wsP), m_doesIntersect(&doesIntersect)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    for (size_t field = 0, end = vec.size(); field < end; ++field) {
      // Loop over fields in vector
      for (size_t i = 0, end = vec.size(); i < end; ++i) {
        V3d vsP;
        // Apply world to object transform
        if (vec[i].doOsToWs) {
          V3d osP;
          vec[i].wsToOs.multVecMatrix(m_wsP, osP);
          vec[i].mapping->worldToVoxel(osP, vsP);
        } else {
          vec[i].mapping->worldToVoxel(m_wsP, vsP);
        }
        // Sample
        if (vec[i].vsBounds.intersects(vsP)) {
          *m_doesIntersect = true;
        } 
      }
    }
  }
  //! Result
  bool result() const
  { return *m_doesIntersect; }
private:
  // Data members
  V3d  m_wsP;
  bool *m_doesIntersect;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::PointIsectMultiple
{
  //! Ctor
  PointIsectMultiple(const size_t n, const float *p, 
                     bool *result)
    : m_neval(n), m_p(p), m_result(result)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    for (size_t field = 0, end = vec.size(); field < end; ++field) {
      // Loop over fields in vector
      for (size_t i = 0, end = vec.size(); i < end; ++i) {
        const Imath::Box3d &vsBounds = vec[i].vsBounds;
        const FieldMapping *m = vec[i].mapping;

        if (vec[i].doOsToWs) {
          for (size_t ieval = 0; ieval < m_neval; ++ieval) {

            const V3d wsP(*reinterpret_cast<const V3f*>(m_p + 3 * ieval));

            // Apply world to object transform
            V3d osP;
            V3d vsP;
            vec[i].wsToOs.multVecMatrix(wsP, osP);
            m->worldToVoxel(osP, vsP);
            // Sample
            if (vsBounds.intersects(vsP))
              m_result[ieval] = true;
          }
        } else {
          for (size_t ieval = 0; ieval < m_neval; ++ieval) {

            const V3d wsP(*reinterpret_cast<const V3f*>(m_p + 3 * ieval));
            V3d vsP;

            // Apply world to object transform
            m->worldToVoxel(wsP, vsP);
            // Sample
            if (vsBounds.intersects(vsP))
              m_result[ieval] = true;
          }
        }
      }
    }
  }
private:
  // Data members
  const int m_neval;
  const float *m_p;
  bool *m_result;
};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//------------------------------------------------------------------------------

#endif // include guard

//------------------------------------------------------------------------------
