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
#include <lava/field3d15/DenseField.h>
#include <lava/field3d15/Field3DFile.h>
#include <lava/field3d15/FieldInterp.h>
#include <lava/field3d15/InitIO.h>
#include <lava/field3d15/MIPField.h>
#include <lava/field3d15/MIPUtil.h>
#include <lava/field3d15/SparseField.h>

// Project includes
#include "FieldWrapper.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//------------------------------------------------------------------------------
// MPL stuff 
//------------------------------------------------------------------------------

namespace mpl       = boost::mpl;
namespace ph        = mpl::placeholders;
namespace fusion    = boost::fusion;
namespace fusion_ro = boost::fusion::result_of;

typedef mpl::vector<Field3D::half, float, double>             ScalarTypes;
typedef mpl::vector<Field3D::V3h, Field3D::V3f, Field3D::V3d> VectorTypes;

//------------------------------------------------------------------------------
// MPL utilities
//------------------------------------------------------------------------------

namespace detail {

//------------------------------------------------------------------------------

template <typename T>
struct MakeDense
{
  typedef typename FieldWrapper<Field3D::DenseField<T> >::Vec type;
};

//------------------------------------------------------------------------------

template <typename T>
struct MakeSparse
{
  typedef typename FieldWrapper<Field3D::SparseField<T> >::Vec type;
};

//------------------------------------------------------------------------------

template <typename T>
struct MakeMIPDense
{
  // typedef typename MIPFieldWrapper<Field3D::MIPDenseField<T> >::Vec type;
  typedef typename 
  MIPFieldWrapper<Field3D::MIPField<Field3D::DenseField<T> > >::Vec type;
};

//------------------------------------------------------------------------------

template <typename T>
struct MakeMIPSparse
{
  // typedef typename MIPFieldWrapper<Field3D::MIPSparseField<T> >::Vec type;
  typedef typename 
  MIPFieldWrapper<Field3D::MIPField<Field3D::SparseField<T> > >::Vec type;
};

//------------------------------------------------------------------------------

template <int Dims_T>
struct LoadFields;

template <>
struct LoadFields<1>
{
  // Ctor
  LoadFields(Field3D::Field3DInputFile &in, const std::string &name,
             const std::string &attribute, Field3D::FieldRes::Vec &results)
    : m_in(in), m_name(name), m_attribute(attribute), m_results(results)
  { }
  // Functor
  template <typename T>
  void operator()(T)
  {
    // Load all fields of type T
    typename Field3D::Field<T>::Vec fields = 
      m_in.readScalarLayers<T>(m_name, m_attribute);
    // Add the fields to the result
    BOOST_FOREACH (const typename Field3D::Field<T>::Ptr &ptr, fields) {
      m_results.push_back(ptr);
    }
  }
  // Data members
  Field3D::Field3DInputFile &m_in;
  const std::string         &m_name;
  const std::string         &m_attribute;
  Field3D::FieldRes::Vec    &m_results;
};

template <>
struct LoadFields<3>
{
  // Ctor
  LoadFields(Field3D::Field3DInputFile &in, const std::string &name,
             const std::string &attribute, Field3D::FieldRes::Vec &results)
    : m_in(in), m_name(name), m_attribute(attribute), m_results(results)
  { }
  // Functor
  template <typename Vec_T>
  void operator()(Vec_T)
  {
    typedef typename Vec_T::BaseType T;

    // Load all fields of type T
    typename Field3D::Field<Vec_T>::Vec fields = 
      m_in.readVectorLayers<T>(m_name, m_attribute);
    // Add the fields to the result
    BOOST_FOREACH (const typename Field3D::Field<Vec_T>::Ptr &ptr, fields) {
      m_results.push_back(ptr);
    }
  }  
  // Data members
  Field3D::Field3DInputFile &m_in;
  const std::string         &m_name;
  const std::string         &m_attribute;
  Field3D::FieldRes::Vec    &m_results;
};

//----------------------------------------------------------------------------//

inline std::vector<V3d> 
cornerPoints(const Box3d &box)
{
  std::vector<V3d> result;
  result.push_back(V3d(box.min.x, box.min.y, box.min.z));
  result.push_back(V3d(box.max.x, box.min.y, box.min.z));
  result.push_back(V3d(box.min.x, box.max.y, box.min.z));
  result.push_back(V3d(box.max.x, box.max.y, box.min.z));
  result.push_back(V3d(box.min.x, box.min.y, box.max.z));
  result.push_back(V3d(box.max.x, box.min.y, box.max.z));
  result.push_back(V3d(box.min.x, box.max.y, box.max.z));
  result.push_back(V3d(box.max.x, box.max.y, box.max.z));
  return result;
}

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
    detail::MakeDense<ph::_1> >::type     MPLDenseTypes;
  typedef typename mpl::transform<
    MPLBaseTypes, 
    detail::MakeSparse<ph::_1> >::type    MPLSparseTypes;
  typedef typename mpl::transform<
    MPLBaseTypes, 
    detail::MakeMIPDense<ph::_1> >::type  MPLMIPDenseTypes;
  typedef typename mpl::transform<
    MPLBaseTypes, 
    detail::MakeMIPSparse<ph::_1> >::type MPLMIPSparseTypes;

  // Map MPL types to boost fusion types
  typedef typename fusion_ro::as_vector<MPLDenseTypes>::type     DenseTypes;
  typedef typename fusion_ro::as_vector<MPLSparseTypes>::type    SparseTypes;
  typedef typename fusion_ro::as_vector<MPLMIPDenseTypes>::type  MIPDenseTypes;
  typedef typename fusion_ro::as_vector<MPLMIPSparseTypes>::type MIPSparseTypes;

  // Ctors ---------------------------------------------------------------------

  //! Default constructor, does nothing
  FieldGroup();
  //! Construct from a set of fields
  FieldGroup(const Field3D::FieldRes::Vec &fields);

  // Main methods --------------------------------------------------------------

  void setup(const Field3D::FieldRes::Vec &fields);

  //! Loads all fields from a given file and optional attribute pattern
  //! \returns Success state
  bool load(const std::string &filename, const std::string &attribute);
  //! The number of fields in the group
  size_t size() const;
  //! Samples the group of fields at the given point. This call will not
  //! include MIP fields, which require a spot size. 
  void sample(const V3d &vsP, float *result, bool isVs) const;
  //! Samples all the MIP fields in the group.
  void sampleMIP(const V3d &vsP, const float wsSpotSize, 
                 float *result, bool isVs) const;
  //! Gets the intersection intervals between the ray and the fields
  bool getIntersections(const Ray3d &ray, IntervalVec &intervals) const;
  //! Returns the min/max range within a given bounding box.
  void getMinMax(const Box3d &wsBounds, float *min, float *max) const;
  //! Returns the memory use in bytes for the fields in the group
  long long int memSize() const;

private:

  // Data members --------------------------------------------------------------
  
  DenseTypes     dense;
  SparseTypes    sparse;
  MIPDenseTypes  mipDense;
  MIPSparseTypes mipSparse;
  
  // Functors ------------------------------------------------------------------

  struct GrabFields;
  struct CountFields;
  struct Sample;
  struct SampleMIP;
  struct GetIntersections;
  struct GetMinMax;
  struct GetMinMaxMIP;
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
{ }

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
FieldGroup<BaseTypeList_T, Dims_T>::FieldGroup
(const Field3D::FieldRes::Vec &fields)
{
  setup(fields);
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::setup(const Field3D::FieldRes::Vec &fields)
{
  // Pick out only scalar fields
  for (size_t i = 0, end = fields.size(); i < end; ++i) {
    GrabFields op(fields[i]);
    fusion::for_each(dense, op);
    fusion::for_each(sparse, op);
    fusion::for_each(mipDense, op);
    fusion::for_each(mipSparse, op);
  }
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
bool 
FieldGroup<BaseTypeList_T, Dims_T>::load
(const std::string &filename, const std::string &attribute)
{
  using namespace Field3D;
  
  FieldRes::Vec results;

  const size_t sizeBeforeLoading = size();

  // Open each file ---

  std::vector<std::string> filenames;
  filenames.push_back(filename);

  BOOST_FOREACH (const std::string fn, filenames) {

    Field3DInputFile in;
    if (!in.open(fn)) {
      return false;
    }

    // Use partition names to determine if fields should be loaded
    std::vector<std::string> names;
    in.getPartitionNames(names);

    BOOST_FOREACH (const std::string &name, names) {
      detail::LoadFields<Dims_T> op(in, name, attribute, results);
      mpl::for_each<BaseTypeList_T>(op);
    }

  }

  // Set up from fields
  setup(results);

  // Done. Check whether we loaded anything and return false if we didn't.
  return size() != sizeBeforeLoading;
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
size_t 
FieldGroup<BaseTypeList_T, Dims_T>::size() const
{
  CountFields op;
  fusion::for_each(dense, op);
  fusion::for_each(sparse, op);
  fusion::for_each(mipDense, op);
  fusion::for_each(mipSparse, op);
  return op.count;
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::sample(const V3d &vsP, 
                                           float *result, 
                                           bool isVs) const
{
  Sample op(vsP, result, isVs);
  fusion::for_each(dense, op);
  fusion::for_each(sparse, op);
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::sampleMIP(const V3d &vsP, 
                                              const float wsSpotSize, 
                                              float *result, 
                                              bool isVs) const
{
  SampleMIP op(vsP, wsSpotSize, result, isVs);
  fusion::for_each(mipDense, op);
  fusion::for_each(mipSparse, op);
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
bool 
FieldGroup<BaseTypeList_T, Dims_T>::getIntersections
(const Ray3d &ray, IntervalVec &intervals) const
{
  GetIntersections op(ray, intervals);
  fusion::for_each(dense, op);
  fusion::for_each(sparse, op);
  fusion::for_each(mipDense, op);
  fusion::for_each(mipSparse, op);
  return intervals.size() > 0;
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
void 
FieldGroup<BaseTypeList_T, Dims_T>::getMinMax(const Box3d &wsBounds, 
                                              float *min, 
                                              float *max) const
{
  GetMinMax op(wsBounds, min, max);
  fusion::for_each(dense, op);
  fusion::for_each(sparse, op);
  GetMinMaxMIP opMIP(wsBounds, min, max);
  fusion::for_each(mipDense, opMIP);
  fusion::for_each(mipSparse, opMIP);    
}

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
long long int
FieldGroup<BaseTypeList_T, Dims_T>::memSize() const
{
  long long int result = 0;
  MemSize op(result);
  fusion::for_each(dense, op);
  fusion::for_each(sparse, op);
  fusion::for_each(mipDense, op);
  fusion::for_each(mipSparse, op);
  return result;
}

//------------------------------------------------------------------------------
// Functor implementations 
//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::GrabFields
{
  //! Ctor
  GrabFields(Field3D::FieldRes::Ptr f)
    : m_field(f)
  { }
  //! Functor
  template <typename WrapperVec_T>
  void operator()(WrapperVec_T &vec) const
  { 
    typedef typename WrapperVec_T::value_type Wrapper_T;
    typedef typename Wrapper_T::field_type    Field_T;
    typedef typename Field_T::Ptr             FieldPtr;
    if (FieldPtr f = 
        Field3D::field_dynamic_cast<Field_T>(m_field)) {
      vec.push_back(f);
    }
  }
  //! The field to work on. Will be matched against the type of operator().
  Field3D::FieldRes::Ptr m_field;
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
struct FieldGroup<BaseTypeList_T, Dims_T>::Sample
{
  //! Ctor
  Sample(const V3d &p, float *result, bool isVs)
    : m_p(p), m_result(result), m_isVs(isVs)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { FieldSampler<T, Dims_T>::sample(vec, m_p, m_result, m_isVs); }
  // Data members
  const V3d &m_p;
  float *m_result;
  bool m_isVs;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::SampleMIP
{
  //! Ctor
  SampleMIP(const V3d &p, const float wsSpotSize, float *result, 
            bool isVs)
    : m_p(p), m_wsSpotSize(wsSpotSize), m_result(result), m_isVs(isVs)
  { }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    FieldSampler<T, Dims_T>::sampleMIP(vec, m_p, m_wsSpotSize, m_result, 
                                       m_isVs); 
  }
  // Data members
  const V3d &m_p;
  const float m_wsSpotSize;
  float *m_result;
  bool m_isVs;
};

//------------------------------------------------------------------------------

template <typename BaseTypeList_T, int Dims_T>
struct FieldGroup<BaseTypeList_T, Dims_T>::GetIntersections
{
  //! Ctor
  GetIntersections(const Ray3d &wsRay, IntervalVec &intervals)
    : m_wsRay(wsRay), m_intervals(intervals)
  { }
  //! Intersect matrix mapping
  void intersectMatrixMapping(const MatrixFieldMapping *mtx) const
  {
    using std::min;

    const float time = 0.0f;
    
    // Transform ray to local space for intersection test
    Ray3d lsRay;
    mtx->worldToLocal(m_wsRay.pos, lsRay.pos, time);
    mtx->worldToLocalDir(m_wsRay.dir, lsRay.dir);
    // Use unit bounding box to intersect against
    Box3d lsBBox(V3d(0.0), V3d(1.0));
    // Calculate intersection points
    double t0, t1;
    // Add the interval if the ray intersects the box
    if (detail::intersect(lsRay, lsBBox, t0, t1)) {
      const V3d    wsVoxelSize = mtx->wsVoxelSize(0, 0, 0);
      const double minLen      = min(min(wsVoxelSize.x, wsVoxelSize.y),
                                     wsVoxelSize.z);
      m_intervals.push_back(Interval(t0, t1, minLen));
    } 
  }
  //! Intersect frustum mapping
  void intersectFrustumMapping(const FrustumFieldMapping *mtx) const
  {
    using std::min;

    typedef std::vector<V3d> PointVec;
    
    const float time = 0.0f;

    // Get the eight corners of the local space bounding box
    Box3d lsBounds(V3d(0.0), V3d(1.0));
    PointVec lsCorners = detail::cornerPoints(lsBounds);
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
      if (p.intersectT(m_wsRay, t)) {
        if (m_wsRay.dir.dot(p.normal) > 0.0) {
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
      const double wsLength    = (m_wsRay(t0) - m_wsRay(1)).length();
      const V3d    wsVoxelSize = mtx->wsVoxelSize(0, 0, 0);
      const double minLen      = min(min(wsVoxelSize.x, wsVoxelSize.y),
                                     wsVoxelSize.z);
      m_intervals.push_back(Interval(t0, t1, wsLength / minLen));
    }
  }
  //! Functor
  template <typename T>
  void operator()(const T &vec) const
  { 
    // Intersect the ray against all the fields
    for (size_t field = 0, end = vec.size(); field < end; ++field) {
      // Pointer to mapping
      const FieldMapping* m = vec[field].mapping;
      // Check matrix mapping
      if (const MatrixFieldMapping *mtx = 
          dynamic_cast<const MatrixFieldMapping*>(m)) {
        intersectMatrixMapping(mtx);
      }
      // Check frustum mapping
      if (const FrustumFieldMapping *f = 
          dynamic_cast<const FrustumFieldMapping*>(m)) {
        intersectFrustumMapping(f);
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

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//------------------------------------------------------------------------------

#endif // include guard

//------------------------------------------------------------------------------
