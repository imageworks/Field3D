//----------------------------------------------------------------------------//

#ifndef _INCLUDED_TemporalFieldUtil_H_
#define _INCLUDED_TemporalFieldUtil_H_

//----------------------------------------------------------------------------//

#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/thread/mutex.hpp>

#include <OpenEXR/ImathFun.h>
#include <ImathHalfLimits.h>

#include "FieldSampler.h"
#include "SparseField.h"
#include "TemporalField.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Utility functions for TemporalFields
//----------------------------------------------------------------------------//

template <typename Data_T>
typename SparseField<Data_T>::Ptr 
integrateTemporalField(typename TemporalField<Data_T>::Ptr tfPtr,
                       const float t0, const float t1);

template <typename Data_T>
typename SparseField<Data_T>::Ptr 
minTemporalField(const typename TemporalField<Data_T>::Ptr &tfPtr,
                 const float t0, const float t1);

template <typename Data_T>
typename SparseField<Data_T>::Ptr 
maxTemporalField(const typename TemporalField<Data_T>::Ptr &tfPtr,
                 const float t0, const float t1);

template <typename Data_T>
typename SparseField<Data_T>::Ptr 
sliceTemporalField(typename TemporalField<Data_T>::Ptr tfPtr, 
                   const float sliceTime);


//----------------------------------------------------------------------------//
// Helper functions
//----------------------------------------------------------------------------//

template <typename T>
T integrate(const float* times, const T* values, const size_t numValues,
            float t0, float t1);

template <typename T>
T min(const float* times, const T* values, const size_t numValues,
      float t0, float t1);

template <typename T>
T max(const float* times, const T* values, const size_t numValues,
      float t0, float t1);

//----------------------------------------------------------------------------//

namespace detail {

  //------------------------------------------------------------------------- //
  
  template <typename T> struct limits
  {
    static T	min()      { return Imath::limits<T>::min(); }
    static T	max()      { return Imath::limits<T>::max(); }
    static T	smallest() { return Imath::limits<T>::smallest(); }
    static T	epsilon()  { return Imath::limits<T>::epsilon(); }
    static bool isIntegral() { return Imath::limits<T>::isIntegral(); }
    static bool isSigned()   { return Imath::limits<T>::isSigned(); }
  };

  //------------------------------------------------------------------------- //
  
  template <typename T> struct limits<Imath::Vec3<T> >
  {
    static Imath::Vec3<T>	min() 
    { return Imath::Vec3<T>(Imath::limits<T>::min()); }

    static Imath::Vec3<T>	max()
    { return Imath::Vec3<T>(Imath::limits<T>::max()); }

    static Imath::Vec3<T>	smallest()
    { return Imath::Vec3<T>(Imath::limits<T>::smallest()); }

    static Imath::Vec3<T>	epsilon() 
    { return Imath::Vec3<T>(Imath::limits<T>::epsilon()); }

    static bool isIntegral() { return false; }
    static bool isSigned()   { return Imath::limits<T>::isSigned(); }
  };
  
  //------------------------------------------------------------------------- //

} // namespace detail

//----------------------------------------------------------------------------//
// Implementations
//----------------------------------------------------------------------------//

template <typename T>
T integrate(const float* times, const T* values, const size_t numValues,
            float t0, float t1)
{
  // Uses piecewise linear integration 

  using namespace std;

  // Swap if arguments are reversed
  //! \todo Should we throw() instead?
  if (t0 > t1) {
    std::swap(t0, t1);
  }

  // If there are no samples, return zero
  if (numValues == 0) {
    return static_cast<T>(0.0);
  }

  // If there is just one sample, compute the integral of that
  if (numValues == 1) {
    return (t1 - t0) * values[0];
  }

  T sum = static_cast<T>(0.0);

  // Walk through samples pair-by-pair
  for (size_t i = 0; i < numValues - 1; ++i) {

    float    st0 = times[i];
    float    st1 = times[i + 1];
    const T &v0  = values[i];
    const T &v1  = values[i + 1];

    // Special case at first and last element in case t0/t1 is outside
    if (i == 0) {
      if (t0 < st0) {
        // Integrate up to first vertex
        sum += v0 * (std::min(st0, t1) - t0);
      }
    } 
    if (i == numValues - 2) {
      if (st1 < t1) {
        // Integrate past last vertex
        sum += v1 * (t1 - std::max(st1, t0));
      }
    }

    // Find overlap between st0/st1 and t0/t1
    float start  = std::max(t0, st0);
    float end    = std::min(t1, st1);
    float length = std::max(0.0f, end - start);

    // Midpoint integration
    if (start == st0 && end == st1) {
      // Optimized case for full segment
      sum += (v1 + v0) * 0.5 * length;
    } else {
      // General case for partial segment
      float interpT = Imath::lerpfactor((start + end) * 0.5f, st0, st1);
      sum += Imath::lerp(v0, v1, interpT) * length;
    }
    
  }

  return sum;
}

template <typename T>
T min(const float* times, const T* values, const size_t numValues,
      float t0, float t1)
{
  // Uses piecewise linear integration 

  using namespace std;

  // Swap if arguments are reversed
  //! \todo Should we throw() instead?
  if (t0 > t1) {
    std::swap(t0, t1);
  }

  // If there are no samples, return zero
  if (numValues == 0) {
    return static_cast<T>(0.0);
  }

  // If there is just one sample, return that sample
  if (numValues == 1) {
    return values[0];
  }

  T minValue = detail::limits< T >::max();

  float    st0 = times[0];
  const T &v0  = values[0];
  float start  = std::max(t0, st0);

  // Startpoint check
  if (start == st0) {
    minValue = v0;
  }

  // Walk through samples pair-by-pair
  for (size_t i = 0; i < numValues - 1; ++i) {

    float    st0 = times[i];
    float    st1 = times[i + 1];
    const T &v0  = values[i];
    const T &v1  = values[i + 1];

    // Find overlap between st0/st1 and t0/t1
    float start  = std::max(t0, st0);
    float end    = std::min(t1, st1);

    // Endpoint check
    if (start == st0 && end == st1) {
      // Case for full segment
      minValue = detail::min(minValue, v1);
    } else {
      // General case for partial segment
      minValue = detail::min(minValue, detail::min(
                  v0 + (start - st0) / (st1 - st0) * (v1 - v0),
                  v0 + (end - st0) / (st1 - st0) * (v1 - v0)));
    }
  }

  return minValue;
}

template <typename T>
T max(const float* times, const T* values, const size_t numValues,
      float t0, float t1)
{
  // Uses piecewise linear integration 

  using namespace std;

  // Swap if arguments are reversed
  //! \todo Should we throw() instead?
  if (t0 > t1) {
    std::swap(t0, t1);
  }

  // If there are no samples, return zero
  if (numValues == 0) {
    return static_cast<T>(0.0);
  }

  // If there is just one sample, return that sample
  if (numValues == 1) {
    return values[0];
  }

  T maxValue = -detail::limits< T >::max();

  float    st0 = times[0];
  const T &v0  = values[0];
  float start  = std::max(t0, st0);

  // Startpoint check
  if (start == st0) {
    maxValue = v0;
  }

  // Walk through samples pair-by-pair
  for (size_t i = 0; i < numValues - 1; ++i) {

    float    st0 = times[i];
    float    st1 = times[i + 1];
    const T &v0  = values[i];
    const T &v1  = values[i + 1];

    // Find overlap between st0/st1 and t0/t1
    float start  = std::max(t0, st0);
    float end    = std::min(t1, st1);

    // Endpoint check
    if (start == st0 && end == st1) {
      // Case for full segment
      maxValue = detail::max(maxValue, v1);
    } else {
      // General case for partial segment
      maxValue = detail::max(maxValue, detail::max(
                  v0 + (start - st0) / (st1 - st0) * (v1 - v0),
                  v0 + (end - st0) / (st1 - st0) * (v1 - v0)));
    }
  }

  return maxValue;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
typename SparseField<Data_T>::Ptr 
integrateTemporalField(typename TemporalField<Data_T>::Ptr tfPtr,
                       const float t0, const float t1)
{
  TemporalField<Data_T> *tf = tfPtr.get();

  typename SparseField<Data_T>::Ptr fieldPtr(new SparseField<Data_T>);
  SparseField<Data_T> *field = fieldPtr.get();

  field->matchDefinition(tf);
  field->copyMetadata(*tf);
  field->name = tf->name;
  field->attribute = tf->attribute;

  float scalingFactor = 1.0 / (t1 - t0);
  
  Imath::V3i blockRes = tf->blockRes();
  for (int bk = 0; bk < blockRes.z; ++bk) {
    for (int bj = 0; bj < blockRes.y; ++bj) {
      for (int bi = 0; bi < blockRes.x; ++bi) {
        TemporalBlock<Data_T> *block = tf->block(bi, bj, bk);
        Imath::Box3i ext = tf->blockExtents(bi, bj, bk);
        if (block->state() == TemporalBlock<Data_T>::ActiveState) {
          for (int k = ext.min.z; k <= ext.max.z; ++k) {
            for (int j = ext.min.y; j <= ext.max.y; ++j) {
              for (int i = ext.min.x; i <= ext.max.x; ++i) {
                size_t numSamples = tf->numSamples(i, j, k);
                const float *times = tf->sampleTimes(i, j, k);
                const Data_T *values = tf->sampleValues(i, j, k);
                Data_T val = integrate(times, values, numSamples, t0, t1);
                field->fastLValue(i, j, k) = val * scalingFactor;
              }
            }
          }
        }
      }
    }
  }

  return fieldPtr;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
typename SparseField<Data_T>::Ptr 
minTemporalField(const typename TemporalField<Data_T>::Ptr &tfPtr,
                 const float t0, const float t1)
{
  TemporalField<Data_T> *tf = tfPtr.get();

  typename SparseField<Data_T>::Ptr fieldPtr(new SparseField<Data_T>);
  SparseField<Data_T> *field = fieldPtr.get();

  field->matchDefinition(tf);
  field->copyMetadata(*tf);
  field->name = tf->name;
  field->attribute = tf->attribute;

  Imath::V3i blockRes = tf->blockRes();
  for (int bk = 0; bk < blockRes.z; ++bk) {
    for (int bj = 0; bj < blockRes.y; ++bj) {
      for (int bi = 0; bi < blockRes.x; ++bi) {
        TemporalBlock<Data_T> *block = tf->block(bi, bj, bk);
        Imath::Box3i ext = tf->blockExtents(bi, bj, bk);
        if (block->state() == TemporalBlock<Data_T>::ActiveState) {
          for (int k = ext.min.z; k <= ext.max.z; ++k) {
            for (int j = ext.min.y; j <= ext.max.y; ++j) {
              for (int i = ext.min.x; i <= ext.max.x; ++i) {
                size_t numSamples = tf->numSamples(i, j, k);
                const float *times = tf->sampleTimes(i, j, k);
                const Data_T *values = tf->sampleValues(i, j, k);
                Data_T val = min(times, values, numSamples, t0, t1);
                field->fastLValue(i, j, k) = val;
              }
            }
          }
        }
      }
    }
  }

  return fieldPtr;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
typename SparseField<Data_T>::Ptr 
maxTemporalField(const typename TemporalField<Data_T>::Ptr &tfPtr,
                 const float t0, const float t1)
{
  TemporalField<Data_T> *tf = tfPtr.get();

  typename SparseField<Data_T>::Ptr fieldPtr(new SparseField<Data_T>);
  SparseField<Data_T> *field = fieldPtr.get();

  field->matchDefinition(tf);
  field->copyMetadata(*tf);
  field->name = tf->name;
  field->attribute = tf->attribute;

  Imath::V3i blockRes = tf->blockRes();
  for (int bk = 0; bk < blockRes.z; ++bk) {
    for (int bj = 0; bj < blockRes.y; ++bj) {
      for (int bi = 0; bi < blockRes.x; ++bi) {
        TemporalBlock<Data_T> *block = tf->block(bi, bj, bk);
        Imath::Box3i ext = tf->blockExtents(bi, bj, bk);
        if (block->state() == TemporalBlock<Data_T>::ActiveState) {
          for (int k = ext.min.z; k <= ext.max.z; ++k) {
            for (int j = ext.min.y; j <= ext.max.y; ++j) {
              for (int i = ext.min.x; i <= ext.max.x; ++i) {
                size_t numSamples = tf->numSamples(i, j, k);
                const float *times = tf->sampleTimes(i, j, k);
                const Data_T *values = tf->sampleValues(i, j, k);
                Data_T val = max(times, values, numSamples, t0, t1);
                field->fastLValue(i, j, k) = val;
              }
            }
          }
        }
      }
    }
  }

  return fieldPtr;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
typename SparseField<Data_T>::Ptr 
sliceTemporalField(typename TemporalField<Data_T>::Ptr tfPtr, 
                   const float sliceTime)
{
  TemporalField<Data_T> *tf = tfPtr.get();

  typename SparseField<Data_T>::Ptr fieldPtr(new SparseField<Data_T>);
  SparseField<Data_T> *field = fieldPtr.get();

  field->matchDefinition(tf);
  field->copyMetadata(*tf);
  field->name = tf->name;
  field->attribute = tf->attribute;

  Imath::V3i blockRes = tf->blockRes();
  for (int bk = 0; bk < blockRes.z; ++bk) {
    for (int bj = 0; bj < blockRes.y; ++bj) {
      for (int bi = 0; bi < blockRes.x; ++bi) {
        TemporalBlock<Data_T> *block = tf->block(bi, bj, bk);
        Imath::Box3i ext = tf->blockExtents(bi, bj, bk);
        if (block->state() == TemporalBlock<Data_T>::ActiveState) {
          for (int k = ext.min.z; k <= ext.max.z; ++k) {
            for (int j = ext.min.y; j <= ext.max.y; ++j) {
              for (int i = ext.min.x; i <= ext.max.x; ++i) {
                field->fastLValue(i, j, k) = tf->fastValue(i, j, k, sliceTime);
              }
            }
          }
        }
      }
    }
  }


  return fieldPtr;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // include guard
