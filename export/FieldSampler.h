//----------------------------------------------------------------------------//

#ifndef __F3DUTIL_FIELDSAMPLER_H__
#define __F3DUTIL_FIELDSAMPLER_H__

//------------------------------------------------------------------------------

// Project includes
#include "Types.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//------------------------------------------------------------------------------
// detail namespace
//------------------------------------------------------------------------------

namespace detail {

  //! Min operation on mixed types
  template <typename T, typename T2>
  T min(const T a, const T2 b)
  {
    return std::min(a, static_cast<T>(b));
  }

  //! Max operation on mixed types
  template <typename T, typename T2>
  T max(const T a, const T2 b)
  {
    return std::max(a, static_cast<T>(b));
  }

  //! Min operation on mixed vector types
  template <typename T, typename T2>
  FIELD3D_VEC3_T<T> min(const FIELD3D_VEC3_T<T> &a, 
                        const FIELD3D_VEC3_T<T2> &b)
  {
    return FIELD3D_VEC3_T<T>(std::min(a.x, static_cast<T>(b.x)),
                             std::min(a.y, static_cast<T>(b.y)),
                             std::min(a.z, static_cast<T>(b.z)));
  }

  //! Max operation on mixed vector types
  template <typename T, typename T2>
  FIELD3D_VEC3_T<T> max(const FIELD3D_VEC3_T<T> &a, 
                        const FIELD3D_VEC3_T<T2> &b)
  {
    return FIELD3D_VEC3_T<T>(std::max(a.x, static_cast<T>(b.x)),
                             std::max(a.y, static_cast<T>(b.y)),
                             std::max(a.z, static_cast<T>(b.z)));
  }

  //! Typedefs float or V3f, depending on Dims_T
  template <int Dims_T>
  struct ScalarOrVector;

  template <>
  struct ScalarOrVector<1>
  {
    typedef float type;
  };

  template <>
  struct ScalarOrVector<3>
  {
    typedef V3f type;
  };

}

//------------------------------------------------------------------------------
// FieldSampler
//------------------------------------------------------------------------------

//! Interface for sampling a vector of fields of the same type
template <typename WrapperVec_T, int Dims_T>
struct FieldSampler
{
  enum Mode {
    Min,
    Max
  };

  typedef typename WrapperVec_T::value_type::field_type Field_T;
  typedef typename Field_T::value_type                  Data_T;
  typedef typename detail::ScalarOrVector<Dims_T>::type Input_T;

  // Ordinary fields
  static void sample(const WrapperVec_T &f, const V3d &wsP, float *value, 
                     size_t &numHits)
  {
    // Reinterpret the pointer according to Dims_T
    Input_T *data = reinterpret_cast<Input_T*>(value);
    // Loop over fields in vector
    for (size_t i = 0, end = f.size(); i < end; ++i) {
      V3d vsP;
      // Apply world to object transform
      if (f[i].doOsToWs) {
        V3d osP;
        f[i].wsToOs.multVecMatrix(wsP, osP);
        f[i].mapping->worldToVoxel(osP, vsP);
      } else {
        f[i].mapping->worldToVoxel(wsP, vsP);
      }
      // Sample
      if (f[i].vsBounds.intersects(vsP)) {
        // Count as within field
        numHits++;
        // Sample and remap
        if (f[i].valueRemapOp) {
          const Data_T unremapped = f[i].interp.sample(*f[i].field, vsP);
          *data += f[i].valueRemapOp->remap(unremapped);
        } else {
          *data += f[i].interp.sample(*f[i].field, vsP);
        }
      } 
    }
  }

  // Ordinary fields
  static void sampleMultiple(const WrapperVec_T &f, const size_t neval,
                             const float *wsPs, float *value, size_t *numHits)
  {
    // Loop over fields in vector
    for (size_t i = 0; i < f.size(); ++i) {
      const typename WrapperVec_T::value_type &field = f[i];

      // Reinterpret the pointer according to Dims_T
      Input_T *data = reinterpret_cast<Input_T*>(value);

      if (field.doOsToWs || field.valueRemapOp) {

        // Loop over samples
        for (size_t ieval = 0; ieval < neval; ++ieval) {
          const V3d wsP(*reinterpret_cast<const V3f*>(wsPs + 3 * ieval));
          V3d vsP;
          // Apply world to object transform
          if (field.doOsToWs) {
            V3d osP;
            field.wsToOs.multVecMatrix(wsP, osP);
            field.mapping->worldToVoxel(osP, vsP);
          } else {
            field.mapping->worldToVoxel(wsP, vsP);
          }
          // Sample
          if (field.vsBounds.intersects(vsP)) {
            // Count as within field
            numHits[ieval]++;
            // Sample and remap
            if (field.valueRemapOp) {
              const Data_T unremapped = field.interp.sample(*field.field, vsP);
              data[ieval] += field.valueRemapOp->remap(unremapped);
            } else {
              data[ieval] += field.interp.sample(*field.field, vsP);
            }
          } 
        }

      } else {

        const Imath::Box3d &vsBounds_d = field.vsBounds;

        // Loop over samples
        for (size_t ieval = 0; ieval < neval; ++ieval) {
          const V3d wsP(*reinterpret_cast<const V3f*>(wsPs + 3 * ieval));
          V3d vsP;

          // Apply world to object transform
          field.mapping->worldToVoxel(wsP, vsP);

          // Sample
          if (vsBounds_d.intersects(vsP)) {
            // Count as within field
            numHits[ieval]++;
            // Sample
            data[ieval] += field.interp.sample(*field.field, vsP);
          }
        }
      }
    }
  }

  // MIP fields
  static void sampleMIP(const WrapperVec_T &f, const V3d &wsP,
                        const float wsSpotSize, float *value, size_t &numHits)
  {
    // Reinterpret the pointer according to Dims_T
    Input_T *data = reinterpret_cast<Input_T*>(value);
    // Loop over fields in vector
    for (size_t i = 0, end = f.size(); i < end; ++i) {
      V3d vsP;
      float spotSize = wsSpotSize / f[i].worldScale;
      // Apply world to object transform
      if (f[i].doOsToWs) {
        V3d osP;
        f[i].wsToOs.multVecMatrix(wsP, osP);
        f[i].mapping->worldToVoxel(osP, vsP);
        spotSize = wsSpotSize / f[i].worldScale;
      } else {
        f[i].mapping->worldToVoxel(wsP, vsP);
      }
      // Sample
      if (f[i].vsBounds.intersects(vsP)) {
        // Count as within field
        numHits++;
        // Sample and remap
        if (f[i].valueRemapOp) {
          const Data_T unremapped = f[i].interp->sample(vsP, spotSize);
          *data += f[i].valueRemapOp->remap(unremapped);
        } else {
          *data += f[i].interp->sample(vsP, spotSize);
        }
      }
    }
  }

  // MIP fields
  static void sampleMIPMultiple(const WrapperVec_T &f, const size_t neval,
                                const float *wsPs, const float *wsSpotSizes,
                                float *value, size_t *numHits)
  {
    // Loop over fields in vector
    for (size_t i = 0; i < f.size(); ++i) {
      const typename WrapperVec_T::value_type &field = f[i];

      // Reinterpret the pointer according to Dims_T
      Input_T *data = reinterpret_cast<Input_T*>(value);

      if (field.doOsToWs || field.valueRemapOp) {

        if (field.valueRemapOp && field.doWsBoundsOptimization) {

          // Loop over samples
          for (size_t ieval = 0; ieval < neval; ++ieval) {
            const V3f &wsP = *reinterpret_cast<const V3f*>(wsPs + 3 * ieval);

            if (field.wsBounds.intersects(wsP)) {

              // Apply world to object transform
              V3d vsP;

              field.wsToVs.multVecMatrix(V3d(wsP), vsP);

              // Sample
              if (field.vsBounds.intersects(vsP)) {
                // Count as within field
                numHits[ieval]++;
                const float spotSize = wsSpotSizes[ieval] / field.worldScale;
                const Data_T unremapped = field.interp->sample(vsP, spotSize);
                data[ieval] += field.valueRemapOp->remap(unremapped);
              }
            }
          }

        } else {
          // Loop over samples
          for (size_t ieval = 0; ieval < neval; ++ieval) {
            const V3d wsP(*reinterpret_cast<const V3f*>(wsPs + 3 * ieval));
            const float wsSpotSize = wsSpotSizes[ieval];
            Input_T *idata = data + ieval;

            V3d vsP;
            float spotSize = wsSpotSize / field.worldScale;
            // Apply world to object transform
            if (field.doOsToWs) {
              V3d osP;
          
              field.wsToOs.multVecMatrix(wsP, osP);
              field.mapping->worldToVoxel(osP, vsP);
              spotSize = wsSpotSize / field.worldScale;
            } else {
              field.mapping->worldToVoxel(wsP, vsP);
            }
            // Sample
            if (field.vsBounds.intersects(vsP)) {
              // Count as within field
              numHits[ieval]++;
              if (field.valueRemapOp) {
                const Data_T unremapped = field.interp->sample(vsP, spotSize);
                *idata += field.valueRemapOp->remap(unremapped);
              } else {
                *idata += field.interp->sample(vsP, spotSize);
              }
            }
          }
        }
      } else {

        const Imath::Box3d &vsBounds_d = field.vsBounds;
        const double worldScale = field.worldScale;

        // Loop over samples
        for (size_t ieval = 0; ieval < neval; ++ieval) {
          const V3d wsP(*reinterpret_cast<const V3f*>(wsPs + 3 * ieval));
          V3d vsP;

          // Apply world to object transform
          field.mapping->worldToVoxel(wsP, vsP);

          // Sample
          if (vsBounds_d.intersects(vsP)) {
            // Count as within field
            numHits[ieval]++;
            const double spotSize = wsSpotSizes[ieval] / worldScale;

            data[ieval] += field.interp->sample(vsP, spotSize);
          }
        }
      }
    }
  }

  // Get min/max
  static void getMinMax(const WrapperVec_T &f, 
                        const Box3d &wsBounds, float *min, float *max)
  {
    // Reinterpret the pointer according to Dims_T
    Input_T *minData = reinterpret_cast<Input_T*>(min);
    Input_T *maxData = reinterpret_cast<Input_T*>(max);
    // Loop over fields in vector
    for (size_t field = 0, end = f.size(); field < end; ++field) {
      // Data window
      const Box3i dw = f[field].field->dataWindow();
      // Transform corners to voxel space and compute bounds
      Box3i dvsBounds;
      if (wsBounds.isInfinite()) {
        dvsBounds = dw;
      } else {
        Box3d vsBounds;
        if (f[field].doOsToWs) {
          Box3d osBounds;
          transformBounds(f[field].wsToOs, wsBounds, osBounds);
          worldToVoxel(f[field].mapping, osBounds, vsBounds);
        } else {
          worldToVoxel(f[field].mapping, wsBounds, vsBounds);
        }
        dvsBounds = clipBounds(discreteBounds(vsBounds), dw);
        // Early termination if no intersection
        if (!dw.intersects(dvsBounds)) {
          return;
        }
      }
      for (int k = dvsBounds.min.z; k <= dvsBounds.max.z; ++k) {
        for (int j = dvsBounds.min.y; j <= dvsBounds.max.y; ++j) {
          for (int i = dvsBounds.min.x; i <= dvsBounds.max.x; ++i) {
            const Data_T val = f[field].field->fastValue(i, j, k);
            *minData = detail::min(val, *minData);
            *maxData = detail::max(val, *maxData);
          }
        }
      }
    }
  }

  // Get min/max from MIP (uses finest level)
  static void getMinMaxMIP(const WrapperVec_T &f, 
                           const Box3d &wsBounds, float *min, float *max)
  {
    // Reinterpret the pointer according to Dims_T
    Input_T *minData = reinterpret_cast<Input_T*>(min);
    Input_T *maxData = reinterpret_cast<Input_T*>(max);
    // Loop over fields in vector
    for (size_t field = 0, end = f.size(); field < end; ++field) {
      // Data window
      const Box3i dw = f[field].field->dataWindow();
      // Transform corners to voxel space and compute bounds
      Box3i dvsBounds;
      if (wsBounds.isInfinite()) {
        dvsBounds = dw;
      } else {
        Box3d vsBounds;
        if (f[field].doOsToWs) {
          Box3d osBounds;
          transformBounds(f[field].wsToOs, wsBounds, osBounds);
          worldToVoxel(f[field].mapping, osBounds, vsBounds);
        } else {
          worldToVoxel(f[field].mapping, wsBounds, vsBounds);
        }
        dvsBounds = clipBounds(discreteBounds(vsBounds), dw);
        // Early termination if no intersection
        if (!dw.intersects(dvsBounds)) {
          return;
        }
      }
      for (int k = dvsBounds.min.z; k <= dvsBounds.max.z; ++k) {
        for (int j = dvsBounds.min.y; j <= dvsBounds.max.y; ++j) {
          for (int i = dvsBounds.min.x; i <= dvsBounds.max.x; ++i) {
            const Data_T val = f[field].field->fastMipValue(0, i, j, k);
            *minData = detail::min(val, *minData);
            *maxData = detail::max(val, *maxData);
          }
        }
      }
    }
  }

  // Get min/max for pre-filtered data
  static void getMinMaxPrefilt(const WrapperVec_T &f, 
                               const Box3d &wsBounds, 
                               float *result, 
                               const Mode mode)
  {
    // Reinterpret the pointer according to Dims_T
    Input_T *data = reinterpret_cast<Input_T*>(result);
    // Loop over fields in vector
    for (size_t field = 0, end = f.size(); field < end; ++field) {
      // Choose the MIP level to check
      const size_t numLevels  = f[field].field->numLevels();
      size_t       level      = 0;
      Box3i        dvsBounds;
      // Infinite bounds?
      if (wsBounds.isInfinite()) {
        // Use the coarsest level
        level = numLevels - 1;
        dvsBounds = f[field].field->mipLevel(level)->dataWindow();
      } else {
        for (size_t i = 0; i < numLevels; ++i) {
          // Update current level
          level = i;
          // Data window of current level
          const Box3i dw = f[field].field->mipLevel(level)->dataWindow();
          Box3d vsBounds;
          if (f[field].doOsToWs) {
            Box3d osBounds;
            transformBounds(f[field].wsToOs, wsBounds, osBounds);
            worldToVoxel(f[field].field->mipLevel(level)->mapping().get(), 
                         osBounds, vsBounds);
          } else {
            worldToVoxel(f[field].field->mipLevel(level)->mapping().get(), 
                         wsBounds, vsBounds);
          }
          dvsBounds = clipBounds(discreteBounds(vsBounds), dw);
          // If size of dvsBounds is <= 2, stop
          Imath::V3i size = dvsBounds.size();
          if (std::max(size.x, std::max(size.y, size.z)) <= 2) {
            break;
          }
        }
      }
      // Level chosen. Run loop
      for (int k = dvsBounds.min.z; k <= dvsBounds.max.z; ++k) {
        for (int j = dvsBounds.min.y; j <= dvsBounds.max.y; ++j) {
          for (int i = dvsBounds.min.x; i <= dvsBounds.max.x; ++i) {
            const Data_T val = f[field].field->fastMipValue(level, i, j, k);
            if (mode == Min) {
              *data = detail::min(val, *data);
            } else {
              *data = detail::max(val, *data);
            }
          }
        }
      }
    }
  }

};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//------------------------------------------------------------------------------

#endif // include guard

//------------------------------------------------------------------------------
