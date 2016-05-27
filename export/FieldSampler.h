//----------------------------------------------------------------------------//

#ifndef __F3DUTIL_FIELDSAMPLER_H__
#define __F3DUTIL_FIELDSAMPLER_H__

//------------------------------------------------------------------------------

// Project includes
#include "Types.h"
#include "MIPField.h"

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

  //! Max operation on mixed types. This is used to compare
  //! individual max values.
  template <typename T, typename T2>
  T max(const T a, const T2 b)
  {
    return std::max(a, static_cast<T>(b));
  }

  //! Composite max operation on mixed types. This is used
  //! when combining overlapping max values
  template <typename T, typename T2>
  T compositeMax(const T a, const T2 b)
  {
    return std::max(std::max(a, static_cast<T>(b)), 
                    static_cast<T>(a + static_cast<T>(b)));
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

  //! Max operation on mixed vector types. This is used to compare
  //! individual max values.
  template <typename T, typename T2>
  FIELD3D_VEC3_T<T> max(const FIELD3D_VEC3_T<T> &a, 
                        const FIELD3D_VEC3_T<T2> &b)
  {
    return FIELD3D_VEC3_T<T>
      (std::max(a.x, static_cast<T>(b.x)),
       std::max(a.y, static_cast<T>(b.y)),
       std::max(a.z, static_cast<T>(b.z)));
  }

  //! Composite max operation on mixed vector types. This is used
  //! when combining overlapping max values
  template <typename T, typename T2>
  FIELD3D_VEC3_T<T> compositeMax(const FIELD3D_VEC3_T<T> &a, 
                                 const FIELD3D_VEC3_T<T2> &b)
  {
    return FIELD3D_VEC3_T<T>
      (std::max(std::max(a.x, static_cast<T>(b.x)), 
                static_cast<T>(a.x + static_cast<T>(b.x))),
       std::max(std::max(a.y, static_cast<T>(b.y)),
                static_cast<T>(a.y + static_cast<T>(b.y))),
       std::max(std::max(a.z, static_cast<T>(b.z)),
                static_cast<T>(a.z + static_cast<T>(b.z))));
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
// Arg structs
//------------------------------------------------------------------------------

struct SampleStochasticArgs
{
  // Ctor
  SampleStochasticArgs(const float *i_wsPs, 
                       const float *i_xiXs, 
                       const float *i_xiYs, 
                       const float *i_xiZs, 
                       float       *i_value, 
                       size_t      *i_numHits,
                       const float *i_active)
    : wsPs(i_wsPs), xiXs(i_xiXs), xiYs(i_xiYs), xiZs(i_xiZs), 
      value(i_value), numHits(i_numHits), active(i_active)
  { }
  // Data members
  const float *wsPs;
  const float *xiXs;
  const float *xiYs;
  const float *xiZs;
  float       *value;
  size_t      *numHits;
  const float *active;
};

//------------------------------------------------------------------------------

struct SampleMIPStochasticArgs
{
  // Ctor
  SampleMIPStochasticArgs(const float *i_wsPs, 
                          const float *i_wsSpotSizes,
                          const float *i_xiXs, 
                          const float *i_xiYs,
                          const float *i_xiZs, 
                          const float *i_xiSpotSizes,
                          float       *i_value, 
                          size_t      *i_numHits, 
                          const float *i_active)
    : wsPs(i_wsPs), wsSpotSizes(i_wsSpotSizes),
      xiXs(i_xiXs), xiYs(i_xiYs), xiZs(i_xiZs), xiSpotSizes(i_xiSpotSizes), 
      value(i_value), numHits(i_numHits), active(i_active)
  { }
  // Data members
  const float *wsPs; 
  const float *wsSpotSizes;
  const float *xiXs; 
  const float *xiYs;
  const float *xiZs; 
  const float *xiSpotSizes;
  float       *value; 
  size_t      *numHits; 
  const float *active;
};

//------------------------------------------------------------------------------

struct SampleTemporalStochasticArgs
{
  // Ctor
  SampleTemporalStochasticArgs(const float *i_wsPs, 
                               const float *i_wsSpotSizes,
                               const float *i_times, 
                               const float *i_xiXs, 
                               const float *i_xiYs,
                               const float *i_xiZs, 
                               const float *i_xiSpotSizes,
                               const float *i_xiTimes, 
                               float       *i_value, 
                               size_t      *i_numHits, 
                               const float *i_active)
    : wsPs(i_wsPs), wsSpotSizes(i_wsSpotSizes), times(i_times), 
      xiXs(i_xiXs), xiYs(i_xiYs), xiZs(i_xiZs), 
      xiSpotSizes(i_xiSpotSizes), xiTimes(i_xiTimes), 
      value(i_value), numHits(i_numHits), active(i_active)
  { }
  // Data members
  const float *wsPs; 
  const float *wsSpotSizes;
  const float *times;
  const float *xiXs; 
  const float *xiYs;
  const float *xiZs; 
  const float *xiSpotSizes;
  const float *xiTimes;
  float       *value; 
  size_t      *numHits; 
  const float *active;
};

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
                             const float *wsPs, float *value, size_t *numHits, 
                             const float *active)
  {
    // Loop over fields in vector
    for (size_t i = 0; i < f.size(); ++i) {
      const typename WrapperVec_T::value_type &field = f[i];

      // Reinterpret the pointer according to Dims_T
      Input_T *data = reinterpret_cast<Input_T*>(value);

      if (field.doOsToWs || field.valueRemapOp) {

        // Loop over samples
        for (size_t ieval = 0; ieval < neval; ++ieval) {
          if (!active || active[ieval]) {
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
              if (field.valueRemapOp) {
                const Data_T unremapped = field.interp.sample(*field.field, vsP);
                data[ieval] += field.valueRemapOp->remap(unremapped);
              } else {
                data[ieval] += field.interp.sample(*field.field, vsP);
              }
            }
          } 
        }

      } else {

        const Imath::Box3d &vsBounds_d = field.vsBounds;

        // Loop over samples
        for (size_t ieval = 0; ieval < neval; ++ieval) {
          if (!active || active[ieval]) {
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
          const Data_T unremapped = f[i].interp->sample(vsP, spotSize, 0.0f);
          *data += f[i].valueRemapOp->remap(unremapped);
        } else {
          *data += f[i].interp->sample(vsP, spotSize, 0.0f);
        }
      }
    }
  }

  // MIP fields
  static void sampleMIPMultiple(const WrapperVec_T &f, const size_t neval,
                                const float *wsPs, const float *wsSpotSizes,
                                float *value, size_t *numHits, const float *active)
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
            if (!active || active[ieval]) {
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
                  const Data_T unremapped = 
                    field.interp->sample(vsP, spotSize, 0.0f);
                  data[ieval] += field.valueRemapOp->remap(unremapped);
                }
              }
            }
          }

        } else {
          // Loop over samples
          for (size_t ieval = 0; ieval < neval; ++ieval) {
            if (!active || active[ieval]) {
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
                  const Data_T unremapped = 
                    field.interp->sample(vsP, spotSize, 0.0f);
                  *idata += field.valueRemapOp->remap(unremapped);
                } else {
                  *idata += field.interp->sample(vsP, spotSize, 0.0f);
                }
              }
            }
          }
        }
      } else {

        const Imath::Box3d &vsBounds_d = field.vsBounds;
        const double worldScale = field.worldScale;

        // Loop over samples
        for (size_t ieval = 0; ieval < neval; ++ieval) {
          if (!active || active[ieval]) {
            const V3d wsP(*reinterpret_cast<const V3f*>(wsPs + 3 * ieval));
            V3d vsP;

            // Apply world to object transform
            field.mapping->worldToVoxel(wsP, vsP);

            // Sample
            if (vsBounds_d.intersects(vsP)) {
              // Count as within field
              numHits[ieval]++;
              const double spotSize = wsSpotSizes[ieval] / worldScale;
              data[ieval] += field.interp->sample(vsP, spotSize, 0.0f);
            }
          }
        }
      }
    }
  }

  // Ordinary fields
  static void sampleStochastic(const WrapperVec_T &f, const size_t neval,
                               const SampleStochasticArgs &args)
  {
    // Loop over fields in vector
    for (size_t i = 0; i < f.size(); ++i) {
      const typename WrapperVec_T::value_type &field = f[i];

      // Reinterpret the pointer according to Dims_T
      Input_T *data = reinterpret_cast<Input_T*>(args.value);

      if (field.doOsToWs || field.valueRemapOp) {

        // Loop over samples
        for (size_t ieval = 0; ieval < neval; ++ieval) {
          if (!args.active || args.active[ieval]) {
            const V3d wsP(*reinterpret_cast<const V3f*>(args.wsPs + 3 * ieval));
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
              args.numHits[ieval]++;
              if (field.valueRemapOp) {
                const Data_T unremapped = 
                  field.stochasticInterp.linear(*field.field, vsP,
                                                args.xiXs[ieval], 
                                                args.xiYs[ieval],
                                                args.xiZs[ieval]);
                data[ieval] += field.valueRemapOp->remap(unremapped);
              } else {
                data[ieval] += 
                  field.stochasticInterp.linear(*field.field, 
                                                vsP,
                                                args.xiXs[ieval], 
                                                args.xiYs[ieval],
                                                args.xiZs[ieval]);
              }
            }
          } 
        }

      } else {

        const Imath::Box3d &vsBounds_d = field.vsBounds;

        // Loop over samples
        for (size_t ieval = 0; ieval < neval; ++ieval) {
          if (!args.active || args.active[ieval]) {
            const V3d wsP(*reinterpret_cast<const V3f*>(args.wsPs + 3 * ieval));
            V3d vsP;

            // Apply world to object transform
            field.mapping->worldToVoxel(wsP, vsP);
            // std::cout << "ws to vs " << wsP << " -> " << vsP << std::endl;

            // Sample
            if (vsBounds_d.intersects(vsP)) {
              // Count as within field
              args.numHits[ieval]++;
              // Sample
              data[ieval] += 
                field.stochasticInterp.linear(*field.field, vsP,
                                              args.xiXs[ieval], 
                                              args.xiYs[ieval],
                                              args.xiZs[ieval]);
            }
          }
        }
      }
    }
  }

  // MIP fields
  static void sampleMIPStochastic(const WrapperVec_T &f, const size_t neval,
                                  const SampleMIPStochasticArgs &args)
  {
    const bool doSpatialStochastic = args.xiXs && args.xiYs && args.xiZs;

    // Loop over fields in vector
    for (size_t i = 0; i < f.size(); ++i) {
      const typename WrapperVec_T::value_type &field = f[i];

      // Reinterpret the pointer according to Dims_T
      Input_T *data = reinterpret_cast<Input_T*>(args.value);

      if (field.doOsToWs || field.valueRemapOp) {

        if (field.valueRemapOp && field.doWsBoundsOptimization) {

          // Loop over samples
          for (size_t ieval = 0; ieval < neval; ++ieval) {
            if (!args.active || args.active[ieval]) {
              const V3f &wsP = 
                *reinterpret_cast<const V3f*>(args.wsPs + 3 * ieval);

              if (field.wsBounds.intersects(wsP)) {

                // Apply world to object transform
                V3d vsP;

                field.wsToVs.multVecMatrix(V3d(wsP), vsP);

                // Sample
                if (field.vsBounds.intersects(vsP)) {
                  // Count as within field
                  args.numHits[ieval]++;
                  const float spotSize = 
                    args.wsSpotSizes[ieval] / field.worldScale;
                  const Data_T unremapped = doSpatialStochastic ? 
                    field.stochasticInterp->linear(vsP, spotSize, 
                                                   args.xiXs[ieval], 
                                                   args.xiYs[ieval],
                                                   args.xiZs[ieval], 
                                                   args.xiSpotSizes[ieval]) :
                    field.stochasticInterp->linear(vsP, spotSize,
                                                   args.xiSpotSizes[ieval]);
                  data[ieval] += field.valueRemapOp->remap(unremapped);
                }
              }
            }
          }

        } else {
          // Loop over samples
          for (size_t ieval = 0; ieval < neval; ++ieval) {
            if (!args.active || args.active[ieval]) {
              const V3d wsP(*reinterpret_cast<const V3f*>
                            (args.wsPs + 3 * ieval));
              const float wsSpotSize = args.wsSpotSizes[ieval];
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
                args.numHits[ieval]++;
                if (field.valueRemapOp) {
                  const Data_T unremapped = doSpatialStochastic ? 
                    field.stochasticInterp->linear(vsP, spotSize, 
                                                   args.xiXs[ieval], 
                                                   args.xiYs[ieval],
                                                   args.xiZs[ieval], 
                                                   args.xiSpotSizes[ieval]) :
                    field.stochasticInterp->linear(vsP, spotSize, 
                                                   args.xiSpotSizes[ieval]);
                  *idata += field.valueRemapOp->remap(unremapped);
                } else {
                  *idata += doSpatialStochastic ? 
                    field.stochasticInterp->linear(vsP, spotSize, 
                                                   args.xiXs[ieval], 
                                                   args.xiYs[ieval],
                                                   args.xiZs[ieval], 
                                                   args.xiSpotSizes[ieval]) :
                    field.stochasticInterp->linear(vsP, spotSize, 
                                                   args.xiSpotSizes[ieval]);
                }
              }
            }
          }
        }

      } else {

        const Imath::Box3d &vsBounds_d = field.vsBounds;
        const double worldScale = field.worldScale;

        // Loop over samples
        for (size_t ieval = 0; ieval < neval; ++ieval) {
          if (!args.active || args.active[ieval]) {
            const V3d wsP(*reinterpret_cast<const V3f*>(args.wsPs + 3 * ieval));
            V3d vsP;

            // Apply world to object transform
            field.mapping->worldToVoxel(wsP, vsP);

            // Sample
            if (vsBounds_d.intersects(vsP)) {
              // Count as within field
              args.numHits[ieval]++;
              const double spotSize = args.wsSpotSizes[ieval] / worldScale;
              data[ieval] += doSpatialStochastic ? 
                field.stochasticInterp->linear(vsP, spotSize, 
                                               args.xiXs[ieval], 
                                               args.xiYs[ieval],
                                               args.xiZs[ieval], 
                                               args.xiSpotSizes[ieval]) :
                field.stochasticInterp->linear(vsP, spotSize, 
                                               args.xiSpotSizes[ieval]);
            }
          }
        }
      }
    }
  }

  static void sampleTemporal(const WrapperVec_T &f, const V3d &p,
                             const float t, float *value, size_t &numHits);

  static void sampleMIPTemporal(const WrapperVec_T &f, const V3d &p,
                                const float wsSpotSize, const float t, 
                                float *value, size_t &numHits);

  static void sampleTemporalMultiple(const WrapperVec_T &f,
                                     const size_t neval,
                                     const float *wsPs,
                                     const float *t,
                                     float *value,
                                     size_t *numHits, 
                                     const float *active = 0x0);

  static void sampleMIPTemporalMultiple(const WrapperVec_T &f,
                                        const size_t neval,
                                        const float *wsPs,
                                        const float *wsSpotSizes, 
                                        const float *t,
                                        float *value,
                                        size_t *numHits, 
                                        const float *active = 0x0);

  static void sampleTemporalStochastic(const WrapperVec_T &f,
                                       const size_t neval,
                                       const SampleTemporalStochasticArgs &a);
    
  static void sampleMIPTemporalStochastic(const WrapperVec_T &f,
                                          const size_t neval,
                                          const SampleTemporalStochasticArgs &a);

  // Get min/max
  static void getMinMax(const WrapperVec_T &f, 
                        const Box3d &wsBounds, float *min, float *max)
  {
    // Reinterpret the pointer according to Dims_T
    Input_T *minData = reinterpret_cast<Input_T*>(min);
    Input_T *maxData = reinterpret_cast<Input_T*>(max);
    // Loop over fields in vector
    for (size_t field = 0, end = f.size(); field < end; ++field) {
      // Store min/max for values in current field
      Input_T thisMin(std::numeric_limits<float>::max());
      Input_T thisMax(-std::numeric_limits<float>::max());
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
            // Sample and remap
            if (f[field].valueRemapOp) {
              const Data_T remapped = f[field].valueRemapOp->remap(val);
              thisMin = detail::min(thisMin, remapped);
              thisMax = detail::max(thisMax, remapped);
            } else {
              thisMin = detail::min(thisMin, val);
              thisMax = detail::max(thisMax, val);
            }
          }
        }
      }
      // With each iteration, update overlapping max
      *minData = detail::min(*minData, thisMin);
      *maxData = detail::compositeMax(*maxData, thisMax);
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
      // Store min/max for values in current field
      Input_T thisMin(std::numeric_limits<float>::max());
      Input_T thisMax(-std::numeric_limits<float>::max());
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
            // Sample and remap
            if (f[field].valueRemapOp) {
              const Data_T remapped = f[field].valueRemapOp->remap(val);
              thisMin = detail::min(thisMin, remapped);
              thisMax = detail::max(thisMax, remapped);
            } else {
              thisMin = detail::min(thisMin, val);
              thisMax = detail::max(thisMax, val);
            }
          }
        }
      }
      // With each iteration, update overlapping max
      *minData = detail::min(*minData, thisMin);
      *maxData = detail::compositeMax(*maxData, thisMax);
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
      // Store min/max for values in current field
      Input_T thisData(mode == Min ? 
        std::numeric_limits<float>::max() : -std::numeric_limits<float>::max());
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
            // Sample and remap
            if (f[field].valueRemapOp) {
              const Data_T remapped = f[field].valueRemapOp->remap(val);
              if (mode == Min) {
                thisData = detail::min(remapped, thisData);
              } else {
                thisData = detail::max(remapped, thisData);
              }
            } else {
              if (mode == Min) {
                thisData = detail::min(val, thisData);
              } else {
                thisData = detail::max(val, thisData);
              }
            }
          }
        }
      }
      // With each iteration, update overlapping max
      if (mode == Min) {
        *data = detail::min(*data, thisData);
      } else {
        *data = detail::compositeMax(*data, thisData);
      }
    }
  }

  static void getMinMaxTemporal(const WrapperVec_T &f, 
                                const Box3d &wsBounds, float *min, float *max);

  static void getMinWsVoxelSize(const WrapperVec_T &f, float &sizeMin);
};

//----------------------------------------------------------------------------//

template <typename WrapperVec_T, int Dims_T>
void FieldSampler<WrapperVec_T, Dims_T>::sampleTemporal
(const WrapperVec_T &f,
 const V3d &wsP,
 const float t,
 float *value, 
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
        const Data_T unremapped = f[i].interp.sample(*f[i].field, vsP, t);
        *data += f[i].valueRemapOp->remap(unremapped);
      } else {
        *data += f[i].interp.sample(*f[i].field, vsP, t);
      }
    } 
  }
}

//----------------------------------------------------------------------------//

template <typename WrapperVec_T, int Dims_T>
void FieldSampler<WrapperVec_T, Dims_T>::sampleMIPTemporal
(const WrapperVec_T &f,
 const V3d &wsP,
 const float wsSpotSize, 
 const float t,
 float *value, 
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
        const Data_T unremapped = 
          f[i].interp->sample(vsP, wsSpotSize, t);
        *data += f[i].valueRemapOp->remap(unremapped);
      } else {
        *data += f[i].interp->sample(vsP, wsSpotSize, t);
      }
    } 
  }
}

//----------------------------------------------------------------------------//

template <typename WrapperVec_T, int Dims_T>
void FieldSampler<WrapperVec_T, Dims_T>::sampleTemporalMultiple
(const WrapperVec_T &f,
 const size_t neval,
 const float *wsPs,
 const float *t,
 float *value,
 size_t *numHits, 
 const float *active)
{
  // Loop over fields in vector
  for (size_t i = 0; i < f.size(); ++i) {

    const typename WrapperVec_T::value_type &field = f[i];

    // Reinterpret the pointer according to Dims_T
    Input_T *data = reinterpret_cast<Input_T*>(value);

    if (field.doOsToWs || field.valueRemapOp) {

      // Loop over samples
      for (size_t ieval = 0; ieval < neval; ++ieval) {
        if (!active || active[ieval]) {
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
            if (field.valueRemapOp) {
              const Data_T unremapped = 
                field.interp.sample(*field.field, vsP, t[ieval]);
              data[ieval] += field.valueRemapOp->remap(unremapped);
            } else {
              data[ieval] += field.interp.sample(*field.field, vsP, t[ieval]);
            }
          }
        } 
      }

    } else {

      const Imath::Box3d &vsBounds_d = field.vsBounds;

      // Loop over samples
      for (size_t ieval = 0; ieval < neval; ++ieval) {
        if (!active || active[ieval]) {

          const V3d wsP(*reinterpret_cast<const V3f*>(wsPs + 3 * ieval));
          V3d vsP;

          // Apply world to object transform
          field.mapping->worldToVoxel(wsP, vsP);

          // Sample
          if (vsBounds_d.intersects(vsP)) {
            // Count as within field
            numHits[ieval]++;
            // Sample
            data[ieval] += field.interp.sample(*field.field, vsP, t[ieval]);
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------//

template <typename WrapperVec_T, int Dims_T>
void FieldSampler<WrapperVec_T, Dims_T>::sampleMIPTemporalMultiple
(const WrapperVec_T &f,
 const size_t neval,
 const float *wsPs,
 const float *wsSpotSizes, 
 const float *t,
 float *value,
 size_t *numHits, 
 const float *active)
{
  // Loop over fields in vector
  for (size_t i = 0; i < f.size(); ++i) {

    const typename WrapperVec_T::value_type &field = f[i];

    // Reinterpret the pointer according to Dims_T
    Input_T *data = reinterpret_cast<Input_T*>(value);

    if (field.doOsToWs || field.valueRemapOp) {

      // Loop over samples
      for (size_t ieval = 0; ieval < neval; ++ieval) {
        if (!active || active[ieval]) {
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
            if (field.valueRemapOp) {
              const Data_T unremapped = 
                field.interp->sample(vsP, wsSpotSizes[ieval], t[ieval]);
              data[ieval] += field.valueRemapOp->remap(unremapped);
            } else {
              data[ieval] += field.interp->sample(vsP, wsSpotSizes[ieval], 
                                                  t[ieval]);
            }
          }
        } 
      }

    } else {

      const Imath::Box3d &vsBounds_d = field.vsBounds;

      // Loop over samples
      for (size_t ieval = 0; ieval < neval; ++ieval) {
        if (!active || active[ieval]) {

          const V3d wsP(*reinterpret_cast<const V3f*>(wsPs + 3 * ieval));
          V3d vsP;

          // Apply world to object transform
          field.mapping->worldToVoxel(wsP, vsP);

          // Sample
          if (vsBounds_d.intersects(vsP)) {
            // Count as within field
            numHits[ieval]++;
            // Sample
            data[ieval] += field.interp.sample(*field.field, vsP, 
                                               wsSpotSizes[ieval], t[ieval]);
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------//

template <typename WrapperVec_T, int Dims_T>
void FieldSampler<WrapperVec_T, Dims_T>::sampleTemporalStochastic
(const WrapperVec_T &f,
 const size_t neval,
 const SampleTemporalStochasticArgs &args)
{
  // Reinterpret the pointer according to Dims_T
  Input_T *out = reinterpret_cast<Input_T*>(args.value);

  // Loop over fields in vector
  for (size_t iField = 0; iField < f.size(); ++iField) {

    const typename WrapperVec_T::value_type &field = f[iField];
      
    // Loop over samples
    for (size_t i = 0; i < neval; ++i) {
      if (!args.active || args.active[i]) {
        const V3d wsP(*reinterpret_cast<const V3f*>(args.wsPs + 3 * i));
        V3d vsP;

        field.mapping->worldToVoxel(wsP, vsP);
        if (field.vsBounds.intersects(vsP)) {
          out[i] += field.stochasticInterp.linear(*field.field, vsP, 
                                                  args.times[i], 
                                                  args.xiXs[i], 
                                                  args.xiYs[i],
                                                  args.xiZs[i]);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------//

template <typename WrapperVec_T, int Dims_T>
void FieldSampler<WrapperVec_T, Dims_T>::sampleMIPTemporalStochastic
(const WrapperVec_T &f,
 const size_t neval,
 const SampleTemporalStochasticArgs &args)
{
  // Reinterpret the pointer according to Dims_T
  Input_T *out = reinterpret_cast<Input_T*>(args.value);

  // Loop over fields in vector
  for (size_t iField = 0; iField < f.size(); ++iField) {

    const typename WrapperVec_T::value_type &field = f[iField];
      
    // Loop over samples
    for (size_t i = 0; i < neval; ++i) {
      if (!args.active || args.active[i]) {
        const V3d wsP(*reinterpret_cast<const V3f*>(args.wsPs + 3 * i));
        V3d vsP;

        field.mapping->worldToVoxel(wsP, vsP);
        if (field.vsBounds.intersects(vsP)) {
          out[i] += field.stochasticInterp.linear(*field.field, vsP, 
                                                  args.wsSpotSizes[i], 
                                                  args.times[i], 
                                                  args.xiXs[i], 
                                                  args.xiYs[i],
                                                  args.xiZs[i]);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
class TemporalField;

template <class Field_T>
class MIPField;

namespace detail {

  template <typename Data_T>
  const Data_T* sampleValues(const MIPField<TemporalField<Data_T> > &field,
                             const int i, const int j, const int k)
  {
    return field.rawMipLevel(0)->sampleValues(i, j, k);
  }

  template <typename Field_T>
  const typename Field_T::value_type* 
  sampleValues(const Field_T &field, const int i, const int j, const int k)
  {
    return field.sampleValues(i, j, k);
  }

  template <typename Data_T>
  size_t
  numSamples(const MIPField<TemporalField<Data_T> > &field,
             const int i, const int j, const int k)
  {
    return field.rawMipLevel(0)->numSamples(i, j, k);
  }

  template <typename Field_T>
  size_t 
  numSamples(const Field_T &field, const int i, const int j, const int k)
  {
    return field.numSamples(i, j, k);
  }

}

//----------------------------------------------------------------------------//

template <typename WrapperVec_T, int Dims_T>
void FieldSampler<WrapperVec_T, Dims_T>::getMinMaxTemporal
(const WrapperVec_T &f,
 const Box3d &wsBounds,
 float *min, float *max)
{
  Input_T *minData = reinterpret_cast<Input_T*>(min);
  Input_T *maxData = reinterpret_cast<Input_T*>(max);

  for (size_t field = 0, end = f.size(); field < end; ++field) {
    // Store min/max for values in current field
    Input_T thisMin(std::numeric_limits<float>::max());
    Input_T thisMax(-std::numeric_limits<float>::max());
    // Data window
    const Box3i dw = f[field].field->dataWindow();
    // Transform corners to voxel space and compute bounds
    Box3i dvsBounds;
    if (wsBounds.isInfinite()) {
      dvsBounds = dw;
    } else {
      Box3d vsBounds;
      worldToVoxel(f[field].mapping, wsBounds, vsBounds);
      dvsBounds = clipBounds(discreteBounds(vsBounds), dw);
      // Early termination if no intersection
      if (!dw.intersects(dvsBounds)) {
        return;
      }
    }
    for (int k = dvsBounds.min.z; k <= dvsBounds.max.z; ++k) {
      for (int j = dvsBounds.min.y; j <= dvsBounds.max.y; ++j) {
        for (int i = dvsBounds.min.x; i <= dvsBounds.max.x; ++i) {
          
          const Data_T *sampleValues = 
            detail::sampleValues(*f[field].field, i, j, k);
          const size_t sEnd = 
            detail::numSamples(*f[field].field, i, j, k);
          
          for (size_t s = 0; s < sEnd; ++s) {
            // Output is cast to float
            const Input_T val = static_cast<Input_T>(sampleValues[s]);
            thisMin = detail::min(val, thisMin);
            thisMax = detail::max(val, thisMax);
          }
        }
      }
    }
    // With each iteration, update overlapping max
    *minData = detail::min(*minData, thisMin);
    *maxData = detail::compositeMax(*maxData, thisMax);
  }
}

//----------------------------------------------------------------------------//

template <typename WrapperVec_T, int Dims_T>
void FieldSampler<WrapperVec_T, Dims_T>::getMinWsVoxelSize
(const WrapperVec_T &f,
 float &sizeMin)
{
  for (size_t field = 0, end = f.size(); field < end; ++field) {

    const Field3D::FieldMapping *mapping = f[field].mapping;
    const V3d vs3f = mapping->wsVoxelSize(0,0,0);
    const float voxelsize = std::min(std::min(vs3f.x, vs3f.y), vs3f.z);

    sizeMin = std::min(sizeMin, voxelsize);
  }
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//------------------------------------------------------------------------------

#endif // include guard

//------------------------------------------------------------------------------
