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
// FieldSampler
//------------------------------------------------------------------------------

//! Interface for sampling a vector of fields of the same type
template <typename WrapperVec_T, int Dims_T>
struct FieldSampler
{
  static void sample(const WrapperVec_T &f, const V3d &p, float *value, 
                     bool isVs);
  static void sampleMIP(const WrapperVec_T &f, const V3d &p,
                        const float wsSpotSize, float *value, bool isVs);
  static void getMinMax(const WrapperVec_T &f, 
                        const Box3d &wsBounds, float *min, float *max);
  static void getMinMaxMIP(const WrapperVec_T &f, 
                           const Box3d &wsBounds, float *min, float *max);
};

//------------------------------------------------------------------------------

//! Scalar specialization
template <typename WrapperVec_T>
struct FieldSampler<WrapperVec_T, 1>
{
  // Ordinary fields
  static void sample(const WrapperVec_T &f, const V3d &p, float *value, 
                     bool isVs)
  {
    if (isVs) {
      for (size_t i = 0, end = f.size(); i < end; ++i) {
        if (f[i].vsBounds.intersects(p)) {
          *value += f[i].interp.sample(*f[i].field, p);
        }
      }
    } else {
      V3d vsP;
      for (size_t i = 0, end = f.size(); i < end; ++i) {
        f[i].mapping->worldToVoxel(p, vsP);
        if (f[i].vsBounds.intersects(vsP)) {
          *value += f[i].interp.sample(*f[i].field, vsP);
        } 
      }
    }
  }
  // MIP fields
  static void sampleMIP(const WrapperVec_T &f, const V3d &p,
                        const float wsSpotSize, float *value, bool isVs)
  {
    if (isVs) {
      for (size_t i = 0, end = f.size(); i < end; ++i) {
        if (f[i].vsBounds.intersects(p)) {
          *value += f[i].interp->sample(p, wsSpotSize);
        }
      }
    } else {
      V3d vsP;
      for (size_t i = 0, end = f.size(); i < end; ++i) {
        f[i].mapping->worldToVoxel(p, vsP);
        if (f[i].vsBounds.intersects(vsP)) {
          *value += f[i].interp->sample(vsP, wsSpotSize);
        }
      }
    }
  }
  // Get min/max
  static void getMinMax(const WrapperVec_T &f, 
                        const Box3d &wsBounds, float *min, float *max)
  {
    for (size_t field = 0, end = f.size(); field < end; ++field) {
      // Data window
      const Box3i dw = f[field].field->dataWindow();
      // Transform corners to voxel space and compute bounds
      Box3d vsBounds;
      worldToVoxel(f[field].mapping, wsBounds, vsBounds);
      Box3i dvsBounds = clipBounds(discreteBounds(vsBounds), dw);
      // Early termination if no intersection
      if (!dw.intersects(dvsBounds)) {
        return;
      }
      for (int k = dvsBounds.min.z; k <= dvsBounds.max.z; ++k) {
        for (int j = dvsBounds.min.y; j <= dvsBounds.max.y; ++j) {
          for (int i = dvsBounds.min.x; i <= dvsBounds.max.x; ++i) {
            float val = f[field].field->fastValue(i, j, k);
            min[0] = std::min(val, min[0]);
            max[0] = std::max(val, max[0]);
          }
        }
      }
    }
  }
  // Get min/max
  static void getMinMaxMIP(const WrapperVec_T &f, 
                           const Box3d &wsBounds, float *min, float *max)
  {
    for (size_t field = 0, end = f.size(); field < end; ++field) {
      // Data window
      const Box3i dw = f[field].field->dataWindow();
      // Transform corners to voxel space and compute bounds
      Box3d vsBounds;
      worldToVoxel(f[field].mapping, wsBounds, vsBounds);
      Box3i dvsBounds = clipBounds(discreteBounds(vsBounds), dw);
      // Early termination if no intersection
      if (!dw.intersects(dvsBounds)) {
        return;
      }
      for (int k = dvsBounds.min.z; k <= dvsBounds.max.z; ++k) {
        for (int j = dvsBounds.min.y; j <= dvsBounds.max.y; ++j) {
          for (int i = dvsBounds.min.x; i <= dvsBounds.max.x; ++i) {
            float val = f[field].field->fastMipValue(0, i, j, k);
            min[0] = std::min(val, min[0]);
            max[0] = std::max(val, max[0]);
          }
        }
      }
    }
  }
};

//------------------------------------------------------------------------------

//! Vector specialization
template <typename WrapperVec_T>
struct FieldSampler<WrapperVec_T, 3>
{
  // Ordinary fields
  static void sample(const WrapperVec_T &f, const V3d &p, float *value, 
                     bool isVs)
  {
    V3f v(value[0], value[1], value[2]);
    if (isVs) {
      for (size_t i = 0, end = f.size(); i < end; ++i) {
        if (f[i].vsBounds.intersects(p)) {
          v += f[i].interp.sample(*f[i].field, p);
        }
      }
    } else {
      V3d vsP;
      for (size_t i = 0, end = f.size(); i < end; ++i) {
        f[i].mapping->worldToVoxel(p, vsP);
        if (f[i].vsBounds.intersects(vsP)) {
          v += f[i].interp.sample(*f[i].field, vsP);
        }
      }
    }
    memcpy(value, &v[0], sizeof(V3f));
  }

  // MIP fields
  static void sampleMIP(const WrapperVec_T &f, const V3d &p,
                        const float wsSpotSize, float *value, bool isVs)
  {
    V3f v(value[0], value[1], value[2]);
    if (isVs) {
      for (size_t i = 0, end = f.size(); i < end; ++i) {
        if (f[i].vsBounds.intersects(p)) {
          v += f[i].interp->sample(p, wsSpotSize);
        }
      }
    } else {
      V3d vsP;
      for (size_t i = 0, end = f.size(); i < end; ++i) {
        f[i].mapping->worldToVoxel(p, vsP);
        if (f[i].vsBounds.intersects(vsP)) {
          v += f[i].interp->sample(vsP, wsSpotSize);
        }
      }
    }
    memcpy(value, &v[0], sizeof(V3f));    
  }

  static void getMinMax(const WrapperVec_T &f, 
                        const Box3d &wsBounds, float *min, float *max)
  {
    for (size_t field = 0, end = f.size(); field < end; ++field) {
      // Data window
      const Box3i dw = f[field].field->dataWindow();
      // Transform corners to voxel space and compute bounds
      Box3d vsBounds;
      worldToVoxel(f[field].mapping, wsBounds, vsBounds);
      Box3i dvsBounds = clipBounds(discreteBounds(vsBounds), dw);
      // Early termination if no intersection
      if (!dw.intersects(dvsBounds)) {
        return;
      }
      for (int k = dvsBounds.min.z; k <= dvsBounds.max.z; ++k) {
        for (int j = dvsBounds.min.y; j <= dvsBounds.max.y; ++j) {
          for (int i = dvsBounds.min.x; i <= dvsBounds.max.x; ++i) {
            V3f val = f[field].field->fastValue(i, j, k);
            min[0] = std::min(val.x, min[0]);
            min[1] = std::min(val.y, min[1]);
            min[2] = std::min(val.z, min[2]);
            max[0] = std::max(val.x, max[0]);
            max[1] = std::max(val.y, max[1]);
            max[2] = std::max(val.z, max[2]);
          }
        }
      }
    }
  }

  static void getMinMaxMIP(const WrapperVec_T &f, 
                           const Box3d &wsBounds, float *min, float *max)
  {
    for (size_t field = 0, end = f.size(); field < end; ++field) {
      // Data window
      const Box3i dw = f[field].field->dataWindow();
      // Transform corners to voxel space and compute bounds
      Box3d vsBounds;
      worldToVoxel(f[field].mapping, wsBounds, vsBounds);
      Box3i dvsBounds = clipBounds(discreteBounds(vsBounds), dw);
      // Early termination if no intersection
      if (!dw.intersects(dvsBounds)) {
        return;
      }
      for (int k = dvsBounds.min.z; k <= dvsBounds.max.z; ++k) {
        for (int j = dvsBounds.min.y; j <= dvsBounds.max.y; ++j) {
          for (int i = dvsBounds.min.x; i <= dvsBounds.max.x; ++i) {
            V3f val = f[field].field->fastMipValue(0, i, j, k);
            min[0] = std::min(val.x, min[0]);
            min[1] = std::min(val.y, min[1]);
            min[2] = std::min(val.z, min[2]);
            max[0] = std::max(val.x, max[0]);
            max[1] = std::max(val.y, max[1]);
            max[2] = std::max(val.z, max[2]);
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
