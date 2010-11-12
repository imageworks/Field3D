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

#ifndef _INCLUDED_Field3D_gpu_dense_field_test_H_
#define _INCLUDED_Field3D_gpu_dense_field_test_H_

#include "Field3D/Types.h"
#include <cutil_math.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>
#include <omp.h>

#include "Field3D/gpu/Timer.h"

#include "gpu_common.h"

//----------------------------------------------------------------------------//
// to be replaced by imath if it becomes cuda compatible
typedef float3 Vec3f;
#define Make_Vec3f make_float3

//----------------------------------------------------------------------------//
//! random sampling across entire field
template< typename INTERPOLATOR, typename ACCESSOR >
struct RandomSampleFunctor
{
  RandomSampleFunctor( ACCESSOR& _ac,
                       INTERPOLATOR _fn,
                       Vec3f* _p,
                       typename INTERPOLATOR::sample_type* _r )
  :ac(_ac), fn(_fn), p(_p), r(_r)
  {
  }

  __host__     __device__
  void operator()( int i )
  {
    fn.sample(ac, p[i], r[i]);
  }

  ACCESSOR& ac;
  INTERPOLATOR fn;
  Vec3f* p;
  typename INTERPOLATOR::sample_type* r;
};

//----------------------------------------------------------------------------//
//! random sampling across a voxel
template< typename INTERPOLATOR, typename ACCESSOR >
struct SuperSampleFunctor
{
  SuperSampleFunctor( ACCESSOR& _ac,
                      INTERPOLATOR _fn,
                      float* _urn,
                      int _urn_count,
                      typename INTERPOLATOR::sample_type* _r )
  :ac(_ac), fn(_fn), urn(_urn), urn_count(_urn_count), r(_r)
  {
    // per-axis super sample count
    sc = SUPER_SAMPLE_COUNT;

    // total number of super-samples
    sc3 = sc * sc * sc;
  }

  __host__   __device__
  void operator()(int i)
  {
    int3 voxelId = fn.getSampler().denseVoxelIndex(i);

    Vec3f voxel_min = Make_Vec3f(voxelId.x, voxelId.y, voxelId.z);
    Vec3f voxel_max = Make_Vec3f(voxelId.x + 1, voxelId.y + 1, voxelId.z + 1);
    Vec3f voxel_delta = voxel_max - voxel_min; // <<1,1,1>>

    // index into random table
    int u_idx = i * sc3 * 3;

#ifndef __CUDA_ARCH__
#if 0
    std::cout << "voxel: " << i << ", ";
    std::cout << "[ " << voxelId.x << ","
    << voxelId.y << ","
    << voxelId.z << "], ";
    std::cout << fn.getSampler().getIndex( voxelId.x, voxelId.y, voxelId.z )
    << std::endl;
    std::cout << "sc: " << sc << std::endl;
#endif
#endif
    // make sure 1d->3d mapping matches 3d->1d mapping
    //kernel_assert( i ==
    //  fn.getSampler().getIndex( voxelId.x, voxelId.y, voxelId.z ) );

    typename INTERPOLATOR::sample_type tally(0);

    // now supersample
    for (int z = 0; z < sc; ++z) {
      for (int y = 0; y < sc; ++y) {
        for (int x = 0; x < sc; ++x) {
          // std::cout << " super_sample("<< x << "," << y << "," << z << ") ";

          float rx = urn[u_idx++ % urn_count];
          float ry = urn[u_idx++ % urn_count];
          float rz = urn[u_idx++ % urn_count];

          float3 s = make_float3((x + rx) / sc, (y + ry) / sc, (z + rz) / sc);
          float3 sample_p = voxel_min + s * voxel_delta;

          // std::cout << " sample_p("<< sample_p.x << "," << sample_p.y
          // << "," << sample_p.z << ")" << std::endl;

          typename INTERPOLATOR::sample_type val;
          fn.sample(ac, sample_p, val);
          tally += val;
        }
      }
    }

    r[i] = tally / float(sc3);
  }

  int sc, sc3;
  ACCESSOR& ac;
  INTERPOLATOR fn;
  float* urn;
  int urn_count;
  typename INTERPOLATOR::sample_type* r;
};

//----------------------------------------------------------------------------//
//! profile a run on device
template< typename FUNCTOR >
inline float RunFunctor(FUNCTOR& f,
                        int count,
                        thrust::device_space_tag)
{
  thrust::counting_iterator<int, thrust::device_space_tag> first(0);
  thrust::counting_iterator<int, thrust::device_space_tag> last(count);

  Field3D::Gpu::GpuTimer t;
  thrust::for_each( first, last, f );
  return t.elapsed();
}

//----------------------------------------------------------------------------//
//! profile a run on the host
template< typename FUNCTOR >
inline float RunFunctor( FUNCTOR& f,
                         int sample_count,
                         thrust::host_space_tag,
                         bool use_openmp = true )
{
  thrust::counting_iterator<int, thrust::host_space_tag> first(0);
  thrust::counting_iterator<int, thrust::host_space_tag> last(sample_count);

  Field3D::Gpu::CpuTimer t;
  if ( use_openmp ) {
    _Pragma( "omp parallel for" )
    for( int i = 0; i < sample_count; ++i ){
      f(i);
    }
  } else {
    for( int i = 0; i < sample_count; ++i ) {
      f(i);
    }
  }

  return t.elapsed();
}

  //----------------------------------------------------------------------------//
  //! return the number of threads that will be processing data on cpu side
inline int threadCount()
{
  int result = 1;
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp master
    {
      result = omp_get_num_threads();
    }
  }
#endif
  return result;
}

#endif // Include guard
