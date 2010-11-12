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


#include "Field3D/gpu/DeviceInfo.h"
#include "cuda_field_test.h"

#include <iostream>

#include "Field3D/gpu/DenseFieldSamplerCuda.h"
#include "Field3D/gpu/SparseFieldSamplerCuda.h"
#include "Field3D/gpu/FieldInterpCuda.h"
#include "Field3D/gpu/DataAccessorCuda.h"
#include "Field3D/gpu/Traits.h"

#include <thrust/device_vector.h>

using namespace Field3D::Gpu;

// actual texture declarations have to be at file scope
GpuFieldTraits< double >::cuda_tex_type tex_double;
GpuFieldTraits<float>::cuda_tex_type tex_float;
GpuFieldTraits< Field3D::half >::cuda_tex_type tex_half;

//----------------------------------------------------------------------------//
//! access via a cuda texture
template< typename T >
struct TexAccessor: public DataAccessor
{
};

//----------------------------------------------------------------------------//
//! specialization for double
template<>
struct TexAccessor<double>
{
  typedef GpuFieldTraits<double>::cuda_tex_value_type cuda_tex_value_type;

  __host__   __device__
  double operator()(int i, double* phi)
  {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 130
    // double support only for compute capability >= 1.3
    int2 v = tex1Dfetch(tex_double, i);
    return __hiloint2double(v.y, v.x);
#else
    // otherwise return 0
    return 0;
#endif
#else
    assert( 0 );
    return 0;
#endif
  }

  GpuFieldTraits<double>::cuda_tex_type& getTex()
  {
    return tex_double;
  }
};

//----------------------------------------------------------------------------//
//! specialization for float
template<>
struct TexAccessor<float>
{
  typedef GpuFieldTraits<float>::cuda_tex_value_type cuda_tex_value_type;

  __host__   __device__
  float operator()(int i, float* phi)
  {
#ifdef __CUDA_ARCH__
    return tex1Dfetch(tex_float, i);
#else
    assert( 0 );
    return 0;
#endif
  }

  GpuFieldTraits<float>::cuda_tex_type& getTex()
  {
    return tex_float;
  }
};

//----------------------------------------------------------------------------//
//! specialization for half
template<>
struct TexAccessor<Field3D::half>
{
  typedef GpuFieldTraits<Field3D::half>::cuda_value_type cuda_value_type;
  typedef GpuFieldTraits<Field3D::half>::cuda_tex_value_type
      cuda_tex_value_type;

  __host__  __device__
  float operator()(int i, cuda_value_type* phi)
  {
#ifdef __CUDA_ARCH__
    return __half2float(tex1Dfetch(tex_half, i));
#else
    assert( 0 );
    return 0;
#endif
  }

  GpuFieldTraits<Field3D::half>::cuda_tex_type& getTex()
  {
    return tex_half;
  }
};

namespace nvcc
{

  //--------------------------------------------------------------------------//
  template<typename INTERP>
  void testHost( thrust::host_vector<Vec3f>& host_p,
                 thrust::host_vector<float>& host_urn,
                 INTERP& interp,
                 bool dump_result )
  {
    {
      GlobalMemAccessor< typename INTERP::value_type > ac;

      int sample_count = host_p.size();

      thrust::host_vector< typename INTERP::sample_type >
        host_result( sample_count, 0.0f );
      RandomSampleFunctor< INTERP,
        GlobalMemAccessor< typename INTERP::value_type > >
        f( ac, interp, &host_p[ 0 ], &host_result[ 0 ] );

      float et = RunFunctor( f, sample_count, thrust::host_space_tag() );

      if ( dump_result ) {
        std::cout << "    nvcc host result        : ";
        dump( host_result );
      } else {
        std::cout << "    nvcc host:\t\trandom sample: " << et << "\t"
            << std::flush;
      }
    }
    if ( !dump_result ) {
      GlobalMemAccessor< typename INTERP::value_type > ac;
      int sample_count = interp.getSampler().dataWindowVoxelCount();
      thrust::host_vector< typename INTERP::sample_type >
        host_result( sample_count, 0.0f );
      SuperSampleFunctor< INTERP,
        GlobalMemAccessor< typename INTERP::value_type > >
        f( ac, interp, &host_urn[ 0 ], host_urn.size(), &host_result[ 0 ] );

      float et = RunFunctor( f, sample_count, thrust::host_space_tag() );

      std::cout << "super sample: " << et << std::endl;
    }
  }

  //--------------------------------------------------------------------------//
  template< typename INTERP >
  void testDevice( thrust::host_vector<Vec3f>& host_p,
                   thrust::host_vector<float>& host_urn,
                   INTERP& interp,
                   bool dump_result )
  {
    {
      GlobalMemAccessor<typename INTERP::value_type> ac;

      int sample_count = host_p.size();
      thrust::device_vector<Vec3f> device_p = host_p;

      thrust::device_vector<typename INTERP::sample_type>
        device_result( sample_count, 0.0f );
      RandomSampleFunctor< INTERP,
        GlobalMemAccessor<typename INTERP::value_type> >
        f( ac, interp, thrust::raw_pointer_cast( &device_p[ 0 ] ),
            thrust::raw_pointer_cast( &device_result[ 0 ] ) );

      float et = RunFunctor(f, sample_count, thrust::device_space_tag());

      if (dump_result) {
        std::cout << "    nvcc device result      : ";
        thrust::host_vector< typename INTERP::sample_type >
          host_result = device_result;
        dump( host_result );
      } else {
        std::cout << "    nvcc device:\trandom sample: " << et << "\t"
            << std::flush;
      }
    }
    if ( !dump_result )
    {
      GlobalMemAccessor< typename INTERP::value_type > ac;
      int sample_count = interp.getSampler().dataWindowVoxelCount();
      thrust::device_vector<float> device_urn = host_urn;
      thrust::device_vector< typename INTERP::sample_type >
        device_result( sample_count, 0.0f );

      SuperSampleFunctor< INTERP,
        GlobalMemAccessor< typename INTERP::value_type > >
        f( ac,
           interp,
           thrust::raw_pointer_cast( &device_urn[ 0 ] ),
           device_urn.size(),
           thrust::raw_pointer_cast( &device_result[ 0 ] ) );

      float et = RunFunctor( f, sample_count, thrust::device_space_tag() );

      std::cout << "super sample: " << et << std::endl;
    }
  }

  //----------------------------------------------------------------------------//
  template< typename INTERP >
  void testTexDevice( thrust::host_vector<Vec3f>& host_p,
                      thrust::host_vector<float>& host_urn,
                      INTERP& interp,
                      bool dump_result )
  {
    {
      TexAccessor< typename INTERP::value_type > ac;
      interp.bindTex( ac );

      int sample_count = host_p.size();
      thrust::device_vector<Vec3f> device_p = host_p;

      thrust::device_vector< typename INTERP::sample_type >
      device_result( sample_count, 0.0f );

      RandomSampleFunctor< INTERP, TexAccessor< typename INTERP::value_type > >
          f( ac,
             interp,
             thrust::raw_pointer_cast( &device_p[ 0 ] ),
             thrust::raw_pointer_cast( &device_result[ 0 ] ) );

      float et = RunFunctor( f, sample_count, thrust::device_space_tag() );

      if ( dump_result ) {
        std::cout << "    nvcc device (tex) result: ";
        thrust::host_vector< typename INTERP::sample_type >
          host_result = device_result;
        dump( host_result );
      } else {
        std::cout << "    nvcc device (tex):\trandom sample: " << et << "\t";
      }

      interp.unbindTex( ac );
    }
    if ( !dump_result )
    {
      TexAccessor< typename INTERP::value_type > ac;
      interp.bindTex( ac );

      int sample_count = interp.getSampler().dataWindowVoxelCount();
      thrust::device_vector<float> device_urn = host_urn;
      thrust::device_vector< typename INTERP::sample_type >
        device_result( sample_count, 0.0f );

      SuperSampleFunctor< INTERP, TexAccessor< typename INTERP::value_type > >
          f( ac,
             interp,
             thrust::raw_pointer_cast( &device_urn[ 0 ] ),
             device_urn.size(),
             thrust::raw_pointer_cast( &device_result[ 0 ] ) );

      float et = RunFunctor( f, sample_count, thrust::device_space_tag() );

      std::cout << "super sample: " << et << std::endl;

      interp.unbindTex( ac );
    }
  }

  //--------------------------------------------------------------------------//
  // double instantiation
  template
  void testHost< LinearFieldInterp< DenseFieldSampler<double, double> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp<DenseFieldSampler<double, double > >&,
    bool );

  template
  void testDevice< LinearFieldInterp< DenseFieldSampler<double, double> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< DenseFieldSampler<double, double> >&,
    bool );

  template
  void testTexDevice< LinearFieldInterp< DenseFieldSampler<double,double> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< DenseFieldSampler<double, double> >&,
    bool );

  template
  void testHost< LinearFieldInterp< SparseFieldSampler<double, double> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< SparseFieldSampler<double, double> >&,
    bool );

  template
  void testDevice< LinearFieldInterp< SparseFieldSampler<double, double> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< SparseFieldSampler<double, double> >&,
    bool );

  template
  void testTexDevice< LinearFieldInterp< SparseFieldSampler<double,double> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< SparseFieldSampler<double, double> >&,
    bool );

  //--------------------------------------------------------------------------//
  // float instantiation
  template
  void testHost< LinearFieldInterp< DenseFieldSampler<float, float> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< DenseFieldSampler<float, float> >&,
    bool );

  template
  void testDevice< LinearFieldInterp< DenseFieldSampler<float, float> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< DenseFieldSampler<float, float> >&,
    bool );

  template
  void testTexDevice< LinearFieldInterp< DenseFieldSampler<float, float> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< DenseFieldSampler<float, float> >&,
    bool );

  template
  void testHost< LinearFieldInterp< SparseFieldSampler<float, float> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< SparseFieldSampler<float, float> >&,
    bool );

  template
  void testDevice< LinearFieldInterp< SparseFieldSampler<float, float> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< SparseFieldSampler<float, float> >&,
    bool );

  template
  void testTexDevice< LinearFieldInterp< SparseFieldSampler<float, float> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< SparseFieldSampler<float, float> >&,
    bool );

  //--------------------------------------------------------------------------//
  // half instantiation
  template
  void testHost<
    LinearFieldInterp< DenseFieldSampler<Field3D::half, float> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< DenseFieldSampler<Field3D::half, float> >&,
    bool );

  template
  void testDevice<
    LinearFieldInterp< DenseFieldSampler<Field3D::half, float> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< DenseFieldSampler<Field3D::half, float> >&,
    bool );

  template
  void testTexDevice<
    LinearFieldInterp< DenseFieldSampler<Field3D::half, float> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< DenseFieldSampler<Field3D::half, float> >&,
    bool );

  template
  void testHost<
    LinearFieldInterp< SparseFieldSampler<Field3D::half, float> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< SparseFieldSampler<Field3D::half, float> >&,
    bool );

  template
  void testDevice<
    LinearFieldInterp< SparseFieldSampler<Field3D::half, float> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< SparseFieldSampler<Field3D::half, float> >&,
    bool );

  template
  void testTexDevice<
    LinearFieldInterp< SparseFieldSampler<Field3D::half, float> > >
  ( thrust::host_vector<Vec3f>&,
    thrust::host_vector<float>&,
    LinearFieldInterp< SparseFieldSampler<Field3D::half, float> >&,
    bool );

}

