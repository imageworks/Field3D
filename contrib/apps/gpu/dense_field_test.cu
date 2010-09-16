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

#include "dense_field_test.h"

#include <iostream>

#include <thrust/device_vector.h>

#include <boost/shared_ptr.hpp>

#include "Field3D/gpu/FieldInterpCuda.h"
#include "Field3D/gpu/Traits.h"
#include "Field3D/gpu/Timer.h"

using namespace Field3D::Gpu;

// actual texture declarations have to be at file scope
GpuFieldTraits< double >::cuda_tex_type tex_double;
GpuFieldTraits< float >::cuda_tex_type tex_float;
GpuFieldTraits< Field3D::half >::cuda_tex_type tex_half;

//----------------------------------------------------------------------------//
//! access via a cuda texture
template< typename T >
struct TexAccessor : public DataAccessor
{};


//----------------------------------------------------------------------------//
//! specialization for double
template< >
struct TexAccessor< double >
{
	typedef GpuFieldTraits< double >::cuda_tex_value_type cuda_tex_value_type;

	__host__  __device__
	double operator()( 	int i, double* phi )
	{
#ifdef __CUDA_ARCH__
		int2 v = tex1Dfetch( tex_double, i );
		return __hiloint2double(v.y, v.x);
#else
		return 0.0f;
#endif
	}

	GpuFieldTraits< double >::cuda_tex_type& tex()
	{
		return tex_double;
	}
};

//----------------------------------------------------------------------------//
//! specialization for float
template< >
struct TexAccessor< float >
{
	typedef GpuFieldTraits< float >::cuda_tex_value_type cuda_tex_value_type;

	__host__  __device__
	float operator()( 	int i, float* phi )
	{
#ifdef __CUDA_ARCH__
		return tex1Dfetch(tex_float, i);
#else
		return 0.0f;
#endif
	}

	GpuFieldTraits< float >::cuda_tex_type& tex()
	{
		return tex_float;
	}
};

//----------------------------------------------------------------------------//
//! specialization for half
template<>
struct TexAccessor< Field3D::half >
{
	typedef GpuFieldTraits< Field3D::half >::cuda_value_type cuda_value_type;
	typedef GpuFieldTraits< Field3D::half >::cuda_tex_value_type cuda_tex_value_type;

	__host__ __device__
	float operator()( int i, cuda_value_type* phi )
	{
#ifdef __CUDA_ARCH__
		short v = tex1Dfetch( tex_half, i );
		return __half2float( v );
#else
		return 0.0f;
#endif
	}

	GpuFieldTraits< Field3D::half >::cuda_tex_type& tex()
	{
		return tex_half;
	}
};


namespace nvcc {

	//----------------------------------------------------------------------------//
	template< typename INTERP >
	void testHost( thrust::host_vector<Vec3f>& host_p, INTERP& interp, bool dump_result )
	{
		GlobalMemAccessor< typename INTERP::value_type > ac;

		int sample_count = host_p.size();

		thrust::host_vector< typename INTERP::sample_type > host_result( sample_count, 0.0f );
		SampleFunctor< INTERP, GlobalMemAccessor< typename INTERP::value_type > > f( ac, interp, &host_p[0], &host_result[0] );

		CpuTimer t;

		thrust::counting_iterator< int, thrust::host_space_tag > first( 0 );
		thrust::counting_iterator< int, thrust::host_space_tag > last( sample_count );

#ifdef _OPENMP
		_Pragma( "omp parallel for" )
		for( int i = 0; i < sample_count; ++i ){
			f(i);
		}
#else
		thrust::for_each( first, last, f );
#endif

		float et = t.elapsed();

		if( dump_result ){
			std::cout << "    nvcc host result        : ";
			dump( host_result );
		} else {
			std::cout << "    nvcc host et: " << et << std::endl;
		}
	}

	//----------------------------------------------------------------------------//
	template< typename INTERP >
	void testDevice( thrust::host_vector<Vec3f>& host_p, INTERP& interp, bool dump_result )
	{
		GlobalMemAccessor< typename INTERP::value_type > ac;

		int sample_count = host_p.size();
		thrust::device_vector<Vec3f> device_p = host_p;

		thrust::device_vector< typename INTERP::sample_type > device_result( sample_count, 0.0f );
		SampleFunctor< INTERP, GlobalMemAccessor< typename INTERP::value_type > > f( ac, interp, thrust::raw_pointer_cast( &device_p[ 0 ] ), thrust::raw_pointer_cast( &device_result[ 0 ] ) );

		GpuTimer t;

		thrust::counting_iterator< int, thrust::device_space_tag > first( 0 );
		thrust::counting_iterator< int, thrust::device_space_tag > last( sample_count );
		thrust::for_each( first, last, f );

		cudaThreadSynchronize(); // as we're profiling
		float et = t.elapsed();

		if( dump_result ){
			std::cout << "    nvcc device result      : ";
			thrust::host_vector< typename INTERP::sample_type > host_result = device_result;
			dump( host_result );
		} else {
			std::cout << "    nvcc device et: " << et << std::endl;
		}
	}

	//----------------------------------------------------------------------------//
	template< typename INTERP >
	void testTexDevice( thrust::host_vector<Vec3f>& host_p, INTERP& interp, bool dump_result )
	{
		TexAccessor< typename INTERP::value_type > ac;

		interp.bindTex( ac );

		int sample_count = host_p.size();
		thrust::device_vector<Vec3f> device_p = host_p;

		thrust::device_vector< typename INTERP::sample_type > device_result( sample_count, 0.0f );
		SampleFunctor< INTERP, TexAccessor< typename INTERP::value_type > > f( ac, interp, thrust::raw_pointer_cast( &device_p[ 0 ] ), thrust::raw_pointer_cast( &device_result[ 0 ] ) );

		GpuTimer t;

		thrust::counting_iterator< int, thrust::device_space_tag > first( 0 );
		thrust::counting_iterator< int, thrust::device_space_tag > last( sample_count );
		thrust::for_each( first, last, f );

		cudaThreadSynchronize(); // as we're profiling
		float et = t.elapsed();

		if( dump_result ){
			std::cout << "    nvcc device (tex) result: ";
			thrust::host_vector< typename INTERP::sample_type > host_result = device_result;
			dump( host_result );
		} else {
			std::cout << "    nvcc device (tex) et: " << et << std::endl;
		}

		interp.unbindTex( ac );
	}

	// double instantiation
	template void testHost< LinearFieldInterp< DenseFieldSampler< double, double > > > ( thrust::host_vector<Vec3f>&, LinearFieldInterp< DenseFieldSampler< double, double > >&, bool );
	template void testDevice< LinearFieldInterp< DenseFieldSampler< double, double > > > ( thrust::host_vector<Vec3f>&, LinearFieldInterp< DenseFieldSampler< double, double > >&, bool );
	template void testTexDevice< LinearFieldInterp< DenseFieldSampler< double, double > > > ( thrust::host_vector<Vec3f>&, LinearFieldInterp< DenseFieldSampler< double, double > >&, bool );

	// float instantiation
	template void testHost< LinearFieldInterp< DenseFieldSampler< float, float > > > ( thrust::host_vector<Vec3f>&, LinearFieldInterp< DenseFieldSampler< float, float > >&, bool );
	template void testDevice< LinearFieldInterp< DenseFieldSampler< float, float > > > ( thrust::host_vector<Vec3f>&, LinearFieldInterp< DenseFieldSampler< float, float > >&, bool );
	template void testTexDevice< LinearFieldInterp< DenseFieldSampler< float, float > > > ( thrust::host_vector<Vec3f>&, LinearFieldInterp< DenseFieldSampler< float, float > >&, bool );

	// half instantiation
	template void testHost< LinearFieldInterp< DenseFieldSampler< Field3D::half, float > > > ( thrust::host_vector<Vec3f>&, LinearFieldInterp< DenseFieldSampler< Field3D::half, float > >&, bool );
	template void testDevice< LinearFieldInterp< DenseFieldSampler< Field3D::half, float > > > ( thrust::host_vector<Vec3f>&, LinearFieldInterp< DenseFieldSampler< Field3D::half, float > >&, bool );
	template void testTexDevice< LinearFieldInterp< DenseFieldSampler< Field3D::half, float > > > ( thrust::host_vector<Vec3f>&, LinearFieldInterp< DenseFieldSampler< Field3D::half, float > >&, bool );
}

