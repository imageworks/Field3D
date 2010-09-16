//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks Inc
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

#ifndef _INCLUDED_Field3D_gpu_FieldInterpCuda_H_
#define _INCLUDED_Field3D_gpu_FieldInterpCuda_H_

// field3d includes
#include "Field3D/ns.h"
#include "Field3D/gpu/Traits.h"
#include "OpenEXR/half.h"

// cuda includes
#include <cutil_math.h>
#include <device_functions.h>

FIELD3D_NAMESPACE_OPEN

#ifdef NVCC
#if CUDART_VERSION < 3010
#error requires cuda version >= 3.1 for half float intrinsics
#endif
#endif

namespace Gpu {


//----------------------------------------------------------------------------//
//! 1d access to data
 struct DataAccessor
{};

//----------------------------------------------------------------------------//
//! access data directly from cuda global memory, or host memory
template< typename T >
struct GlobalMemAccessor : public DataAccessor
{
	inline
	__host__ __device__
	T operator()( int i, T* phi )
	{
		return phi[i];
	}
};

//----------------------------------------------------------------------------//
//! specialization of GlobalMemAccessor for half float
template<>
struct GlobalMemAccessor<Field3D::half>
{
	inline
	__host__ __device__
	float operator()( int i, short* phi )
	{
#ifdef __CUDA_ARCH__
	// half to float intrinsic is only available in device code
	return __half2float( phi[i] );
#else
	// ILM's half to float conversion is only available in host code
	return *(half*)&phi[i];
#endif
	}
};


//----------------------------------------------------------------------------//
//! discrete sampling from a dense voxel grid
template< typename VALUE_TYPE, typename SAMPLE_TYPE >
struct DenseFieldSampler
{
	typedef VALUE_TYPE value_type;
	typedef typename GpuFieldTraits<VALUE_TYPE>::cuda_value_type cuda_value_type;
	typedef SAMPLE_TYPE sample_type;

	DenseFieldSampler( int _nx, int _ny, int _nz, cuda_value_type* _phi )
	: nx( _nx )
	, ny( _ny )
	, nz( _nz )
	, ystride( nx )
	, zstride( nx*ny )
	, nxnynz( nx*ny*nz )
	, phi( _phi )
	, dataWindowMin( make_int3( 0, 0, 0 ) )
	, dataWindowMax( make_int3( _nx - 1, _ny - 1, _nz - 1 ) )
	{}

	//! 3d to 1d index mapping
	__host__ __device__
	int getIndex( int x, int y, int z ) const
	{
		return z * zstride + y * ystride + x;
	}

	//! 1d to 3d index mapping
	__host__ __device__
	int3 getIndex( int i ) const
	{
		int3 id;
		id.z = i / zstride;
		id.y = (i / ystride) % ny;
		id.x = i % ystride;
		return id;
	}

	//! get value using 1d index
	template< typename ACCESSOR >
	__host__ __device__
	SAMPLE_TYPE getValue( ACCESSOR& ac, int idx ) const
	{
		return ac( idx, phi );
	}

	//! get value using 3d index
	template< typename ACCESSOR >
	__host__ __device__
	SAMPLE_TYPE getValue(ACCESSOR& ac, int x, int y, int z ) const
	{
		int idx = getIndex( x,y,z );
		return getValue( ac, idx );
	}

	//! expose data pointer for texture binding
	cuda_value_type* dataPtr() const
	{
		return phi;
	}

	//! expose data size for texture binding
	size_t texMemSize() const
	{
		return nxnynz*sizeof(VALUE_TYPE);
	}

	const int nx, ny, nz;
	const int ystride;
	const int zstride;
	const int nxnynz;
	const int3 dataWindowMin;
	const int3 dataWindowMax;
	//! data ptr
	cuda_value_type* phi;
};


//----------------------------------------------------------------------------//
//! continuous (interpolated) sampling via a uniform grid sampler (ie providing i,j,k access)
template< typename S >
struct LinearFieldInterp
{
	typedef LinearFieldInterp<S> interp_type;
	typedef S sampler_type;
	typedef typename S::sample_type sample_type;
	typedef typename S::value_type value_type;

	LinearFieldInterp( S _sampler )
	: sampler( _sampler )
	{}

	template< typename ACCESSOR >
	__host__ __device__
	void sample( ACCESSOR& ac, const float3& vsP, sample_type& dst ) const
	{
		// Voxel centers are at .5 coordinates
		// NOTE: Don't use contToDisc for this, we're looking for sample
		// point locations, not coordinate shifts.
		float3 p = vsP - make_float3(0.5);

		// Lower left corner
		int3 c1 = make_int3(	static_cast<int>(floorf(p.x)),
								static_cast<int>(floorf(p.y)),
								static_cast<int>(floorf(p.z)));
		// Upper right corner
		int3 c2(c1 + make_int3(1));
		// C1 fractions
		float3 f1(make_float3(c2) - p);
		// C2 fraction
		float3 f2(make_float3(1.0) - f1);

		// Clamp the indexing coordinates
		if( true ) {
			c1.x = max(sampler.dataWindowMin.x, min(c1.x,sampler.dataWindowMax.x));
			c2.x = max(sampler.dataWindowMin.x, min(c2.x,sampler.dataWindowMax.x));
			c1.y = max(sampler.dataWindowMin.y, min(c1.y,sampler.dataWindowMax.y));
			c2.y = max(sampler.dataWindowMin.y, min(c2.y,sampler.dataWindowMax.y));
			c1.z = max(sampler.dataWindowMin.z, min(c1.z,sampler.dataWindowMax.z));
			c2.z = max(sampler.dataWindowMin.z, min(c2.z,sampler.dataWindowMax.z));
		}

		dst =
		    (f1.x * (f1.y * (f1.z * sampler.getValue(ac, c1.x, c1.y, c1.z) +
		                     f2.z * sampler.getValue(ac, c1.x, c1.y, c2.z)) +
		             f2.y * (f1.z * sampler.getValue(ac, c1.x, c2.y, c1.z) +
		                     f2.z * sampler.getValue(ac, c1.x, c2.y, c2.z))) +
		     f2.x * (f1.y * (f1.z * sampler.getValue(ac, c2.x, c1.y, c1.z) +
		                     f2.z * sampler.getValue(ac, c2.x, c1.y, c2.z)) +
		             f2.y * (f1.z * sampler.getValue(ac, c2.x, c2.y, c1.z) +
		                     f2.z * sampler.getValue(ac, c2.x, c2.y, c2.z))));
	}

#ifdef NVCC
	template< typename TEX_ACCESSOR >
	void bindTex( TEX_ACCESSOR& tex ) const
	{
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc< typename TEX_ACCESSOR::cuda_tex_value_type >();
		tex.tex().filterMode = cudaFilterModePoint;
		tex.tex().normalized = false;
		tex.tex().channelDesc = channelDesc;
		if (cudaBindTexture( NULL, &tex.tex(), sampler.dataPtr(), &channelDesc, sampler.texMemSize() ) != (unsigned int) CUDA_SUCCESS) {
			throw std::runtime_error( "failed to bind texture" );
		}
	}

	template< typename TEX_ACCESSOR >
	void unbindTex( TEX_ACCESSOR& tex ) const
	{
		cudaUnbindTexture( &tex.tex() );
	}
#endif

	//! member vars
	const S sampler;
};

}

FIELD3D_NAMESPACE_HEADER_CLOSE

#endif // Include guard
