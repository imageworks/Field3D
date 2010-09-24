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

#ifndef _INCLUDED_Field3D_gpu_DenseFieldSamplerCuda_H_
#define _INCLUDED_Field3D_gpu_DenseFieldSamplerCuda_H_

#include "Field3D/gpu/ns.h"
#include "Field3D/gpu/Traits.h"
#include "Field3D/gpu/FieldSamplerCuda.h"

FIELD3D_GPU_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
//! discrete sampling from a dense voxel grid
template< typename VALUE_TYPE, typename SAMPLE_TYPE >
struct DenseFieldSampler
: public FieldSampler
{
	typedef VALUE_TYPE value_type;
	typedef typename GpuFieldTraits< VALUE_TYPE >::cuda_value_type cuda_value_type;
	typedef SAMPLE_TYPE sample_type;
	typedef FieldSampler base;

	DenseFieldSampler( 	const Field3D::V3i& _dataResolution,
						const Field3D::Box3i& _dataWindow,
						cuda_value_type* _phi )
	: base( _dataResolution, _dataWindow )
	, phi( _phi )
	{}

	//! 3d to 1d index mapping
	inline __host__  __device__
	int getIndex( 	int i,
					int j,
					int k ) const
	{
		kernel_assert( i >= dataWindowMin.x );
		kernel_assert( i <= dataWindowMax.x );
		kernel_assert( j >= dataWindowMin.y );
		kernel_assert( j <= dataWindowMax.y );
		kernel_assert( k >= dataWindowMin.z );
		kernel_assert( k <= dataWindowMax.z );

		// Add crop window offset
		applyDataWindowOffset( i, j, k );

		return k * zstride + j * ystride + i;
	}

	int allocatedVoxelCount() const
	{
		return dataWindowVoxelCount();
	}

	//! get value using 1d index
	template< typename ACCESSOR >
	inline __host__  __device__
	SAMPLE_TYPE getValue( 	ACCESSOR& ac,
							int idx ) const
	{
		return ac( idx, phi );
	}

	//! get value using 3d index
	template< typename ACCESSOR >
	inline __host__  __device__
	SAMPLE_TYPE getValue( 	ACCESSOR& ac,
							int x,
							int y,
							int z ) const
	{
		return getValue( ac, getIndex( x, y, z ) );
	}

	//! expose data pointer for texture binding
	cuda_value_type* dataPtr() const
	{
		return phi;
	}

	//! expose data size for texture binding
	size_t texMemSize() const
	{
		return nxnynz * sizeof(VALUE_TYPE);
	}

	//! data ptr
	cuda_value_type* phi;
};

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard

