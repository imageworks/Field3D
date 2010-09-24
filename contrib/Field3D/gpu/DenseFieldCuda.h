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

#ifndef _INCLUDED_Field3D_gpu_DenseFieldCuda_H_
#define _INCLUDED_Field3D_gpu_DenseFieldCuda_H_

#ifdef NVCC
#error This file is intended for GCC and isn't compatible with NVCC compiler due to Field3D includes
#endif

#include "Field3D/DenseField.h"
#include "Field3D/Types.h"
#include "Field3D/gpu/ns.h"
#include "Field3D/gpu/FieldInterpCuda.h"

#include <thrust/device_vector.h>

FIELD3D_GPU_NAMESPACE_OPEN

// forward declarations
template< typename A, typename B > struct DenseFieldSampler;
template< typename A > struct LinearFieldInterp;

//----------------------------------------------------------------------------//
//! Cuda layer for DenseFields
template< typename Data_T >
struct DenseFieldCuda : public DenseField< Data_T >
{
	typedef typename GpuFieldTraits< Data_T >::value_type value_type;
	typedef typename GpuFieldTraits< Data_T >::cuda_value_type cuda_value_type;
	typedef typename GpuFieldTraits< Data_T >::interpolation_type interpolation_type;
	typedef DenseFieldSampler< value_type, interpolation_type > sampler_type;
	typedef LinearFieldInterp< sampler_type > linear_interp_type;
	typedef DenseField< Data_T > base;

	//----------------------------------------------------------------------------//
	//! manufacture an interpolator for device
	boost::shared_ptr< linear_interp_type > getLinearInterpolatorDevice() const
	{
		// load data onto device
		hostToDevice( DenseField< Data_T >::m_data, m_deviceData );

		V3i res = DenseField< Data_T >::dataResolution();
		return boost::shared_ptr< linear_interp_type >( new linear_interp_type( sampler_type( base::dataResolution(), base::dataWindow(),
				(cuda_value_type*) thrust::raw_pointer_cast( &m_deviceData[ 0 ] ) ) ) );
	}

	//----------------------------------------------------------------------------//
	//! manufacture an interpolator for host
	boost::shared_ptr< linear_interp_type > getLinearInterpolatorHost() const
	{
		V3i res = DenseField< Data_T >::dataResolution();
		return boost::shared_ptr< linear_interp_type >( new linear_interp_type( sampler_type( base::dataResolution(), base::dataWindow(),
				(cuda_value_type*) &DenseField< Data_T >::m_data[ 0 ] ) ) );
	}

	//----------------------------------------------------------------------------//
	//! general host to device transfer
	template< typename T, typename OT >
	void hostToDevice( const std::vector<T>& src, thrust::device_vector<OT>& dst ) const
	{
		dst = src;
	}

	//----------------------------------------------------------------------------//
	//! specialization of host to device transfer for half float
	void hostToDevice( const std::vector<Field3D::half>& src, thrust::device_vector<short>& dst ) const
	{
		dst.resize( src.size() );
		cudaMemcpy( thrust::raw_pointer_cast(&dst[0]), &src[0], src.size() * sizeof(short), cudaMemcpyHostToDevice );
	}

private:
	mutable thrust::device_vector< cuda_value_type > m_deviceData;
};

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
