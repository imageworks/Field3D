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

#ifndef _INCLUDED_Field3D_gpu_DenseFieldGPU_H_
#define _INCLUDED_Field3D_gpu_DenseFieldGPU_H_

#ifdef NVCC
#error "This file is intended for GCC and isn't compatible with NVCC compiler due to Field3D includes"
#endif

#include "Field3D/gpu/Traits.h"
#include "Field3D/gpu/buffer/Copy.h"

#ifdef INCLUDE_FIELD3D_CUDA
#include "Field3D/gpu/FieldInterpCuda.h"
#include "Field3D/gpu/buffer/BufferCuda.h"
#endif

#include "Field3D/DenseField.h"
#include "Field3D/Types.h"

#include "Field3D/gpu/ns.h"

FIELD3D_GPU_NAMESPACE_OPEN

// forward declarations
template< typename A, typename B > struct DenseFieldSampler;
template< typename A > struct LinearFieldInterp;

//----------------------------------------------------------------------------//
//! Cuda layer for DenseFields
template< typename Data_T >
struct DenseFieldGPU : public RefBase
{
	typedef boost::intrusive_ptr< DenseFieldGPU > Ptr;
	typedef typename GpuFieldTraits< Data_T >::value_type value_type;
	typedef typename GpuFieldTraits< Data_T >::cuda_value_type cuda_value_type;
	typedef typename GpuFieldTraits< Data_T >::cl_value_type cl_value_type;
	typedef typename GpuFieldTraits< Data_T >::interpolation_type interpolation_type;
	typedef DenseFieldSampler< value_type, interpolation_type > sampler_type;
	typedef LinearFieldInterp< sampler_type > LinearInterp;
	typedef typename boost::shared_ptr< LinearInterp > LinearInterpPtr;

	typedef DenseField< Data_T > field3d_type;
	typedef typename field3d_type::Ptr field_ptr;

	//! set the Field3D field
	field_ptr getField()
	{
		return m_field;
	}

	//! access the Field3D field
	void setField( field_ptr& ptr )
	{
		m_field = ptr;
	}


#ifdef INCLUDE_FIELD3D_CUDA
	//----------------------------------------------------------------------------//
	//! Manufacture an interpolator for device
	LinearInterpPtr getLinearInterpolatorDevice() const
	{
		hostToDevice( m_bufferCuda );

		return LinearInterpPtr( new LinearInterp( sampler_type( m_field->dataResolution(), m_field->dataWindow(),
				(cuda_value_type*) thrust::raw_pointer_cast( &m_bufferCuda[ 0 ] ) ) ) );
	}

	//----------------------------------------------------------------------------//
	//! Manufacture an interpolator for host
	LinearInterpPtr getLinearInterpolatorHost() const
	{
		Field3D::Box3i dw = m_field->dataWindow();
		const Data_T* ptr = &m_field->fastValue( dw.min.x, dw.min.y, dw.min.z );

		return LinearInterpPtr( new LinearInterp( sampler_type( m_field->dataResolution(), m_field->dataWindow(),
				(cuda_value_type*) ptr ) ) );
	}
#endif

	//----------------------------------------------------------------------------//
	//! Transfer data from host to device
	template< typename Buffer >
	void hostToDevice( Buffer& dst ) const
	{
		Field3D::V3i mem_size = m_field->internalMemSize();
		size_t element_count = mem_size.x * mem_size.y * mem_size.z;
		Field3D::Box3i dw = m_field->dataWindow();
		const Data_T* ptr = &m_field->fastValue( dw.min.x, dw.min.y, dw.min.z );

		dst.resize( element_count );
		Field3D::Gpu::copy( ptr, ptr + element_count, dst.begin() );
	}

private:
	//! A pointer to the Field3D (non-GPU) field
	field_ptr m_field;

#ifdef INCLUDE_FIELD3D_CUDA
	mutable BufferCuda< Data_T > m_bufferCuda;
#endif
};

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
