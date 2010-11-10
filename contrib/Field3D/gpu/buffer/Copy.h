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

#ifndef _INCLUDED_Field3D_gpu_Copy_H_
#define _INCLUDED_Field3D_gpu_Copy_H_

#ifdef INCLUDE_FIELD3D_CUDA
#include <thrust/device_vector.h>
#endif

#include "Field3D/gpu/ns.h"
#include "Field3D/gpu/buffer/IteratorTraits.h"

#define FIELD3D_GPU_BUFFER_COPY_LOG 0

FIELD3D_GPU_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
template< typename InputIterator, typename OutputIterator >
OutputIterator copy(	InputIterator first,
                    	InputIterator last,
						OutputIterator result )
{
	// dispatch the appropriate type of copy
	return copy( first, last, result, typename IteratorTraits<InputIterator>::type(), typename IteratorTraits<OutputIterator>::type() );
}

//----------------------------------------------------------------------------//
//! Copy from host to host
template< typename InputIterator, typename OutputIterator >
OutputIterator copy( 	InputIterator first,
						InputIterator last,
						OutputIterator result,
						host_tag,
						host_tag )
{
#if FIELD3D_GPU_BUFFER_COPY_LOG
	std::cout << "copying host->host\n";
#endif

	return std::copy( first, last, result );
}

#ifdef INCLUDE_FIELD3D_CUDA
//----------------------------------------------------------------------------//
//! Copy from host to cuda buffer
template< typename InputIterator, typename OutputIterator >
OutputIterator copy( 	InputIterator first,
						InputIterator last,
						OutputIterator result,
						host_tag,
						cuda_tag )
{
#if FIELD3D_GPU_BUFFER_COPY_LOG
	std::cout << "copying host->cuda\n";
#endif

	// avoid using thrust copy while half is stored as short
	// return thrust::copy( first, last, result );
	size_t n = std::distance( first, last );
	cudaMemcpy( thrust::raw_pointer_cast( &( *result ) ), &( *first ), sizeof( *first ) * n, cudaMemcpyHostToDevice );
	return result + n;
}

//----------------------------------------------------------------------------//
//! Copy from cuda buffer to host
template< typename InputIterator, typename OutputIterator >
OutputIterator copy( 	InputIterator first,
						InputIterator last,
						OutputIterator result,
						cuda_tag,
						host_tag )
{
#if FIELD3D_GPU_BUFFER_COPY_LOG
	std::cout << "copying cuda->host\n";
#endif

	// avoid using thrust copy while half is stored as short
	// return thrust::copy( first, last, result );
	size_t n = last - first;
	cudaMemcpy( &( *result ), thrust::raw_pointer_cast( &( *first ) ), sizeof(typename OutputIterator::value_type) * n, cudaMemcpyDeviceToHost );
	return result + n;
}
#endif

#ifdef INCLUDE_FIELD3D_OPENCL
//----------------------------------------------------------------------------//
//! Copy from host to OpenCL buffer
template< typename InputIterator, typename OutputIterator >
OutputIterator copy( 	InputIterator first,
						InputIterator last,
						OutputIterator result,
						host_tag,
						opencl_tag )
{
#if FIELD3D_GPU_BUFFER_COPY_LOG
	std::cout << "copying host->opencl\n";
#endif

	result.vec()->setValue( result.index(), first, last );
}

//----------------------------------------------------------------------------//
//! Copy from OpenCL buffer to host
template< typename InputIterator, typename OutputIterator >
OutputIterator copy( 	InputIterator first,
						InputIterator last,
						OutputIterator result,
						opencl_tag,
						host_tag )
{
#ifdef FIELD3D_GPU_BUFFER_COPY_LOG
	std::cout << "copying opencl->host\n";
#endif

	first.vec()->getValue( first.index(), last.index(), result );
}
#endif

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
