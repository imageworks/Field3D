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

#ifndef _INCLUDED_Field3D_gpu_BufferDump_H_
#define _INCLUDED_Field3D_gpu_BufferDump_H_

#include "Field3D/gpu/Traits.h"
#include "Field3D/gpu/NameOf.h"

#include "Field3D/gpu/ns.h"

FIELD3D_GPU_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// forward declaration
template< typename InputIterator, typename OutputIterator >
OutputIterator copy(	InputIterator first,
                    	InputIterator last,
						OutputIterator result );

//----------------------------------------------------------------------------//
//! dump a host buffer
template< typename T >
void dumpBuffer( const std::vector<T>& b )
{
	std::cout << "type: " << nameOf<T>() << "\n";
	typename std::vector< T >::const_iterator i( b.begin() ), e( b.end() );
	for( ; i != e; ++i ){
		std::cout << *i << " ";
	}
	std::cout << std::endl;
}

//----------------------------------------------------------------------------//
//! dump a device buffer
template< typename Buffer >
void dumpBuffer( const Buffer& device_buffer )
{
	std::vector< typename Buffer::value_type > host_buffer( device_buffer.size() );

	// device->host
	Field3D::Gpu::copy( device_buffer.begin(), device_buffer.end(), host_buffer.begin() );

	dumpBuffer( host_buffer );
}

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
