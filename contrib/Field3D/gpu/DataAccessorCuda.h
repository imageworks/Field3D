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

#ifndef _INCLUDED_Field3D_gpu_DataAccessorCuda_H_
#define _INCLUDED_Field3D_gpu_DataAccessorCuda_H_

// field3d includes
#include "Field3D/gpu/ns.h"
#include "Field3D/Types.h"

// cuda includes
#include <host_defines.h>

FIELD3D_GPU_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
//! 1d access to data
struct DataAccessor
{
};

//----------------------------------------------------------------------------//
//! access data directly from cuda global memory, or host memory
template< typename T >
struct GlobalMemAccessor: public DataAccessor
{
	inline __host__  __device__
	T operator()( 	int i, T* phi )
	{
		return phi[ i ];
	}
};

//----------------------------------------------------------------------------//
//! specialization of GlobalMemAccessor for half float
template< >
struct GlobalMemAccessor< Field3D::half >
{
	inline __host__  __device__
	float operator()( 	int i,
						short* phi )
	{
#ifdef __CUDA_ARCH__
#if CUDART_VERSION < 3010
#error requires cuda version >= 3.1 for half float intrinsics
#endif
		// half to float intrinsic is only available in device code
		return __half2float( phi[i] );
#else
		// ILM's half to float conversion is only available in host code
		return *reinterpret_cast< half* > ( &phi[ i ] );
#endif
	}
};

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
