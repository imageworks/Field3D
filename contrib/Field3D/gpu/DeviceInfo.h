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

#ifndef _INCLUDED_Field3D_gpu_DeviceInfo_H_
#define _INCLUDED_Field3D_gpu_DeviceInfo_H_

#include "Field3D/gpu/ns.h"
#include <cassert>
#include <cuda_runtime_api.h>

FIELD3D_GPU_NAMESPACE_OPEN

//! What compute capability are we running on?
/*! \note To do this within a kernel use the defined __CUDA_ARCH__ */
inline int deviceComputeCapability( int deviceId = 0 )
{
	cudaDeviceProp deviceProp;

	int nDevCount = 0;
	cudaGetDeviceCount( &nDevCount );
	assert( nDevCount > deviceId );
	cudaGetDeviceProperties( &deviceProp, deviceId );
	return deviceProp.major * 100 + deviceProp.minor * 10;
}

//! Does the device support double precision float?
inline bool deviceSupportsDoublePrecision( int deviceId = 0 )
{
	return deviceComputeCapability( deviceId ) >= 130;
}

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
