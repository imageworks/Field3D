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

#ifndef _INCLUDED_Field3D_gpu_Traits_H_
#define _INCLUDED_Field3D_gpu_Traits_H_

#include "Field3D/gpu/ns.h"
#include "Field3D/Types.h"

FIELD3D_GPU_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
//! Traits class that defines gpu storage and access types
template< typename T >
struct GpuFieldTraits
{
};

//----------------------------------------------------------------------------//
//! specialization for double precision float scalar field
template< >
struct GpuFieldTraits< double >
{
	typedef double value_type;
	typedef double interpolation_type;
	typedef double cuda_value_type;
	typedef double cl_value_type;
#ifdef NVCC
	typedef int2 cuda_tex_value_type;
	typedef texture< cuda_tex_value_type, 1, cudaReadModeElementType > cuda_tex_type;
#endif
};

//----------------------------------------------------------------------------//
//! specialization for single precision float scalar field
template< >
struct GpuFieldTraits< float >
{
	typedef float value_type;
	typedef float interpolation_type;
	typedef float cuda_value_type;
	typedef float cl_value_type;
#ifdef NVCC
	typedef float cuda_tex_value_type;
	typedef texture< cuda_tex_value_type, 1, cudaReadModeElementType > cuda_tex_type;
#endif
};

//----------------------------------------------------------------------------//
//! specialization for half precision float scalar field
template< >
struct GpuFieldTraits< Field3D::half >
{
	typedef Field3D::half value_type;
	typedef float interpolation_type;
	typedef short cuda_value_type;
	typedef Field3D::half cl_value_type;
#ifdef NVCC
	typedef short cuda_tex_value_type;
	typedef texture< cuda_tex_value_type, 1, cudaReadModeElementType > cuda_tex_type;
#endif
};

//----------------------------------------------------------------------------//
//! specialization for ints
template< >
struct GpuFieldTraits< int >
{
	typedef int value_type;
	typedef int cuda_value_type;
	typedef int cl_value_type;
};

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
