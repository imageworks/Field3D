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

#ifndef _INCLUDED_Field3D_gpu_IteratorTraits_H_
#define _INCLUDED_Field3D_gpu_IteratorTraits_H_

#include <vector>

#include "Field3D/gpu/ns.h"

FIELD3D_GPU_NAMESPACE_OPEN

// forward declaration
template< typename T >
struct IteratorCL;
template< typename T >
struct IteratorTraits;

//----------------------------------------------------------------------------//
//! A flag to identify host iterators etc.
struct host_tag {};
//! A flag to identify Cuda iterators etc.
struct cuda_tag {};
//! A flag to identify OpenCL iterators etc.
struct opencl_tag {};

//----------------------------------------------------------------------------//
/*! traits for a raw pointer (assumes pointing to host memory)
 * \note: use thrust::device_ptr<T> when pointing to device memory
 */
template< typename T >
struct IteratorTraits< const T* >
{
	typedef host_tag type;
};

#ifdef INCLUDE_FIELD3D_CUDA
//----------------------------------------------------------------------------//
//! traits for a cuda iterator
template< typename T >
struct IteratorTraits< thrust::detail::normal_iterator< thrust::device_ptr< T > > >
{
	typedef cuda_tag type;
};

//----------------------------------------------------------------------------//
//! traits for a thrust device pointer
template< typename T >
struct IteratorTraits< thrust::device_ptr<T> >
{
	typedef cuda_tag type;
};

#endif

#ifdef INCLUDE_FIELD3D_OPENCL
//----------------------------------------------------------------------------//
//! traits for an OpenCL iterator
template< typename T >
struct IteratorTraits< IteratorCL<T> >
{
	typedef opencl_tag type;
};
#endif

//----------------------------------------------------------------------------//
//! traits for an std::vector const_iterator
template< typename T >
struct IteratorTraits< __gnu_cxx::__normal_iterator<const T*, std::vector<T, std::allocator<T> > > >
{
	typedef host_tag type;
};

//----------------------------------------------------------------------------//
//! traits for an std::vector iterator
template< typename T >
struct IteratorTraits< __gnu_cxx::__normal_iterator<T*, std::vector<T, std::allocator<T> > > >
{
	typedef host_tag type;
};


// TODO: add raw pointer traits (may need some guard to differentiate host/device)

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
