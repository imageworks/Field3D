//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks
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

#ifndef _INCLUDED_Field3D_gpu_common_H_
#define _INCLUDED_Field3D_gpu_common_H_

#ifdef INCLUDE_FIELD3D_OPENCL
#warning "Including Field3D OpenCL code"
#endif

#ifdef INCLUDE_FIELD3D_CUDA
#warning "Including Field3D Cuda code"
#endif

#include <iostream>
#include "Field3D/Types.h"
#include <OpenEXR/ImathRandom.h>
typedef Imath::Rand48 Rand;

#ifdef NDEBUG
#define TEST_RESOLUTION 200
#define PROFILE_SAMPLE_COUNT 8000000
#define SUPER_SAMPLE_COUNT 3
#else
#define TEST_RESOLUTION 73
#define PROFILE_SAMPLE_COUNT 800000
#define SUPER_SAMPLE_COUNT 3
#endif


//----------------------------------------------------------------------------//
//! just for debugging
#define LOGLINE { std::cerr << __FILE__ << ": " << __LINE__ << std::endl; }

//----------------------------------------------------------------------------//
//! dump a vec to std::cout
template< typename VEC >
void dump( const VEC& v )
{
	typename VEC::const_iterator i( v.begin() ), e( v.end() );
	for ( ; i != e ; ++i )
		std::cout << *i << ", ";
	std::cout << std::endl;
}

//----------------------------------------------------------------------------//
//! generate some random values
template< typename VEC >
void randomValues( float minv, float maxv, VEC& dst )
{
	Rand rng(5171);
	typename VEC::iterator i( dst.begin() ), e( dst.end() );
	for ( ; i != e ; ++i )
		*i = rng.nextf( minv, maxv );
}

//----------------------------------------------------------------------------//
//! generate some random locations based on an input resolution
template< typename VEC >
void randomLocations(	const Field3D::Box3i& bounds,
						VEC& dst )
{
	Rand rng( 1877 );
	typename VEC::value_type v;

	// random sampling over entire field
	typename VEC::iterator i( dst.begin() ), e( dst.end() );
	for ( ; i != e ; ++i )
	{
		v.x = rng.nextf( bounds.min.x, bounds.max.x );
		v.y = rng.nextf( bounds.min.y, bounds.max.y );
		v.z = rng.nextf( bounds.min.z, bounds.max.z );
		*i = v;
	}
}

#endif  // Include guard
