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

#ifndef _INCLUDED_Field3D_gpu_dense_field_test_H_
#define _INCLUDED_Field3D_gpu_dense_field_test_H_

#include "Field3D/Types.h"
#include <cutil_math.h>
#include <OpenEXR/ImathRandom.h>
#include <iostream>

//----------------------------------------------------------------------------//
// to be replaced by imath once it's cuda compatible
typedef float3 Vec3f;
#define MakeVec3f make_float3

//----------------------------------------------------------------------------//
// just some pretty formatting of type names
template< typename T > inline const char* nameOf()
{
	return typeid(T).name();
}
template<> inline const char* nameOf< Field3D::half >(){ return "HALF"; }
template<> inline const char* nameOf< float >(){ return "FLOAT"; }
template<> inline const char* nameOf< double >(){ return "DOUBLE"; }

//----------------------------------------------------------------------------//
//! user defined sampling operation
template< typename INTERPOLATOR, typename ACCESSOR >
struct SampleFunctor
{
	SampleFunctor( 	ACCESSOR& _ac,
	               	INTERPOLATOR _fn,
					Vec3f* _p,
					typename INTERPOLATOR::sample_type* _r ) :
		ac( _ac ), fn( _fn ), p( _p ), r( _r )
	{
	}

	__host__    __device__
	void operator()( int i )
	{
		fn.sample( ac, p[ i ], r[ i ] );
	}

	ACCESSOR& ac;
	INTERPOLATOR fn;
	Vec3f* p;
	typename INTERPOLATOR::sample_type* r;
};

typedef Imath::Rand48 Rand;

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

#define LOGLINE { std::cerr << __FILE__ << ": " << __LINE__ << std::endl; }

//----------------------------------------------------------------------------//
//! generate some random locations based on an input resolution
template< typename VEC >
void randomLocations( int res, VEC& dst )
{
	Rand rng(1877);
	float minv = 0.5f;
	float maxv = float(res)-0.5f;

	typename VEC::iterator i( dst.begin() ), e( dst.end() );
	for ( ; i != e ; ++i )
		*i = MakeVec3f( rng.nextf(minv,maxv), rng.nextf(minv,maxv), rng.nextf(minv,maxv) );
};

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



#endif // Include guard
