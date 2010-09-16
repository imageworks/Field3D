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

#include "dense_field_test.h"

#include <iostream>
#include <string>

#include <Field3D/DenseField.h>
#include <Field3D/InitIO.h>
#include <Field3D/Field3DFile.h>
#include <Field3D/FieldInterp.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Field3D/gpu/DenseFieldCuda.h"
#include "Field3D/gpu/FieldInterpCuda.h"
#include "Field3D/gpu/Timer.h"

//----------------------------------------------------------------------------//

using namespace std;

//----------------------------------------------------------------------------//
namespace nvcc
{
	template< typename INTERP >
	void testHost( thrust::host_vector<Vec3f>& host_p, INTERP& interp, bool dump_result );

	template< typename INTERP >
	void testDevice( thrust::host_vector<Vec3f>& host_p, INTERP& interp, bool dump_result );

	template< typename INTERP >
	void testTexDevice( thrust::host_vector<Vec3f>& host_p, INTERP& interp, bool dump_result );
}

//----------------------------------------------------------------------------//
namespace gcc
{
	Field3D::V3d assign_op (Vec3f src) { return Field3D::V3d(src.x,src.y,src.z); }

	//----------------------------------------------------------------------------//
	template< typename FIELD >
	void testField3D( thrust::host_vector<Vec3f>& host_p, const FIELD& f, bool dump_result )
	{
		int sample_count = host_p.size();
		std::vector<typename FIELD::value_type> host_result( sample_count );
		std::vector<Field3D::V3d> p( host_p.size() );
		std::transform( host_p.begin(), host_p.end(), p.begin(), assign_op );

		Field3D::LinearFieldInterp< typename FIELD::value_type > interp;

		Field3D::Gpu::CpuTimer t;

		_Pragma( "omp parallel for" )
		for( int i = 0; i < sample_count; ++i ){
			host_result[i] = interp.sample( f, p[i] );
		}

		float et = t.elapsed();

		if( dump_result ){
			std::cout << "    FieldInterp result      : ";
			dump( host_result );
		} else {
			std::cout << "    FieldInterp et: " << et << std::endl;
		}
	}

	//----------------------------------------------------------------------------//
	template< typename INTERP >
	void testHost( thrust::host_vector<Vec3f>& host_p, INTERP& interp, bool dump_result )
	{
		Field3D::Gpu::GlobalMemAccessor< typename INTERP::value_type > ac;
		int sample_count = host_p.size();
		thrust::host_vector< typename INTERP::sample_type > host_result( sample_count, 0.0f );
		SampleFunctor< INTERP, Field3D::Gpu::GlobalMemAccessor< typename INTERP::value_type > > f( ac, interp, &host_p[0], &host_result[0] );

		Field3D::Gpu::CpuTimer t;

		_Pragma( "omp parallel for" )
		for( int i = 0; i < sample_count; ++i ){
			f(i);
		}

		float et = t.elapsed();

		if( dump_result ){
			std::cout << "    gcc host result         : ";
			dump( host_result );
		} else {
			std::cout << "    gcc host et: " << et << std::endl;
		}
	}
}

//----------------------------------------------------------------------------//
template< typename Data_T >
void testField()
{
	std::cout << "\ntesting " << nameOf< Data_T>() << " scalar field" << std::endl;
	int res = 256;

	typedef Field3D::Gpu::DenseFieldCuda< Data_T > FieldType;

	// create a field of the desired ytpe
	boost::intrusive_ptr<FieldType> field( new FieldType );

	field->name = "hello";
	field->attribute = "world";
	field->setSize( Field3D::V3i( res, res, res ) );
	field->clear( 1.0f );
	randomValues( -10.0f, 10.0f, *field );
	field->setStrMetadata( "my_attribute", "my_value" );

	std::cout << "  verbose run...\n";

	int sample_count = 8;
	bool dump_result = true;
	thrust::host_vector<Vec3f> host_p( sample_count );
	randomLocations( res, host_p );

	{
		boost::shared_ptr< typename FieldType::linear_interp_type > interp = field->getLinearInterpolatorHost();
		gcc::testField3D( host_p, *field, dump_result );
		gcc::testHost( host_p, *interp, dump_result );
		nvcc::testHost( host_p, *interp, dump_result );
	}
	{
		boost::shared_ptr< typename FieldType::linear_interp_type > interp = field->getLinearInterpolatorDevice();
		nvcc::testDevice( host_p, *interp, dump_result );
		nvcc::testTexDevice( host_p, *interp, dump_result );
	}

	std::cout << "  profiling run...\n";

	sample_count = 8000000;
	host_p.resize( sample_count );
	randomLocations( res, host_p );

	dump_result = false;
	{
		boost::shared_ptr< typename FieldType::linear_interp_type > interp = field->getLinearInterpolatorHost();
		gcc::testField3D( host_p, *field, dump_result );
		gcc::testHost( host_p, *interp, dump_result );
		nvcc::testHost( host_p, *interp, dump_result );
	}
	{
		boost::shared_ptr< typename FieldType::linear_interp_type > interp = field->getLinearInterpolatorDevice();
		nvcc::testDevice( host_p, *interp, dump_result );
		nvcc::testTexDevice( host_p, *interp, dump_result );
	}
}

//----------------------------------------------------------------------------//

int main( 	int argc,
			char **argv )
{
	// Call initIO() to initialize standard I/O methods and load plugins
	//Field3D::initIO();

	testField<double>();
	testField<float>();
	testField< Field3D::half >();

	return 0;
}

//----------------------------------------------------------------------------//

