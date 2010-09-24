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

#include "gpu_field_test.h"

// field3d includes
#include "Field3D/DenseField.h"
#include "Field3D/InitIO.h"
#include "Field3D/Field3DFile.h"
#include "Field3D/FieldInterp.h"

// field3d gpu includes
#include "Field3D/gpu/DenseFieldCuda.h"
#include "Field3D/gpu/DenseFieldSamplerCuda.h"
#include "Field3D/gpu/SparseFieldCuda.h"
#include "Field3D/gpu/SparseFieldSamplerCuda.h"
#include "Field3D/gpu/FieldInterpCuda.h"
#include "Field3D/gpu/DataAccessorCuda.h"
#include "Field3D/gpu/Timer.h"

// std includes
#include <iostream>
#include <string>

//----------------------------------------------------------------------------//

using namespace std;

//----------------------------------------------------------------------------//
namespace nvcc
{
	template< typename INTERP >
	void testHost( thrust::host_vector<Vec3f>& host_p, thrust::host_vector<float>& host_urn, INTERP& interp, bool dump_result );

	template< typename INTERP >
	void testDevice( thrust::host_vector<Vec3f>& host_p, thrust::host_vector<float>& host_urn, INTERP& interp, bool dump_result );

	template< typename INTERP >
	void testTexDevice( thrust::host_vector<Vec3f>& host_p, thrust::host_vector<float>& host_urn, INTERP& interp, bool dump_result );
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
			std::cout << "    FieldInterp:\trandom sample: " << et << std::endl;
		}
	}

	//----------------------------------------------------------------------------//
	template< typename INTERP >
	void testHost( thrust::host_vector<Vec3f>& host_p, thrust::host_vector<float>& host_urn, INTERP& interp, bool dump_result )
	{
		{
			Field3D::Gpu::GlobalMemAccessor< typename INTERP::value_type > ac;
			int sample_count = host_p.size();
			thrust::host_vector< typename INTERP::sample_type > host_result( sample_count, 0.0f );
			RandomSampleFunctor< INTERP, Field3D::Gpu::GlobalMemAccessor< typename INTERP::value_type > > f( ac, interp, &host_p[0], &host_result[0] );

			float et = RunFunctor( f, sample_count, thrust::host_space_tag() );

			if( dump_result ){
				std::cout << "    gcc host result         : ";
				dump( host_result );
			} else {
				std::cout << "    gcc host:\t\trandom sample: " << et << "\t" << std::flush;
			}
		}
		if( !dump_result )
		{
			Field3D::Gpu::GlobalMemAccessor< typename INTERP::value_type > ac;
			int sample_count = interp.getSampler().dataWindowVoxelCount();
			thrust::host_vector< typename INTERP::sample_type > host_result( sample_count, 0.0f );
			SuperSampleFunctor< INTERP, Field3D::Gpu::GlobalMemAccessor< typename INTERP::value_type > > f( ac, interp, &host_urn[0], host_urn.size(), &host_result[0] );

			float et = RunFunctor( f, sample_count, thrust::host_space_tag() );

			std::cout << "super sample: " << et << std::endl;
		}
	}
}

//----------------------------------------------------------------------------//
//! test a field in various evaluation contexts
template< typename FieldType >
void testField()
{
	int res = TEST_RESOLUTION;

	std::cout << "\ntesting " << nameOf< FieldType >() << " at " << res << "x" << res << "x" << res << std::endl;

	// create a field of the desired ytpe
	boost::intrusive_ptr<FieldType> field( new FieldType );

	field->name = "hello";
	field->attribute = "world";
	// test with data window
	field->setSize(	Field3D::Box3i( Field3D::V3i( 0, 0, 0 ), Field3D::V3i( res, res, res ) ),
					Field3D::Box3i( Field3D::V3i( 2, 2, 2 ), Field3D::V3i( res - 2, res - 2, res - 2 ) ) );
	field->clear( 1.0f );
	randomValues( -10.0f, 10.0f, *field );
	field->setStrMetadata( "my_attribute", "my_value" );

	std::cout << "  verbose run...\n";

	// table of uniform random numbers
	thrust::host_vector<float> host_urn( 9999991 );
	randomValues( 0, 1, host_urn );

	int sample_count = 8;
	bool dump_result = true;
	thrust::host_vector<Vec3f> host_p( sample_count );
	randomLocations( field->dataWindow(), host_p );

	{
		gcc::testField3D( host_p, *field, dump_result );

		boost::shared_ptr< typename FieldType::linear_interp_type > interp = field->getLinearInterpolatorHost();
		if( interp != NULL ){
			gcc::testHost( host_p, host_urn, *interp, dump_result );
			nvcc::testHost( host_p, host_urn, *interp, dump_result );
		}
	}
	{
		boost::shared_ptr< typename FieldType::linear_interp_type > interp = field->getLinearInterpolatorDevice();
		if( interp != NULL ){
			nvcc::testDevice( host_p, host_urn, *interp, dump_result );
			nvcc::testTexDevice( host_p, host_urn, *interp, dump_result );
		}
	}

	sample_count = PROFILE_SAMPLE_COUNT;
	std::cout << "  profiling run (" << sample_count / 1000000 << "M samples, wall clock seconds)...\n";

	host_p.resize( sample_count );
	randomLocations( field->dataWindow(), host_p );

	//dump( host_urn );

	dump_result = false;
	{
		gcc::testField3D( host_p, *field, dump_result );

		boost::shared_ptr< typename FieldType::linear_interp_type > interp = field->getLinearInterpolatorHost();
		if( interp != NULL ){
			gcc::testHost( host_p, host_urn, *interp, dump_result );
			nvcc::testHost( host_p, host_urn, *interp, dump_result );
		}
	}
	{
		boost::shared_ptr< typename FieldType::linear_interp_type > interp = field->getLinearInterpolatorDevice();
		if( interp != NULL ){
			nvcc::testDevice( host_p, host_urn, *interp, dump_result );
			nvcc::testTexDevice( host_p, host_urn, *interp, dump_result );
		}
	}
}

//----------------------------------------------------------------------------//
int main( 	int argc,
			char **argv )
{
	// Call initIO() to initialize standard I/O methods and load plugins
	//Field3D::initIO();

	std::cout << "\ntesting using " << threadCount() << " threads" << std::endl;
	std::cout << std::fixed << std::setprecision(6);

	testField< Field3D::Gpu::DenseFieldCuda< double > >();
	testField< Field3D::Gpu::DenseFieldCuda< float > >();
	testField< Field3D::Gpu::DenseFieldCuda< Field3D::half > >();

	testField< Field3D::Gpu::SparseFieldCuda< double > >();
	testField< Field3D::Gpu::SparseFieldCuda< float > >();
	testField< Field3D::Gpu::SparseFieldCuda< Field3D::half > >();
	return 0;
}

//----------------------------------------------------------------------------//

