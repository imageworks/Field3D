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

#include "cuda_field_test.h"

// field3d includes
#include "Field3D/DenseField.h"
#include "Field3D/InitIO.h"
#include "Field3D/Field3DFile.h"
#include "Field3D/FieldInterp.h"

// field3d gpu includes
#include "Field3D/gpu/DeviceInfo.h"

#include "Field3D/gpu/DenseFieldGPU.h"
#include "Field3D/gpu/DenseFieldSamplerCuda.h"
#include "Field3D/gpu/SparseFieldGPU.h"
#include "Field3D/gpu/SparseFieldSamplerCuda.h"

#include "Field3D/gpu/FieldInterpCuda.h"
#include "Field3D/gpu/DataAccessorCuda.h"
#include "Field3D/gpu/Timer.h"
#include "Field3D/gpu/NameOf.h"

// thrust
#include <thrust/host_vector.h>

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
	float testField3D( thrust::host_vector<Vec3f>& host_p, const FIELD& f, bool dump_result )
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
		return et;
	}

	//----------------------------------------------------------------------------//
	template< typename INTERP >
	float testHost( thrust::host_vector<Vec3f>& host_p, thrust::host_vector<float>& host_urn, INTERP& interp, bool dump_result, bool use_openmp = true )
	{
		float et = 0;
		{
			Field3D::Gpu::GlobalMemAccessor< typename INTERP::value_type > ac;
			int sample_count = host_p.size();
			thrust::host_vector< typename INTERP::sample_type > host_result( sample_count, 0.0f );
			RandomSampleFunctor< INTERP, Field3D::Gpu::GlobalMemAccessor< typename INTERP::value_type > > f( ac, interp, &host_p[0], &host_result[0] );

			et = RunFunctor( f, sample_count, thrust::host_space_tag(), use_openmp );

			if( dump_result ){
				std::cout << "    gcc host result         : ";
				dump( host_result );
			} else {
				if( use_openmp ){
					std::cout << "    gcc host:\t\trandom sample: " << et << "\t" << std::flush;
				} else {
					std::cout << "    gcc host (1 thrd):\trandom sample: " << et << "\t" << std::flush;
				}
			}
		}
		if( !dump_result )
		{
			Field3D::Gpu::GlobalMemAccessor< typename INTERP::value_type > ac;
			int sample_count = interp.getSampler().dataWindowVoxelCount();
			thrust::host_vector< typename INTERP::sample_type > host_result( sample_count, 0.0f );
			SuperSampleFunctor< INTERP, Field3D::Gpu::GlobalMemAccessor< typename INTERP::value_type > > f( ac, interp, &host_urn[0], host_urn.size(), &host_result[0] );

			et = RunFunctor( f, sample_count, thrust::host_space_tag(), use_openmp );

			std::cout << "super sample: " << et << std::endl;
		}
		return et;
	}
}

//----------------------------------------------------------------------------//
//! test a field in various evaluation contexts
template< typename FieldType >
void testField()
{
	typedef typename Field3D::Gpu::GpuFieldType< FieldType >::type FieldTypeGPU;

	int res = TEST_RESOLUTION;

	std::cout << "\ntesting " << Field3D::Gpu::nameOf< FieldType >() << " at " << res << "x" << res << "x" << res << std::endl;

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

	// create a GPU field and attach it to the Field3D field
	typename FieldTypeGPU::Ptr gpu_field( new FieldTypeGPU );
	gpu_field->setField( field );

	std::cout << "  verbose run...\n";

	// table of uniform random numbers
	thrust::host_vector<float> host_urn( 9999991 );
	randomValues( 0, 1, host_urn );

	//----------------------------------------------------------------------------//
	// verbose run
	//----------------------------------------------------------------------------//

	int sample_count = 8;
	bool dump_result = true;
	thrust::host_vector<Vec3f> host_p( sample_count );
	randomLocations( field->dataWindow(), host_p );

	{
		gcc::testField3D( host_p, *field, dump_result );

		typename FieldTypeGPU::LinearInterpPtr interp = gpu_field->getLinearInterpolatorHost();
		if( interp != NULL ){
			gcc::testHost( host_p, host_urn, *interp, dump_result );
			nvcc::testHost( host_p, host_urn, *interp, dump_result );
		}
	}
	{
		typename FieldTypeGPU::LinearInterpPtr interp = gpu_field->getLinearInterpolatorDevice();
		if( interp != NULL ){
			nvcc::testDevice( host_p, host_urn, *interp, dump_result );
			nvcc::testTexDevice( host_p, host_urn, *interp, dump_result );
		}
	}

	//----------------------------------------------------------------------------//
	// profiling run
	//----------------------------------------------------------------------------//

	sample_count = PROFILE_SAMPLE_COUNT;
	std::cout << "  profiling run (" << sample_count / 1000000 << "M samples, wall clock seconds)...\n";

	host_p.resize( sample_count );
	randomLocations( field->dataWindow(), host_p );

	//dump( host_urn );

	dump_result = false;
	{
		gcc::testField3D( host_p, *field, dump_result );

		typename FieldTypeGPU::LinearInterpPtr interp = gpu_field->getLinearInterpolatorHost();
		if( interp != NULL ){
			gcc::testHost( host_p, host_urn, *interp, dump_result, false );
			gcc::testHost( host_p, host_urn, *interp, dump_result );
			nvcc::testHost( host_p, host_urn, *interp, dump_result );
		}
	}
	{
		typename FieldTypeGPU::LinearInterpPtr interp = gpu_field->getLinearInterpolatorDevice();
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
	std::cout << "\ntesting using " << threadCount() << " threads" << std::endl;
	std::cout << "device compute capability: " << Field3D::Gpu::deviceComputeCapability() / 100.0f << std::endl;
	std::cout << std::fixed << std::setprecision(6);

	// dense fields

	if( Field3D::Gpu::deviceSupportsDoublePrecision() )
		testField< Field3D::DenseField< double > >();
	else
		std::cout << "\nskipping " << Field3D::Gpu::nameOf< Field3D::Gpu::DenseFieldGPU< double > >() << std::endl;
	testField< Field3D::DenseField< float > >();
	testField< Field3D::DenseField< Field3D::half > >();

	// sparse fields

	if( Field3D::Gpu::deviceSupportsDoublePrecision() )
		testField< Field3D::SparseField< double > >();
	else
		std::cout << "\nskipping " << Field3D::Gpu::nameOf< Field3D::Gpu::SparseFieldGPU< double > >() << std::endl;
	testField< Field3D::SparseField< float > >();
	testField< Field3D::SparseField< Field3D::half > >();

	return 0;
}

//----------------------------------------------------------------------------//

