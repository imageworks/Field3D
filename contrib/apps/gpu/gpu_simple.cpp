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

#include "Field3D/gpu/DenseFieldCuda.h"
#include "Field3D/gpu/DenseFieldSamplerCuda.h"

#include "Field3D/gpu/SparseFieldCuda.h"
#include "Field3D/gpu/SparseFieldSamplerCuda.h"

//----------------------------------------------------------------------------//
namespace nvcc
{
	template< typename Interp >
	void testDevice( const Field3D::Box3i& dataWindow, Interp& interp );
}

//----------------------------------------------------------------------------//
//! run a test on a field
template< typename FieldType >
void testField()
{
	std::cout << "testing a field of type " << nameOf< FieldType > () << std::endl;

	// create a test field
	boost::intrusive_ptr< FieldType > field( new FieldType );
	field->name = "hello";
	field->attribute = "world";
	field->setSize( Field3D::V3i( TEST_RESOLUTION, TEST_RESOLUTION, TEST_RESOLUTION ) );

	// fill with random values
	randomValues( -10.0f, 10.0f, *field );
	field->setStrMetadata( "my_attribute", "my_value" );

	//! get a GPU interpolator for the field
	boost::shared_ptr< typename FieldType::linear_interp_type > interp = field->getLinearInterpolatorDevice();
	nvcc::testDevice( field->dataWindow(), *interp );

	std::cout << std::endl;
}

//----------------------------------------------------------------------------//
//! entry point
int main( 	int argc,
			char **argv )
{
	testField< Field3D::Gpu::DenseFieldCuda< float > > ();
	testField< Field3D::Gpu::SparseFieldCuda< float > > ();

	return 0;
}
