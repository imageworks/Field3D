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

#include "Field3D/gpu/Traits.h"
#include "Field3D/gpu/DataAccessorCuda.h"
#include "Field3D/gpu/FieldInterpCuda.h"
#include "Field3D/gpu/DenseFieldSamplerCuda.h"
#include "Field3D/gpu/SparseFieldSamplerCuda.h"

#include "Field3D/Types.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace Field3D::Gpu;

namespace nvcc
{
	template< typename Interp >
	void testDevice( const Field3D::Box3i& dataWindow, Interp& interp )
	{
		GlobalMemAccessor<typename Interp::value_type> ac;

		int sample_count = 10;
		// set up some random locations
		thrust::host_vector< Vec3f > host_p(sample_count);
        randomLocations(dataWindow, host_p);
		// copy to device
		thrust::device_vector<Vec3f> device_p = host_p;
		// allocate result vector
		thrust::device_vector<typename Interp::sample_type>
		  device_result( sample_count, 0.0f );

		// make our user defined functor
		RandomSampleFunctor< Interp,
		  GlobalMemAccessor< typename Interp::value_type > >
		  f(ac,
		    interp,
		    thrust::raw_pointer_cast(&device_p[0]),
			thrust::raw_pointer_cast(&device_result[0]));

		// execute it
		RunFunctor(f, sample_count, thrust::device_space_tag());

		// copy the result back to host for logging
		thrust::host_vector< typename Interp::sample_type >
		  host_result = device_result;
		std::cout << "gpu result: ";
		dump( host_result );
	}

	// explicit instantiation for different field types
	template
	void testDevice< LinearFieldInterp< DenseFieldSampler<float, float> > >
	( const Field3D::Box3i& dataWindow,
	  LinearFieldInterp< DenseFieldSampler<float, float> >& );

	template
	void testDevice< LinearFieldInterp< SparseFieldSampler<float, float> > >
	( const Field3D::Box3i& dataWindow,
	  LinearFieldInterp< SparseFieldSampler<float, float> >& );
}
