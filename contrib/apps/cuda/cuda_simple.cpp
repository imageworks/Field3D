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

#include "Field3D/gpu/DenseFieldGPU.h"
#include "Field3D/gpu/DenseFieldSamplerCuda.h"

#include "Field3D/gpu/SparseFieldGPU.h"
#include "Field3D/gpu/SparseFieldSamplerCuda.h"

#include "Field3D/gpu/NameOf.h"
#include "Field3D/gpu/Traits.h"

//----------------------------------------------------------------------------//
namespace nvcc
{
  template< typename Interp >
  void testDevice( const Field3D::Box3i& dataWindow,
                   Interp& interp );
}

//----------------------------------------------------------------------------//
//! run a test on a field
template< typename FieldType >
void testField()
{
  // identify the corresponding GPU field type
  typedef typename Field3D::Gpu::GpuFieldType< FieldType >::type FieldTypeGPU;

  std::cout << "testing a field of type "
      << Field3D::Gpu::nameOf< FieldTypeGPU >() << std::endl;

  // create a test Field3D field
  typename FieldType::Ptr field( new FieldType );
  field->name = "hello";
  field->attribute = "world";
  field->setSize(Field3D::V3i(TEST_RESOLUTION,
                              TEST_RESOLUTION,
                              TEST_RESOLUTION));

  // fill with random values
  randomValues( -10.0f, 10.0f, *field );
  field->setStrMetadata( "my_attribute", "my_value" );

  // create a GPU field and attach it to the field3d field
  typename FieldTypeGPU::Ptr gpu_field(new FieldTypeGPU);
  gpu_field->setField(field);

  //! get a GPU interpolator for the field
  typename FieldTypeGPU::LinearInterpPtr interp =
      gpu_field->getLinearInterpolatorDevice();
  nvcc::testDevice(field->dataWindow(), *interp);

  std::cout << std::endl;
}

//----------------------------------------------------------------------------//
//! entry point
int main( int argc,
          char **argv )
{
  testField< Field3D::DenseField<float> > ();
  testField< Field3D::SparseField<float> > ();

  return 0;
}
