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

/* Field3D examples - mixed types

This sample application creates multiple fields and writes them all to 
the same file. 

*/

//----------------------------------------------------------------------------//

#include <iostream>
#include <string>

#include <Field3D/DenseField.h>
#include <Field3D/MACField.h>
#include <Field3D/SparseField.h>

#include <Field3D/InitIO.h>
#include <Field3D/Field3DFile.h>

#include <Field3D/Types.h>

//----------------------------------------------------------------------------//

using namespace std;

using namespace Field3D;

//----------------------------------------------------------------------------//

int main(int argc, char **argv)
{
  typedef Field3D::half half; 

  // Call initIO() to initialize standard I/O methods and load plugins ---

  Field3D::initIO();

  // Create a set of fields with different types and bit depths ---

  // ... First a DenseField<half>

  DenseField<half>::Ptr denseField(new DenseField<half>);
  denseField->name = "density_source";
  denseField->attribute = "density";
  denseField->setSize(V3i(50, 50, 50));
  denseField->clear(0.0f);

  // ... Then two SparseFields to make up a moving levelset

  SparseField<float>::Ptr sparseField(new SparseField<float>);
  sparseField->name = "character";
  sparseField->attribute = "levelset";
  sparseField->setSize(V3i(250, 250, 250));
  sparseField->clear(0.0f);
  
  SparseField<V3f>::Ptr sparseVField(new SparseField<V3f>);
  sparseVField->name = "character";
  sparseVField->attribute = "v";
  sparseVField->setSize(V3i(50, 50, 50));
  sparseVField->clear(V3f(0.0f));

  // ... Finally a MACField<V3f>, using the typedefs

  MACField3f::Ptr macField(new MACField3f);
  macField->name = "simulation";
  macField->attribute = "v";
  macField->setSize(V3i(120, 50, 100));
  macField->clear(V3f(0.0f));
  
  // Write the output ---

  Field3DOutputFile out;
  out.create("mixed_file.f3d");
  out.writeScalarLayer<half>(denseField); 
  out.writeScalarLayer<float>(sparseField); 
  out.writeVectorLayer<float>(sparseVField); 
  out.writeVectorLayer<float>(macField); 
}

//----------------------------------------------------------------------------//

