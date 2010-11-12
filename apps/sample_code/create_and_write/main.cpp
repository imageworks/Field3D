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

/* Field3D examples - create and write
   
This sample application creates a DenseField object and writes it to disk.

*/

//----------------------------------------------------------------------------//

#include <iostream>
#include <string>

#include <Field3D/DenseField.h>
#include <Field3D/InitIO.h>
#include <Field3D/Field3DFile.h>

//----------------------------------------------------------------------------//

using namespace std;

using namespace Field3D;

//----------------------------------------------------------------------------//

int main(int argc, char **argv)
{
  // Call initIO() to initialize standard I/O methods and load plugins 
  Field3D::initIO();

  DenseField<float>::Ptr field(new DenseField<float>);
  field->name = "hello";
  field->attribute = "world";
  field->setSize(V3i(50, 50, 50));
  field->clear(1.0f);
  field->metadata().setStrMetadata("my_attribute", "my_value");

  Field3DOutputFile out;
  out.create("field3d_file.f3d");
  out.writeScalarLayer<float>(field); 
}

//----------------------------------------------------------------------------//

