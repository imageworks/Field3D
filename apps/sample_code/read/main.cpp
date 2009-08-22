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

/* Field3D examples - read

This sample application reads all fields found in a given .f3d file and
prints their types, names and attributes.

*/

//----------------------------------------------------------------------------//

#include <iostream>
#include <string>

#include <Field3D/DenseField.h>
#include <Field3D/MACField.h>
#include <Field3D/SparseField.h>
#include <Field3D/InitIO.h>
#include <Field3D/Field3DFile.h>

//----------------------------------------------------------------------------//

using namespace std;

using namespace Field3D;

//----------------------------------------------------------------------------//

template <typename Data_T>
void readLayersAndPrintInfo(Field3DInputFile &in, const std::string &name)
{
  typedef FIELD3D_VEC3_T<Data_T> VecData_T;
  typedef typename Field<Data_T>::Vec SFieldList;
  typedef typename Field<FIELD3D_VEC3_T<Data_T> >::Vec VFieldList;

  // Note that both scalar and vector calls take the scalar type as argument
  SFieldList sFields = in.readScalarLayers<Data_T>(name);
  VFieldList vFields = in.readVectorLayers<Data_T>(name);

  // Print info about the found fields ---

  if (sFields.size() > 0) {

    for (typename SFieldList::const_iterator i = sFields.begin(); 
         i != sFields.end(); ++i) {

      if (field_dynamic_cast<DenseField<Data_T> >(*i)) {
        cout << "  DenseField" << endl;
      }
      else if (field_dynamic_cast<SparseField<Data_T> >(*i)) {
        cout << "  SparseField" << endl;
      }

      cout << "    Name:      " << (**i).name << endl;
      cout << "    Attribute: " << (**i).attribute << endl;

    }

  } else {
    cout << "  Found no scalar fields" << endl;
  }

  if (vFields.size() > 0) {

    for (typename VFieldList::const_iterator i = vFields.begin(); 
         i != vFields.end(); ++i) {

      if (field_dynamic_cast<DenseField<VecData_T> >(*i)) {
        cout << "  DenseField" << endl;
      }
      else if (field_dynamic_cast<SparseField<VecData_T> >(*i)) {
        cout << "  SparseField" << endl;
      }
      else if (field_dynamic_cast<MACField<VecData_T> >(*i)) {
        cout << "  MACField" << endl;
      }

      cout << "    Name:      " << (**i).name << endl;
      cout << "    Attribute: " << (**i).attribute << endl;

    }

  } else {
    cout << "  Found no vector fields" << endl;
  }

}

//----------------------------------------------------------------------------//

int main(int argc, char **argv)
{
  typedef Field3D::half half;

  // Call initIO() to initialize standard I/O methods and load plugins ---

  Field3D::initIO();

  // Process command line ---

  if (argc < 2) {
    cout << "Usage: read <file> [name]" << endl;
    return 1;
  }

  string filename = string(argv[1]);
  string name;

  if (argc == 3) {
    name = string(argv[2]);
  }

  // Load file ---

  Field3DInputFile in;
  if (!in.open(filename)) {
    cout << "Aborting because of errors" << endl;
    return 1;
  }

  cout << "Reading <half> layers" << endl;
  readLayersAndPrintInfo<half>(in, name);

  cout << "Reading <float> layers" << endl;
  readLayersAndPrintInfo<float>(in, name);

  cout << "Reading <double> layers" << endl;
  readLayersAndPrintInfo<double>(in, name);

}

//----------------------------------------------------------------------------//

