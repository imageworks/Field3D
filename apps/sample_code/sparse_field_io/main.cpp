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

#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/timer.hpp>

#include <Field3D/SparseField.h>
#include <Field3D/SparseFile.h>
#include <Field3D/Field3DFile.h>
#include <Field3D/FieldInterp.h>
#include <Field3D/InitIO.h>
#include <Field3D/Log.h>

//----------------------------------------------------------------------------//

using namespace boost;
using namespace std;

using namespace Field3D;

//----------------------------------------------------------------------------//

//! Converts any class with operator<< to a string using boost::lexical_cast
template <class T>
std::string str(const T& t)
{
  return boost::lexical_cast<std::string>(t);
}

//----------------------------------------------------------------------------//

int main(int argc, char **argv) 
{
  std::string filename("test_file.f3d");
  std::string attribName("attrib");

  // Initialize IO
  initIO();

  int numFields = 5;

  if (argc == 2) {
    try {
      numFields = lexical_cast<int>(argv[1]);
    } 
    catch (boost::bad_lexical_cast &e) {
      Msg::print("Couldn't parse integer number. Aborting");
      exit(1);
    }
  } else {
    // No voxel res given
    Msg::print("Usage: " + str(argv[0]) + 
               " <num_fields>");
    Msg::print("Got no number of fields. Using default.");
  }

  // Create fields ---

  Msg::print("Creating " + str(numFields) + " fields");

  SparseFieldf::Vec fields;

  for (int i = 0; i < numFields; i++) {
    // Create
    SparseFieldf::Ptr field(new SparseFieldf);
    field->setSize(V3i(128));
    field->name = str(i);
    field->attribute = attribName;
    // Fill with values
    SparseFieldf::iterator fi = field->begin(), fend = field->end();
    for (; fi != fend; ++fi) {
      *fi = i;
    }
    fields.push_back(field);
  }

  // Write fields ---

  Msg::print("Writing fields");

  Field3DOutputFile out;
  out.create(filename);

  for (int i = 0; i < numFields; i++) {
    out.writeScalarLayer<float>(fields[i]);
  }

  // Read with dynamic loading ---

  Msg::print("Reading fields");

  SparseFileManager::singleton().setLimitMemUse(true);

  Field<float>::Vec loadedFields;

  Field3DInputFile in;
  in.open(filename);

  for (int i = 0; i < numFields; i++) {  
    Field<float>::Vec fields = in.readScalarLayers<float>(str(i), attribName);
    if (fields.size() != 1) {
      Msg::print("Got the wrong # of fields. Aborting.");
      exit(1);
    }
    loadedFields.push_back(fields[0]);
  }

  // Compare fields ---

  Msg::print("Comparing fields");

  FIELD3D_RAND48 rng(10512);

  Msg::print("  Mem use before access: ");

  for (int i = 0; i < numFields; i++) {
    Msg::print("    Field " + str(i) + "       : " + 
               str(fields[i]->memSize()));
    Msg::print("    Loaded field " + str(i) + ": " + 
               str(loadedFields[i]->memSize()));
  }

  LinearFieldInterp<float> interp;
  for (int sample = 0; sample < 1000; sample++) {  
    V3d lsP(rng.nextf(), rng.nextf(), rng.nextf()), vsP;
    for (int i = 0; i < numFields; i++) {
      fields[i]->mapping()->localToVoxel(lsP, vsP);
      float value = interp.sample(*fields[i], vsP);
      float loadedValue = interp.sample(*loadedFields[i], vsP);
      if (value != loadedValue) {
        Msg::print("Got a bad value at " + str(vsP));
        exit(1);
      }
    }
  }

  Msg::print("  Mem use after access: ");

  for (int i = 0; i < numFields; i++) {
    Msg::print("    Field " + str(i) + "       : " + 
               str(fields[i]->memSize()));
    Msg::print("    Loaded field " + str(i) + ": " + 
               str(loadedFields[i]->memSize()));
  }

}

//----------------------------------------------------------------------------//
