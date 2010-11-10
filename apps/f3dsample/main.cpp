//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks Inc.
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

#include <iostream>
#include <vector>
#include <map>
#include <string>

#include <boost/program_options.hpp>
#include <boost/foreach.hpp>

#include <Field3D/DenseField.h>
#include <Field3D/MACField.h>
#include <Field3D/SparseField.h>
#include <Field3D/InitIO.h>
#include <Field3D/Field3DFile.h>

//----------------------------------------------------------------------------//

using namespace std;
using namespace Field3D;

//----------------------------------------------------------------------------//
// Options struct
//----------------------------------------------------------------------------//

struct Options {
  Options()
    : name("field_name"), attribute("field_attribute"),
      resolution(64), fieldType("DenseField"), fill("small_sphere"),
      bits(32), isVectorField(false)
  { }
  string     filename;
  string     name;
  string     attribute;
  Imath::V3i resolution;
  string     fieldType;
  string     fill;
  int        bits;
  bool       isVectorField;
};

//----------------------------------------------------------------------------//
// Function prototypes
//----------------------------------------------------------------------------//

Options parseOptions(int argc, char **argv);

void createField(const Options &options);

template <typename Data_T>
void createConcreteScalarField(const Options &options);

template <typename Data_T>
void createConcreteVectorField(const Options &options);

void setCommon(const FieldRes::Ptr field, const Options &options);

//----------------------------------------------------------------------------//
// Function implementations
//----------------------------------------------------------------------------//

int main(int argc, char **argv)
{
  Field3D::initIO();

  Options options = parseOptions(argc, argv);

  createField(options);
}

//----------------------------------------------------------------------------//

Options parseOptions(int argc, char **argv)
{
  namespace po = boost::program_options;

  Options options;

  po::options_description desc("Available options");

  desc.add_options()
    ("help", "Display help")
    ("output-file", po::value<vector<string> >(), "Output file(s)")
    ("name,n", po::value<string>(), "Field name")
    ("attribute,a", po::value<string>(), "Field attribute")
    ("type,t", po::value<string>(), "Field type (DenseField/SparseField/MACField)")
    ("fill,f", po::value<string>(), "Fill with (full_sphere/small_sphere)")
    ("xres,x", po::value<int>(), "X resolution")
    ("yres,y", po::value<int>(), "Y resolution")
    ("zres,z", po::value<int>(), "Z resolution")
    ("bits,b", po::value<int>(), "Bit depth (16/32/64)")
    ("vector,v", "Whether to create a vector field")    
    ;
  
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    
  
  po::positional_options_description p;
  p.add("output-file", -1);
  
  po::store(po::command_line_parser(argc, argv).
            options(desc).positional(p).run(), vm);
  po::notify(vm);
  
  if (vm.count("help")) {
    cout << desc << endl;
    exit(0);
  }

  if (vm.count("output-file"))
  {
    if (vm.count("output-file") > 1) {
      cout << "WARNING: Got more than one output filename. "
           << "First entry will be used." << endl;
    }
    options.filename = vm["output-file"].as<vector<string> >()[0];
  } else {
    cout << "No output file specified." << endl;
    exit(0);
  }

  if (vm.count("name"))
  {
    options.name = vm["name"].as<string>();
  }
  if (vm.count("attribute"))
  {
    options.attribute = vm["attribute"].as<string>();
  }
  if (vm.count("type"))
  {
    options.fieldType = vm["type"].as<string>();
  }
  if (vm.count("xres"))
  {
    options.resolution.x = vm["xres"].as<int>();
  }
  if (vm.count("yres"))
  {
    options.resolution.y = vm["yres"].as<int>();
  }
  if (vm.count("zres"))
  {
    options.resolution.z = vm["zres"].as<int>();
  }
  if (vm.count("bits"))
  {
    options.bits = vm["bits"].as<int>();
  }
  if (vm.count("vector"))
  {
    options.isVectorField = true;
  }

  return options;
}

//----------------------------------------------------------------------------//

void createField(const Options &options)
{
  if (options.isVectorField) {
    switch (options.bits) {
    case 64:
      createConcreteVectorField<double>(options);
      break;
    case 32:
      createConcreteVectorField<float>(options);
      break;
    case 16:
    default:
      createConcreteVectorField<half>(options);
      break;
    }
  } else {
    switch (options.bits) {
    case 64:
      createConcreteScalarField<double>(options);
      break;
    case 32:
      createConcreteScalarField<float>(options);
      break;
    case 16:
    default:
      createConcreteScalarField<half>(options);
      break;
    }
  }
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void createConcreteScalarField(const Options &options)
{
  typedef typename ResizableField<Data_T>::Ptr Ptr;
  Ptr field;

  if (options.fieldType == "SparseField") {
    field = Ptr(new SparseField<Data_T>);
  } else {
    field = Ptr(new DenseField<Data_T>);
  }

  field->setSize(options.resolution);
  setCommon(field, options);

  Field3DOutputFile out;
  out.create(options.filename);
  out.writeScalarLayer<Data_T>(field);
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void createConcreteVectorField(const Options &options)
{
  typedef typename ResizableField<FIELD3D_VEC3_T<Data_T> >::Ptr Ptr;
  Ptr field;

  if (options.fieldType == "SparseField") {
    field = Ptr(new SparseField<FIELD3D_VEC3_T<Data_T> >);
  } else if (options.fieldType == "MACField") {
    field = Ptr(new MACField<FIELD3D_VEC3_T<Data_T> >);
  } else {
    field = Ptr(new DenseField<FIELD3D_VEC3_T<Data_T> >); 
  }

  field->setSize(options.resolution);  
  setCommon(field, options);

  Field3DOutputFile out;
  out.create(options.filename);
  out.writeVectorLayer<Data_T>(field);
}

//----------------------------------------------------------------------------//

void setCommon(const FieldRes::Ptr field, const Options &options)
{
  field->name = options.name;
  field->attribute = options.attribute; 
  field->setFloatMetadata("float_metadata", 1.0f);
  field->setVecFloatMetadata("vec_float_metadata", V3f(1.0f));
  field->setIntMetadata("int_metadata", 1);
  field->setVecIntMetadata("vec_int_metadata", V3i(1));
  field->setStrMetadata("str_metadata", "string");

  M44d localToWorld;
  localToWorld.setScale(options.resolution);
  localToWorld *= M44d().setTranslation(V3d(1.0, 2.0, 3.0));

  MatrixFieldMapping::Ptr mapping(new MatrixFieldMapping);
  mapping->setLocalToWorld(localToWorld);
  field->setMapping(mapping);
}

//----------------------------------------------------------------------------//
