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

#include <fnmatch.h>

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
  vector<string> inputFiles;
  vector<string> names;
  vector<string> attributes;
};

//----------------------------------------------------------------------------//
// Function prototypes
//----------------------------------------------------------------------------//

//! Parses command line options, puts them in Options struct.
Options parseOptions(int argc, char **argv);

//! Prints information about all fields in file.
void printFileInfo(const std::string &filename, const Options &options);

//! Prints the information about a single field
template <typename Data_T>
void printFieldInfo(typename Field<Data_T>::Ptr field, const Options &options);

//! Prints a std::map. Used for metadata. Called from printFieldInfo.
template <typename T>
void printMap(const map<string, T> m, const string &indent);

//! Prints information about a mapping. Called from printFieldInfo.
void printMapping(FieldMapping::Ptr mapping);

//! Pattern matching used for field names and attributes
bool matchString(const std::string &str, const vector<string> &patterns);

//----------------------------------------------------------------------------//
// Function implementations
//----------------------------------------------------------------------------//

int main(int argc, char **argv)
{
  Field3D::initIO();

  Options options = parseOptions(argc, argv);

  BOOST_FOREACH (const string &file, options.inputFiles) {
    printFileInfo(file, options);
  }
}

//----------------------------------------------------------------------------//

Options parseOptions(int argc, char **argv)
{
  namespace po = boost::program_options;

  Options options;

  po::options_description desc("Available options");

  desc.add_options()
    ("help", "Display help")
    ("input-file", po::value<vector<string> >(), "Input files")
    ("name,n", po::value<vector<string> >(), "Load field(s) by name")
    ("attribute,a", po::value<vector<string> >(), "Load field(s) by attribute")
    ;
  
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    
  
  po::positional_options_description p;
  p.add("input-file", -1);
  
  po::store(po::command_line_parser(argc, argv).
            options(desc).positional(p).run(), vm);
  po::notify(vm);
  
  if (vm.count("help")) {
    cout << desc << endl;
    exit(0);
  }

  if (vm.count("input-file"))
  {
    options.inputFiles = vm["input-file"].as<std::vector<std::string> >();
  }
  if (vm.count("name"))
  {
    options.names = vm["name"].as<std::vector<std::string> >();
  }
  if (vm.count("attribute"))
  {
    options.attributes = vm["attribute"].as<std::vector<std::string> >();
  }

  return options;
}

//----------------------------------------------------------------------------//

template <typename T>
void printMap(const map<string, T> m, const string &indent)
{
  typedef pair<string, T> KeyValuePair;

  if (m.size() == 0) {
    cout << indent << "None" <<  endl;
  }

  BOOST_FOREACH(const KeyValuePair &i, m) {
    cout << indent << i.first << " : " << i.second << endl;
  }
}

//----------------------------------------------------------------------------//

void printMapping(FieldMapping::Ptr mapping)
{
  cout << "    Mapping:" << endl;
  cout << "      Type: " << mapping->className() << endl;

  // In the case of a MatrixFieldMapping, we print the local to world matrix.

  MatrixFieldMapping::Ptr matrixMapping = 
    boost::dynamic_pointer_cast<MatrixFieldMapping>(mapping);

  if (matrixMapping) {
    M44d m = matrixMapping->localToWorld();
    cout << "      Local to world transform:" << endl;
    for (int j = 0; j < 4; ++j) {
      cout << "        ";
      for (int i = 0; i < 4; ++i) {
        cout << m[i][j] << " ";
      }
      cout << endl;
    }
  }
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void printFieldInfo(typename Field<Data_T>::Ptr field, const Options &options)
{
  Box3i dataWindow = field->dataWindow();
  Box3i extents = field->extents();

  cout << "  Field: " << endl
       << "    Name:        " << field->name << endl
       << "    Attribute:   " << field->attribute << endl
       << "    Field type:  " << field->className() << endl
       << "    Data type:   " << field->dataTypeString() << endl
       << "    Extents:     " << extents.min << " " << extents.max << endl
       << "    Data window: " << dataWindow.min << " " << dataWindow.max << endl;

  printMapping(field->mapping());

  cout << "    Int metadata:" << endl;
  printMap(field->metadata().intMetadata(), "      ");
  cout << "    Float metadata:" << endl;
  printMap(field->metadata().floatMetadata(), "      ");
  cout << "    V3i metadata:" << endl;
  printMap(field->metadata().vecIntMetadata(), "      ");
  cout << "    V3f metadata:" << endl;
  printMap(field->metadata().vecFloatMetadata(), "      ");
  cout << "    String metadata:" << endl;
  printMap(field->metadata().strMetadata(), "      ");
}

//----------------------------------------------------------------------------//

bool matchString(const std::string &str, const vector<string> &patterns)
{
  // If patterns is empty all strings match
  if (patterns.size() == 0) {
    return true;
  }
  // Check all patterns
  BOOST_FOREACH (const string &pattern, patterns) {
    if (fnmatch(pattern.c_str(), str.c_str(), 0) != FNM_NOMATCH) {
      return true;
    }
  }
  // If no pattern matched return false
  return false;
}

//----------------------------------------------------------------------------//

void printFileInfo(const std::string &filename, const Options &options)
{
  typedef Field3D::half half;

  Field3DInputFile in;

  if (!in.open(filename)) {
    cout << "Error: Couldn't open f3d file: " << filename << endl;
    exit(1);
  }

  cout << "Field3D file: " << filename << endl;

  vector<string> partitions;
  in.getPartitionNames(partitions);

  BOOST_FOREACH (const string &partition, partitions) {

    if (!matchString(partition, options.names)) {
      continue;
    }

    vector<string> scalarLayers, vectorLayers;
    in.getScalarLayerNames(scalarLayers, partition);
    in.getVectorLayerNames(vectorLayers, partition);

    BOOST_FOREACH (const string &scalarLayer, scalarLayers) {

      if (!matchString(scalarLayer, options.attributes)) {
        continue;
      }  

      Field<half>::Vec hScalarFields = 
        in.readScalarLayers<half>(partition, scalarLayer);
      BOOST_FOREACH (Field<half>::Ptr field, hScalarFields) {
        printFieldInfo<half>(field, options);
      }

      Field<float>::Vec fScalarFields = 
        in.readScalarLayers<float>(partition, scalarLayer);
      BOOST_FOREACH (Field<float>::Ptr field, fScalarFields) {
        printFieldInfo<float>(field, options);
      }

      Field<double>::Vec dScalarFields = 
        in.readScalarLayers<double>(partition, scalarLayer);
      BOOST_FOREACH (Field<double>::Ptr field, dScalarFields) {
        printFieldInfo<double>(field, options);
      }

    }

    BOOST_FOREACH (const string &vectorLayer, vectorLayers) {
      
      if (!matchString(vectorLayer, options.attributes)) {
        continue;
      }  

      Field<V3h>::Vec hVectorFields = 
        in.readVectorLayers<half>(partition, vectorLayer);
      BOOST_FOREACH (Field<V3h>::Ptr field, hVectorFields) {
        printFieldInfo<V3h>(field, options);
      }

      Field<V3f>::Vec fVectorFields = 
        in.readVectorLayers<float>(partition, vectorLayer);
      BOOST_FOREACH (Field<V3f>::Ptr field, fVectorFields) {
        printFieldInfo<V3f>(field, options);
      }

      Field<V3d>::Vec dVectorFields = 
        in.readVectorLayers<double>(partition, vectorLayer);
      BOOST_FOREACH (Field<V3d>::Ptr field, dVectorFields) {
        printFieldInfo<V3d>(field, options);
      }

    }
  }

  cout << "  Global metadata" << endl;

  cout << "    Int metadata:" << endl;
  printMap(in.metadata().intMetadata(), "      ");
  cout << "    Float metadata:" << endl;
  printMap(in.metadata().floatMetadata(), "      ");
  cout << "    V3i metadata:" << endl;
  printMap(in.metadata().vecIntMetadata(), "      ");
  cout << "    V3f metadata:" << endl;
  printMap(in.metadata().vecFloatMetadata(), "      ");
  cout << "    String metadata:" << endl;
  printMap(in.metadata().strMetadata(), "      ");

}

//----------------------------------------------------------------------------//

