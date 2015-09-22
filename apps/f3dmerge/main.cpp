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

#include <boost/regex.hpp>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>

#include <Field3D/Field3DFile.h>
#include <Field3D/InitIO.h>
#include <Field3D/PatternMatch.h>

//----------------------------------------------------------------------------//

using namespace std;
using namespace Field3D;

//----------------------------------------------------------------------------//
// Options struct
//----------------------------------------------------------------------------//

struct Options {
  Options() 
    : numThreads(1), doOgawa(true)
  { }
  vector<string> inputFiles;
  string         outputFile;
  vector<string> names;
  vector<string> attributes;
  size_t         numThreads;
  bool           doOgawa;
};

//----------------------------------------------------------------------------//
// Function prototypes
//----------------------------------------------------------------------------//

//! Parses command line options, puts them in Options struct.
Options parseOptions(int argc, char **argv);

//! Prints information about all fields in file.
void writeOutput(const std::string &filename, const Options &options,
                 Field3DOutputFile &out);

//----------------------------------------------------------------------------//
// Function implementations
//----------------------------------------------------------------------------//

int main(int argc, char **argv)
{
  Field3D::initIO();

  Options options = parseOptions(argc, argv);

  // Set num threads ---

  Field3D::setNumIOThreads(options.numThreads);
  
  // Set HDF5/Ogawa ---

  Field3DOutputFile::useOgawa(options.doOgawa);

  if (options.doOgawa) {
    cout << "Converting to Ogawa." << endl;
  } else {
    cout << "Converting to HDF5." << endl;
  }

  // Open output file ---

  Field3DOutputFile out;
  if (!out.create(options.outputFile)) {
    cout << "ERROR: Couldn't create output file: " 
         << options.outputFile << endl;
    return 1;
  }

  // Write inputs to output ---

  BOOST_FOREACH (const string &file, options.inputFiles) {
    writeOutput(file, options, out);
  }
}

//----------------------------------------------------------------------------//

Options parseOptions(int argc, char **argv)
{
  namespace po = boost::program_options;

  Options options;

  po::options_description desc("Available options");

  desc.add_options()
    ("help,h", "Display help")
    ("input-file,i", po::value<vector<string> >(), "Input files")
    ("name,n", po::value<vector<string> >(), "Load field(s) by name")
    ("attribute,a", po::value<vector<string> >(), "Load field(s) by attribute")
    ("ogawa,g", po::value<bool>(), "Whether to output an Ogawa file.")
    ("num-threads,t", po::value<size_t>(), "Number of threads to use")
    ("output-file,o", po::value<string>(), "Output file")
    ;
  
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
  } catch(...) {
    cerr << "Unknown command line option.\n";
    cout << desc << endl;
    exit(1);
  }
  po::notify(vm);
  
  po::positional_options_description p;
  p.add("input-file", -1);
  
  try {
    po::store(po::command_line_parser(argc, argv).
              options(desc).positional(p).run(), vm);
  } catch(...) {
    cerr << "Unknown command line option.\n";
    cout << desc << endl;
    exit(1);
  }
  po::notify(vm);
  
  if (vm.count("help")) {
    cout << desc << endl;
    exit(0);
  }

  if (vm.count("input-file")) {
    options.inputFiles = vm["input-file"].as<std::vector<std::string> >();
  }
  if (vm.count("ogawa")) {
    options.doOgawa = vm["ogawa"].as<bool>();
  }
  if (vm.count("num-threads")) {
    options.numThreads = vm["num-threads"].as<size_t>();
  }
  if (vm.count("name")) {
    options.names = vm["name"].as<std::vector<std::string> >();
  }
  if (vm.count("attribute")) {
    options.attributes = vm["attribute"].as<std::vector<std::string> >();
  }
  if (vm.count("output-file")) {
    options.outputFile = vm["output-file"].as<std::string>();
  }

  return options;
}

//----------------------------------------------------------------------------//

template <typename T>
void writeField(const typename Field<Imath::Vec3<T> >::Ptr f, 
                Field3DOutputFile &out)
{
  out.writeVectorLayer<T>(f);
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void writeField(const typename Field<Data_T>::Ptr &f, Field3DOutputFile &out)
{
  out.writeScalarLayer<Data_T>(f);
}

//----------------------------------------------------------------------------//

void writeOutput(const std::string &filename, const Options &options,
                 Field3DOutputFile &out)
{
  typedef Field3D::half half;

  Field3DInputFile in;

  if (!in.open(filename)) {
    cout << "Error: Couldn't open f3d file: " << filename << endl;
    exit(1);
  }

  cout << "Opening file: " << endl << "  " << filename << endl;

  vector<string> partitions;
  in.getPartitionNames(partitions);

  BOOST_FOREACH (const string &partition, partitions) {

    if (!match(partition, options.names)) {
      continue;
    }

    vector<string> scalarLayers, vectorLayers;
    in.getScalarLayerNames(scalarLayers, partition);
    in.getVectorLayerNames(vectorLayers, partition);

    BOOST_FOREACH (const string &scalarLayer, scalarLayers) {

      if (!match(scalarLayer, options.attributes)) {
        continue;
      }  

      Field<half>::Vec hScalarFields = 
        in.readScalarLayers<half>(partition, scalarLayer);
      BOOST_FOREACH (Field<half>::Ptr field, hScalarFields) {
        writeField<half>(field, out);
      }

      Field<float>::Vec fScalarFields = 
        in.readScalarLayers<float>(partition, scalarLayer);
      BOOST_FOREACH (Field<float>::Ptr field, fScalarFields) {
        writeField<float>(field, out);
      }

      Field<double>::Vec dScalarFields = 
        in.readScalarLayers<double>(partition, scalarLayer);
      BOOST_FOREACH (Field<double>::Ptr field, dScalarFields) {
        writeField<double>(field, out);
      }

    }

    BOOST_FOREACH (const string &vectorLayer, vectorLayers) {
      
      if (!match(vectorLayer, options.attributes)) {
        continue;
      }  

      Field<V3h>::Vec hVectorFields = 
        in.readVectorLayers<half>(partition, vectorLayer);
      BOOST_FOREACH (Field<V3h>::Ptr field, hVectorFields) {
        writeField<V3h>(field, out);
      }

      Field<V3f>::Vec fVectorFields = 
        in.readVectorLayers<float>(partition, vectorLayer);
      BOOST_FOREACH (Field<V3f>::Ptr field, fVectorFields) {
        writeField<V3f>(field, out);
      }

      Field<V3d>::Vec dVectorFields = 
        in.readVectorLayers<double>(partition, vectorLayer);
      BOOST_FOREACH (Field<V3d>::Ptr field, dVectorFields) {
        writeField<V3d>(field, out);
      }

    }
  }

}

//----------------------------------------------------------------------------//

