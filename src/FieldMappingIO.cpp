//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks Inc
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

/*! \file FieldMappingIO.cpp
  \brief Contains the FieldMapping base class and the NullFieldMapping and
  MatrixFieldMapping subclass implementations.
*/

//----------------------------------------------------------------------------//

#include "Hdf5Util.h"

#include "FieldMappingIO.h"
#include "OgIO.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Field3D namespaces
//----------------------------------------------------------------------------//

using namespace std;
using namespace Exc;
using namespace Hdf5Util;

//----------------------------------------------------------------------------//

namespace {
  //! \todo This is duplicated in FieldMapping.cpp. Fix.
  const string k_nullMappingName("NullFieldMapping");
  //! \todo This is duplicated in FieldMapping.cpp. Fix.
  const string k_matrixMappingName("MatrixFieldMapping");
  const string k_frustumMappingName("FrustumFieldMapping");

  const string k_nullMappingDataName("NullFieldMapping data");
  
  const string k_matrixMappingDataName("MatrixFieldMapping data");
  const string k_matrixMappingNumSamples("num_time_samples");
  const string k_matrixMappingTime("time_");
  const string k_matrixMappingMatrix("matrix_");

  const string k_frustumMappingNumSamples("num_time_samples");
  const string k_frustumMappingTime("time_");
  const string k_frustumMappingScreenMatrix("screen_to_world_");
  const string k_frustumMappingCameraMatrix("camera_to_world_");
  const string k_frustumMappingZDistribution("z_distribution");
}

//----------------------------------------------------------------------------//

FieldMapping::Ptr
NullFieldMappingIO::read(hid_t mappingGroup)
{
  string nfmData;
  if (!readAttribute(mappingGroup, k_nullMappingDataName, nfmData)) {
    Msg::print(Msg::SevWarning, "Couldn't read attribute " + k_nullMappingDataName);
    return NullFieldMapping::Ptr();
  }
  return NullFieldMapping::Ptr(new NullFieldMapping);
}

//----------------------------------------------------------------------------//

FieldMapping::Ptr
NullFieldMappingIO::read(const OgIGroup &mappingGroup)
{
  OgIAttribute<string> data = 
    mappingGroup.findAttribute<string>(k_nullMappingDataName);
  if (!data.isValid()) {
    Msg::print(Msg::SevWarning, "Couldn't read attribute " + 
               k_nullMappingDataName);
    return NullFieldMapping::Ptr();
  }
  return NullFieldMapping::Ptr(new NullFieldMapping);
}

//----------------------------------------------------------------------------//

bool
NullFieldMappingIO::write(hid_t mappingGroup, FieldMapping::Ptr /* nm */)
{
  string nfmAttrData("NullFieldMapping has no data");
  if (!writeAttribute(mappingGroup, k_nullMappingDataName, nfmAttrData)) {
    Msg::print(Msg::SevWarning, "Couldn't add attribute " + k_nullMappingDataName);
    return false;
  }
  return true;
}

//----------------------------------------------------------------------------//

bool
NullFieldMappingIO::write(OgOGroup &mappingGroup, FieldMapping::Ptr /* nm */)
{
  string nfmAttrData("NullFieldMapping has no data");
  OgOAttribute<string> data(mappingGroup, k_nullMappingDataName, nfmAttrData);
  return true;
}

//----------------------------------------------------------------------------//

std::string NullFieldMappingIO::className() const
{ 
  return k_nullMappingName; 
}

//----------------------------------------------------------------------------//
// MatrixFieldMapping
//----------------------------------------------------------------------------//

FieldMapping::Ptr
MatrixFieldMappingIO::read(hid_t mappingGroup)
{
  M44d mtx;
  int numSamples=0;

  MatrixFieldMapping::Ptr mm(new MatrixFieldMapping);
  
  // For backward compatibility, we first try to read the non-time-varying
  // mapping.

  try {
    readAttribute(mappingGroup, k_matrixMappingDataName, 16, mtx.x[0][0]);
    mm->setLocalToWorld(mtx);
    return mm;
  } 
  catch (...) {
    // Do nothing
  }

  // If we didn't find the non-time-varying matrix data then we attempt
  // to read time samples

  try {
    if (!readAttribute(mappingGroup, k_matrixMappingNumSamples, 1, numSamples)) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + 
                 k_matrixMappingNumSamples);
      return FieldMapping::Ptr();
    }
  } catch (...) {
    //do nothing
  }

  for (int i = 0; i < numSamples; ++i) {
    float time;
    string timeAttr = k_matrixMappingTime + boost::lexical_cast<string>(i);
    string matrixAttr = k_matrixMappingMatrix + boost::lexical_cast<string>(i);
    if (!readAttribute(mappingGroup, timeAttr, 1, time)) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + timeAttr);
      return FieldMapping::Ptr();
    }
    std::vector<unsigned int> attrSize;
    attrSize.assign(2,4);

    if (!readAttribute(mappingGroup, matrixAttr, attrSize, mtx.x[0][0])) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + matrixAttr);
      return FieldMapping::Ptr();
    }
    mm->setLocalToWorld(time, mtx);
  }
  
  return mm;
}

//----------------------------------------------------------------------------//

FieldMapping::Ptr
MatrixFieldMappingIO::read(const OgIGroup &mappingGroup)
{
  M44d mtx;
  int numSamples = 0;

  MatrixFieldMapping::Ptr mm(new MatrixFieldMapping);

  try {
    OgIAttribute<int> numSamplesAttr = 
      mappingGroup.findAttribute<int>(k_matrixMappingNumSamples);
    if (!numSamplesAttr.isValid()) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + 
                 k_matrixMappingNumSamples);
      return FieldMapping::Ptr();
    }
    numSamples = numSamplesAttr.value();
  } catch (...) {
    //do nothing
  }

  for (int i = 0; i < numSamples; ++i) {
    string timeAttr = k_matrixMappingTime + boost::lexical_cast<string>(i);
    string matrixAttr = k_matrixMappingMatrix + boost::lexical_cast<string>(i);
    // Read time
    OgIAttribute<float32_t> time = 
      mappingGroup.findAttribute<float32_t>(timeAttr);
    if (!time.isValid()) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + timeAttr);
      return FieldMapping::Ptr();
    }
    // Read matrix
    OgIAttribute<mtx64_t> mtx = 
      mappingGroup.findAttribute<mtx64_t>(matrixAttr);
    if (!mtx.isValid()) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + matrixAttr);
      return FieldMapping::Ptr();
    }
    mm->setLocalToWorld(time.value(), mtx.value());
  }
  
  return mm;
}

//----------------------------------------------------------------------------//

bool
MatrixFieldMappingIO::write(hid_t mappingGroup, FieldMapping::Ptr mapping)
{
  typedef MatrixFieldMapping::MatrixCurve::SampleVec SampleVec;

  MatrixFieldMapping::Ptr mm =
    FIELD_DYNAMIC_CAST<MatrixFieldMapping>(mapping);

  if (!mm) {
    Msg::print(Msg::SevWarning, "Couldn't get MatrixFieldMapping from pointer");
    return false;
  }

  // First write number of time samples

  const SampleVec &samples = mm->localToWorldSamples();
  int numSamples = static_cast<int>(samples.size());

  if (!writeAttribute(mappingGroup, k_matrixMappingNumSamples, 1, numSamples)) {
    Msg::print(Msg::SevWarning, "Couldn't add attribute " + 
               k_matrixMappingNumSamples);
    return false;
  }

  // Then write each sample

  for (int i = 0; i < numSamples; ++i) {
    string timeAttr = k_matrixMappingTime + boost::lexical_cast<string>(i);
    string matrixAttr = k_matrixMappingMatrix + boost::lexical_cast<string>(i);
    if (!writeAttribute(mappingGroup, timeAttr, 1, samples[i].first)) {
      Msg::print(Msg::SevWarning, "Couldn't add attribute " + timeAttr);
      return false;
    }
    std::vector<unsigned int> attrSize;
    attrSize.assign(2,4);
    if (!writeAttribute(mappingGroup, matrixAttr, attrSize, 
                        samples[i].second.x[0][0])) {
      Msg::print(Msg::SevWarning, "Couldn't add attribute " + matrixAttr);
      return false;
    }
  }

  return true;
}

//----------------------------------------------------------------------------//

bool
MatrixFieldMappingIO::write(OgOGroup &mappingGroup, FieldMapping::Ptr mapping)
{
  typedef MatrixFieldMapping::MatrixCurve::SampleVec SampleVec;

  MatrixFieldMapping::Ptr mm =
    FIELD_DYNAMIC_CAST<MatrixFieldMapping>(mapping);

  if (!mm) {
    Msg::print(Msg::SevWarning, "Couldn't get MatrixFieldMapping from pointer");
    return false;
  }

  // First write number of time samples

  const SampleVec &samples    = mm->localToWorldSamples();
  const int        numSamples = static_cast<int>(samples.size());

  OgOAttribute<int> numSamplesAttr(mappingGroup, k_matrixMappingNumSamples,
                                   numSamples);

  // Then write each sample

  for (int i = 0; i < numSamples; ++i) {
    // Attribute names
    const string timeAttr   = 
      k_matrixMappingTime + boost::lexical_cast<string>(i);
    const string matrixAttr = 
      k_matrixMappingMatrix + boost::lexical_cast<string>(i);
    OgOAttribute<float32_t> time(mappingGroup, timeAttr, samples[i].first);
    OgOAttribute<mtx64_t> mtx (mappingGroup, matrixAttr, samples[i].second);
  }

  return true;
}

//----------------------------------------------------------------------------//

std::string MatrixFieldMappingIO::className() const
{ 
  return k_matrixMappingName;
}

//----------------------------------------------------------------------------//
// FrustumFieldMapping
//----------------------------------------------------------------------------//

FieldMapping::Ptr
FrustumFieldMappingIO::read(hid_t mappingGroup)
{
  float time;
  M44d ssMtx, csMtx;
  int numSamples=0;

  FrustumFieldMapping::Ptr fm(new FrustumFieldMapping);
  
  // Read number of time samples

  try {
    if (!readAttribute(mappingGroup, k_frustumMappingNumSamples, 1, numSamples)) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + 
                 k_frustumMappingNumSamples);
      return FieldMapping::Ptr();
    }
  } catch (...) {
    //do nothing
  }

  // Read each time sample

  for (int i = 0; i < numSamples; ++i) {
    string timeAttr = k_frustumMappingTime + boost::lexical_cast<string>(i);
    string ssAttr = k_frustumMappingScreenMatrix + boost::lexical_cast<string>(i);
    string csAttr = k_frustumMappingCameraMatrix + boost::lexical_cast<string>(i);
    if (!readAttribute(mappingGroup, timeAttr, 1, time)) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + timeAttr);
      return FieldMapping::Ptr();
    }
    std::vector<unsigned int> attrSize;
    attrSize.assign(2,4);

    if (!readAttribute(mappingGroup, ssAttr, attrSize, ssMtx.x[0][0])) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + ssAttr);
      return FieldMapping::Ptr();
    }
    if (!readAttribute(mappingGroup, csAttr, attrSize, csMtx.x[0][0])) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + csAttr);
      return FieldMapping::Ptr();
    }

    fm->setTransforms(time, ssMtx, csMtx);
  }


  // Read Z distribution

  int distInt;
  FrustumFieldMapping::ZDistribution dist;
  
  try {
    if (!readAttribute(mappingGroup, k_frustumMappingZDistribution, 1, distInt)) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + 
                 k_frustumMappingZDistribution);
      return FieldMapping::Ptr();
    }
    dist = static_cast<FrustumFieldMapping::ZDistribution>(distInt); 
  } catch (...) {
    dist = FrustumFieldMapping::PerspectiveDistribution;
  }

  fm->setZDistribution(dist);

  return fm;
}

//----------------------------------------------------------------------------//

FieldMapping::Ptr
FrustumFieldMappingIO::read(const OgIGroup &mappingGroup)
{
  int numSamples = 0;

  FrustumFieldMapping::Ptr fm(new FrustumFieldMapping);
  
  // Read number of time samples

  try {
    OgIAttribute<int> numSamplesAttr = 
      mappingGroup.findAttribute<int>(k_frustumMappingNumSamples);
    if (!numSamplesAttr.isValid()) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + 
                 k_frustumMappingNumSamples);
      return FieldMapping::Ptr();
    }
  } catch (...) {
    //do nothing
  }

  // Read each time sample

  for (int i = 0; i < numSamples; ++i) {
    // Attribute names
    string timeAttr = k_frustumMappingTime + boost::lexical_cast<string>(i);
    string ssAttr = k_frustumMappingScreenMatrix + boost::lexical_cast<string>(i);
    string csAttr = k_frustumMappingCameraMatrix + boost::lexical_cast<string>(i);
    // Read time
    OgIAttribute<float> time = 
      mappingGroup.findAttribute<float>(timeAttr);
    if (!time.isValid()) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + timeAttr);
      return FieldMapping::Ptr();
    }
    // Read matrices
    OgIAttribute<mtx64_t> ssMtx = 
      mappingGroup.findAttribute<mtx64_t>(ssAttr);
    OgIAttribute<mtx64_t> csMtx = 
      mappingGroup.findAttribute<mtx64_t>(csAttr);
    if (!ssMtx.isValid()) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + ssAttr);
      return FieldMapping::Ptr();
    }
    if (!csMtx.isValid()) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + csAttr);
      return FieldMapping::Ptr();
    }

    fm->setTransforms(time.value(), ssMtx.value(), csMtx.value());
  }


  // Read Z distribution

  FrustumFieldMapping::ZDistribution dist;
  
  try {
    OgIAttribute<int> zDist = 
      mappingGroup.findAttribute<int>(k_frustumMappingZDistribution);
    if (!zDist.isValid()) {
      Msg::print(Msg::SevWarning, "Couldn't read attribute " + 
                 k_frustumMappingZDistribution);
      return FieldMapping::Ptr();
    }
    dist = static_cast<FrustumFieldMapping::ZDistribution>(zDist.value()); 
  } catch (...) {
    dist = FrustumFieldMapping::PerspectiveDistribution;
  }

  fm->setZDistribution(dist);

  return fm;
}

//----------------------------------------------------------------------------//

bool
FrustumFieldMappingIO::write(hid_t mappingGroup, FieldMapping::Ptr mapping)
{
  typedef FrustumFieldMapping::MatrixCurve::SampleVec SampleVec;

  FrustumFieldMapping::Ptr fm =
    FIELD_DYNAMIC_CAST<FrustumFieldMapping>(mapping);

  if (!fm) {
    Msg::print(Msg::SevWarning, "Couldn't get FrustumFieldMapping from pointer");
    return false;
  }

  // First write number of time samples

  const SampleVec &ssSamples = fm->screenToWorldSamples();
  const SampleVec &csSamples = fm->cameraToWorldSamples();
  int numSamples = static_cast<int>(ssSamples.size());

  if (!writeAttribute(mappingGroup, k_frustumMappingNumSamples, 1, numSamples)) {
    Msg::print(Msg::SevWarning, "Couldn't add attribute " + 
               k_frustumMappingNumSamples);
    return false;
  }

  // Then write each sample

  for (int i = 0; i < numSamples; ++i) {
    string timeAttr = k_frustumMappingTime + boost::lexical_cast<string>(i);
    string ssAttr = k_frustumMappingScreenMatrix + boost::lexical_cast<string>(i);
    string csAttr = k_frustumMappingCameraMatrix + boost::lexical_cast<string>(i);
    if (!writeAttribute(mappingGroup, timeAttr, 1, ssSamples[i].first)) {
      Msg::print(Msg::SevWarning, "Couldn't add attribute " + timeAttr);
      return false;
    }

    std::vector<unsigned int> attrSize;
    attrSize.assign(2,4);

    if (!writeAttribute(mappingGroup, ssAttr,attrSize,
                        ssSamples[i].second.x[0][0])) {
      Msg::print(Msg::SevWarning, "Couldn't add attribute " + ssAttr);
      return false;
    }
    if (!writeAttribute(mappingGroup, csAttr, attrSize,
                        csSamples[i].second.x[0][0])) {
      Msg::print(Msg::SevWarning, "Couldn't add attribute " + csAttr);
      return false;
    }
  }

  // Write distribution type

  int dist = static_cast<int>(fm->zDistribution());

  if (!writeAttribute(mappingGroup, k_frustumMappingZDistribution, 1, dist)) {
    Msg::print(Msg::SevWarning, "Couldn't add attribute " + 
               k_frustumMappingNumSamples);
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------//

bool
FrustumFieldMappingIO::write(OgOGroup &mappingGroup, FieldMapping::Ptr mapping)
{
  typedef FrustumFieldMapping::MatrixCurve::SampleVec SampleVec;

  FrustumFieldMapping::Ptr fm =
    FIELD_DYNAMIC_CAST<FrustumFieldMapping>(mapping);

  if (!fm) {
    Msg::print(Msg::SevWarning, 
               "Couldn't get FrustumFieldMapping from pointer");
    return false;
  }

  // First write number of time samples ---

  const SampleVec &ssSamples  = fm->screenToWorldSamples();
  const SampleVec &csSamples  = fm->cameraToWorldSamples();
  const int        numSamples = static_cast<int>(ssSamples.size());

  OgOAttribute<int> numSamplesAttr(mappingGroup, k_frustumMappingNumSamples,
                                   numSamples);

  // Then write each sample ---

  for (int i = 0; i < numSamples; ++i) {
    const string timeAttr = k_frustumMappingTime + 
      boost::lexical_cast<string>(i);
    const string ssAttr   = k_frustumMappingScreenMatrix + 
      boost::lexical_cast<string>(i);
    const string csAttr   = k_frustumMappingCameraMatrix + 
      boost::lexical_cast<string>(i);
    
    OgOAttribute<float> time(mappingGroup, timeAttr, ssSamples[i].first);
    OgOAttribute<mtx64_t> ss(mappingGroup, ssAttr, ssSamples[i].second);
    OgOAttribute<mtx64_t> cs(mappingGroup, csAttr, csSamples[i].second);
  }

  // Write distribution type ---

  int dist = static_cast<int>(fm->zDistribution());

  OgOAttribute<int> zDist(mappingGroup, k_frustumMappingZDistribution, dist);

  return true;
}

//----------------------------------------------------------------------------//

std::string FrustumFieldMappingIO::className() const
{ 
  return k_frustumMappingName;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
