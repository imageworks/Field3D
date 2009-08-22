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

/*! \file FieldMappingFactory.cpp
  \ingroup field
  \brief Contains implementation of FieldMappingFactory.
*/

//----------------------------------------------------------------------------//

#include <iostream>
#include <vector>

#include "FieldMappingFactory.h"
#include "Hdf5Util.h"

//----------------------------------------------------------------------------//

using namespace boost;
using namespace std;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Field3D namespaces
//----------------------------------------------------------------------------//

using namespace Exc;
using namespace Hdf5Util;

//----------------------------------------------------------------------------//
// Local namespace
//----------------------------------------------------------------------------//

namespace {
  //! \todo This is duplicated in FieldMapping.cpp. Fix.
  const string k_nullMappingName("NullFieldMapping");
  //! \todo This is duplicated in FieldMapping.cpp. Fix.
  const string k_matrixMappingName("MatrixFieldMapping");
  //! \todo This is duplicated in FieldMapping.cpp. Fix.
  const string k_frustumMappingName("FrustumFieldMapping");

  const string k_mappingTypeAttrName("mapping_type");
  const string k_nullMappingDataName("NullFieldMapping data");
  const string k_matrixMappingDataName("MatrixFieldMapping data");
}

//----------------------------------------------------------------------------//
// FieldMappingFactory static data members
//----------------------------------------------------------------------------//

FieldMappingFactory *FieldMappingFactory::ms_theFieldMappingFactory = NULL;

//----------------------------------------------------------------------------//
// FieldMappingFactory implementations
//----------------------------------------------------------------------------//

bool FieldMappingFactory::write(hid_t mappingGroup, FieldMapping::Ptr mapping)
{
  std::string mappingType = mapping->typeName();

  if (!writeAttribute(mappingGroup, k_mappingTypeAttrName, mappingType)) {
    Log::print(Log::SevWarning, "Couldn't add " + mappingType + " attribute");
    return false;
  }

  // Add in a test for each subclass here

  NullFieldMapping::Ptr nm = 
    dynamic_pointer_cast<NullFieldMapping>(mapping);
  MatrixFieldMapping::Ptr mm = 
    dynamic_pointer_cast<MatrixFieldMapping>(mapping);

  if (nm)
    return writeNullMapping(mappingGroup, nm);
  else if (mm)
    return writeMatrixMapping(mappingGroup, mm);

  Log::print(Log::SevWarning, 
            "Unknown mapping type in FieldMappingFactory::write: " + 
            mappingType);

  return false;
}

//----------------------------------------------------------------------------//

FieldMapping::Ptr FieldMappingFactory::read(hid_t mappingGroup)
{
  std::string mappingType;

  if (!readAttribute(mappingGroup, k_mappingTypeAttrName, mappingType)) {
    Log::print(Log::SevWarning, "Couldn't find " + k_mappingTypeAttrName + 
              " attribute");
    return FieldMapping::Ptr();    
  }

  if (mappingType == k_nullMappingName) 
    return readNullMapping(mappingGroup);
  
  if (mappingType == k_matrixMappingName) 
    return readMatrixMapping(mappingGroup);

  Log::print(Log::SevWarning, "No registered function for reading " + mappingType);

  return FieldMapping::Ptr();  
}

//----------------------------------------------------------------------------//

FieldMappingFactory& 
FieldMappingFactory::theFieldMappingFactoryInstance()
{
  if (!ms_theFieldMappingFactory) {
    ms_theFieldMappingFactory = new FieldMappingFactory;
  }

  return *ms_theFieldMappingFactory;
}

//----------------------------------------------------------------------------//

bool FieldMappingFactory::writeNullMapping(hid_t mappingGroup, 
                                           NullFieldMapping::Ptr nm)
{
  string nfmAttrData("NullFieldMapping has no data");
  if (!writeAttribute(mappingGroup, k_nullMappingDataName, nfmAttrData)) {
    Log::print(Log::SevWarning, "Couldn't add attribute " + k_nullMappingDataName);
    return false;
  }
  return true;
}

//----------------------------------------------------------------------------//

NullFieldMapping::Ptr 
FieldMappingFactory::readNullMapping(hid_t mappingGroup)
{
  string nfmData;
  if (!readAttribute(mappingGroup, k_nullMappingDataName, nfmData)) {
    Log::print(Log::SevWarning, "Couldn't read attribute " + k_nullMappingDataName);
    return NullFieldMapping::Ptr();
  }
  return NullFieldMapping::Ptr(new NullFieldMapping);
}

//----------------------------------------------------------------------------//

bool FieldMappingFactory::writeMatrixMapping(hid_t mappingGroup, 
                                             MatrixFieldMapping::Ptr mm)
{
  if (!writeAttribute(mappingGroup, k_matrixMappingDataName, 16, 
                    mm->localToWorld().x[0][0])) {
    Log::print(Log::SevWarning, "Couldn't add attribute " + k_matrixMappingDataName);
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------//

MatrixFieldMapping::Ptr 
FieldMappingFactory::readMatrixMapping(hid_t mappingGroup)
{
  M44d mtx;

  if (!readAttribute(mappingGroup, k_matrixMappingDataName, 16,
                     mtx.x[0][0])) {
    Log::print(Log::SevWarning, "Couldn't read attribute " + k_matrixMappingDataName);
    return MatrixFieldMapping::Ptr();
  }

  MatrixFieldMapping::Ptr mm(new MatrixFieldMapping);

  mm->setLocalToWorld(mtx);

  return mm;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
