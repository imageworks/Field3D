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

/*! \file FieldMappingIO.cpp
  \brief Contains the FieldMapping base class and the NullFieldMapping and
  MatrixFieldMapping subclass implementations.
*/

//----------------------------------------------------------------------------//

#include "Hdf5Util.h"

#include "FieldMappingIO.h"

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

  const string k_nullMappingDataName("NullFieldMapping data");
  const string k_matrixMappingDataName("MatrixFieldMapping data");
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

//! Returns the class name
std::string NullFieldMappingIO::className() const
{ return k_nullMappingName; }

//----------------------------------------------------------------------------//

FieldMapping::Ptr
MatrixFieldMappingIO::read(hid_t mappingGroup)
{
  M44d mtx;

  if (!readAttribute(mappingGroup, k_matrixMappingDataName, 16,
                     mtx.x[0][0])) {
    Msg::print(Msg::SevWarning, "Couldn't read attribute " + k_matrixMappingDataName);
    return MatrixFieldMapping::Ptr();
  }

  MatrixFieldMapping::Ptr mm(new MatrixFieldMapping);

  mm->setLocalToWorld(mtx);

  return mm;
}

//----------------------------------------------------------------------------//

bool
MatrixFieldMappingIO::write(hid_t mappingGroup, FieldMapping::Ptr mapping)
{
  MatrixFieldMapping::Ptr mm =
    boost::dynamic_pointer_cast<MatrixFieldMapping>(mapping);
  if (!mm) {
    Msg::print(Msg::SevWarning, "Couldn't get MatrixFieldMapping from pointer");
    return false;
  }
  if (!writeAttribute(mappingGroup, k_matrixMappingDataName, 16, 
                    mm->localToWorld().x[0][0])) {
    Msg::print(Msg::SevWarning, "Couldn't add attribute " + k_matrixMappingDataName);
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------//

//! Returns the class name
std::string MatrixFieldMappingIO::className() const
{ return k_matrixMappingName; }

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
