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

/*! \file MACFieldIO.cpp
  \brief Contains implementations for the MACFieldIO class.
*/

//----------------------------------------------------------------------------//

#include "MACFieldIO.h"

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
// Static members
//----------------------------------------------------------------------------//

const int         MACFieldIO::k_versionNumber(1);
const std::string MACFieldIO::k_versionAttrName("version");
const std::string MACFieldIO::k_extentsStr("extents");
const std::string MACFieldIO::k_dataWindowStr("data_window");
const std::string MACFieldIO::k_componentsStr("components");
const std::string MACFieldIO::k_bitsPerComponentStr("bits_per_component");
const std::string MACFieldIO::k_uDataStr("u_data");
const std::string MACFieldIO::k_vDataStr("v_data");
const std::string MACFieldIO::k_wDataStr("w_data");

//----------------------------------------------------------------------------//

FieldBase::Ptr
MACFieldIO::read(hid_t layerGroup, const std::string &filename, 
                 const std::string &layerPath)
{
  Box3i extents, dataW;
  int components;

  //hsize_t dims[1];
  
  if (layerGroup == -1)
    throw BadHdf5IdException("Bad layer group in MACFieldIO::read");

  int version;
  if (!readAttribute(layerGroup, k_versionAttrName, 1, version))
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_versionAttrName);

  if (version != k_versionNumber)
    throw UnsupportedVersionException("MACField version not supported: " + 
                                      lexical_cast<std::string>(version));

  if (!readAttribute(layerGroup, k_extentsStr, 6, extents.min.x)) 
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_extentsStr);

  if (!readAttribute(layerGroup, k_dataWindowStr, 6, dataW.min.x)) 
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_dataWindowStr);
  
  if (!readAttribute(layerGroup, k_componentsStr, 1, components)) 
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_componentsStr);
  // Check the data type ---
  int bits;
  if (!readAttribute(layerGroup, k_bitsPerComponentStr, 1, bits)) 
    throw MissingAttributeException("Couldn't find attribute: " +
                                    k_bitsPerComponentStr);  

  // Build a MACField to store everything in
  FieldBase::Ptr result;
  switch (bits) {
  case 16:
    {
      MACField<V3h>::Ptr field(new MACField<V3h>);
      field->setSize(extents, dataW);
      readData<V3h>(layerGroup, field);

      result = field;      
    }
    break;
  case 64:
    {
      MACField<V3d>::Ptr field(new MACField<V3d>);
      field->setSize(extents, dataW);
      readData<V3d>(layerGroup, field);

      result = field;      
    }
    break;
  case 32:
  default:
    {
      MACField<V3f>::Ptr field(new MACField<V3f>);
      field->setSize(extents, dataW);
      readData<V3f>(layerGroup, field);

      result = field;      
    }
  }

  return result;
}

//----------------------------------------------------------------------------//

bool
MACFieldIO::write(hid_t layerGroup, FieldBase::Ptr field)
{
  if (layerGroup == -1) {
    throw BadHdf5IdException("Bad layer group in MACFieldIO::write");
  }

  // Add version attribute
  if (!writeAttribute(layerGroup, k_versionAttrName, 
                      1, k_versionNumber)) {
    throw WriteAttributeException("Couldn't write attribute " + 
                                  k_versionAttrName);
  }

  MACField<V3h>::Ptr vecHalfField = 
    dynamic_pointer_cast<MACField<V3h> >(field);
  MACField<V3f>::Ptr vecFloatField = 
    dynamic_pointer_cast<MACField<V3f> >(field);
  MACField<V3d>::Ptr vecDoubleField = 
    dynamic_pointer_cast<MACField<V3d> >(field);

  bool success = true;
  if (vecFloatField) {
    success = writeInternal<V3f>(layerGroup, vecFloatField);
  } else if (vecHalfField) {
    success = writeInternal<V3h>(layerGroup, vecHalfField);
  } else if (vecDoubleField) {
    success = writeInternal<V3d>(layerGroup, vecDoubleField);
  } else {
    throw WriteLayerException("MACFieldIO does not support the given "
                              "MACField template parameter");
  }

  return success;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
