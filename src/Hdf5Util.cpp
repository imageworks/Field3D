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

/*! \file Hdf5Util.cpp
  \todo The write attrib calls need some serious cleanup. They should
  be using scoped objects to open attributes and groups instead of
  all the replicated close calls.
*/

//----------------------------------------------------------------------------//

#include "Hdf5Util.h"

#include <iostream>
#include <vector>

//----------------------------------------------------------------------------//

using namespace std;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

using namespace Exc;

namespace Hdf5Util {

//----------------------------------------------------------------------------//
// Implementations
//----------------------------------------------------------------------------//

bool 
readAttribute(hid_t location, const string& attrName, string& value)
{
  H5T_class_t typeClass;
  H5A_info_t attrInfo;
  hsize_t strLen;

  if (H5Aexists(location, attrName.c_str()) < 1)
    throw MissingAttributeException("Couldn't find attribute " + attrName);

  H5ScopedAopen attr(location, attrName.c_str(), H5P_DEFAULT);
  H5ScopedAget_space attrSpace(attr);
  H5ScopedAget_type attrType(attr);

  if (H5Aget_info(attr, &attrInfo) < 0) {
    throw MissingAttributeException("Couldn't get attribute info " + attrName);
  } else {
    strLen = attrInfo.data_size;
  }

  typeClass = H5Tget_class(attrType);

  if (typeClass != H5T_STRING)
    throw MissingAttributeException("Bad attribute type class for " + attrName);

  H5ScopedTget_native_type nativeType(attrType, H5T_DIR_ASCEND);

  std::vector<char> tempString(strLen + 1);

  if (H5Aread(attr, nativeType, &tempString[0]) < 0) 
    throw MissingAttributeException("Couldn't read attribute " + attrName);

  value = string(&tempString[0]);

  return true;

}

//----------------------------------------------------------------------------//

bool 
readAttribute(hid_t location, const string& attrName, 
              unsigned int attrSize, int &value)
{
  H5T_class_t typeClass;

  if (H5Aexists(location, attrName.c_str()) < 1)
    throw MissingAttributeException("Couldn't find attribute " + attrName);

  H5ScopedAopen attr(location, attrName.c_str(), H5P_DEFAULT);
  H5ScopedAget_space attrSpace(attr);
  H5ScopedAget_type attrType(attr);

  if (H5Sget_simple_extent_ndims(attrSpace) != 1) 
    throw MissingAttributeException("Bad attribute rank for attribute " + 
                                    attrName);

  hsize_t dims[1];
  H5Sget_simple_extent_dims(attrSpace, dims, NULL);

  if (dims[0] != attrSize) 
    throw MissingAttributeException("Invalid attribute size for attribute " + 
                                    attrName);

  typeClass = H5Tget_class(attrType);

  if (typeClass != H5T_INTEGER) 
    throw MissingAttributeException("Bad attribute type class for " + 
                                    attrName);

  H5ScopedTget_native_type nativeType(attrType, H5T_DIR_ASCEND);

  if (H5Aread(attr, nativeType, &value) < 0) 
    throw MissingAttributeException("Couldn't read attribute " + attrName);

  return true;

}

//----------------------------------------------------------------------------//

bool 
readAttribute(hid_t location, const string& attrName, 
             unsigned int attrSize, float &value)
{
  H5T_class_t typeClass;

  if (H5Aexists(location, attrName.c_str()) < 1)
    throw MissingAttributeException("Couldn't find attribute " + attrName);

  H5ScopedAopen attr(location, attrName.c_str(), H5P_DEFAULT);
  H5ScopedAget_space attrSpace(attr);
  H5ScopedAget_type attrType(attr);

  if (H5Sget_simple_extent_ndims(attrSpace) != 1) 
    throw MissingAttributeException("Bad attribute rank for attribute " + 
                                    attrName);

  hsize_t dims[1];
  H5Sget_simple_extent_dims(attrSpace, dims, NULL);

  if (dims[0] != attrSize) 
    throw MissingAttributeException("Invalid attribute size for attribute " + 
                                    attrName);

  typeClass = H5Tget_class(attrType);

  if (typeClass != H5T_FLOAT)
    throw MissingAttributeException("Bad attribute type class for " + 
                                    attrName);

  H5ScopedTget_native_type nativeType(attrType, H5T_DIR_ASCEND);

  if (H5Aread(attr, nativeType, &value) < 0) 
    throw MissingAttributeException("Couldn't read attribute " + attrName);

  return true;
}

//----------------------------------------------------------------------------//

bool 
readAttribute(hid_t location, const string& attrName, 
             unsigned int attrSize, double &value)
{
  H5T_class_t typeClass;

  if (H5Aexists(location, attrName.c_str()) < 0)
    throw MissingAttributeException("Couldn't find attribute " + attrName);

  H5ScopedAopen attr(location, attrName.c_str(), H5P_DEFAULT);
  H5ScopedAget_space attrSpace(attr);
  H5ScopedAget_type attrType(attr);

  if (H5Sget_simple_extent_ndims(attrSpace) != 1) 
    throw MissingAttributeException("Bad attribute rank for attribute " + 
                                    attrName);

  hsize_t dims[1];
  H5Sget_simple_extent_dims(attrSpace, dims, NULL);

  if (dims[0] != attrSize) 
    throw MissingAttributeException("Invalid attribute size for attribute " + 
                                    attrName);

  typeClass = H5Tget_class(attrType);

  if (typeClass != H5T_FLOAT)
    throw MissingAttributeException("Bad attribute type class for " + 
                                    attrName);

  H5ScopedTget_native_type nativeType(attrType, H5T_DIR_ASCEND);

  if (H5Aread(attr, nativeType, &value) < 0) 
    throw MissingAttributeException("Couldn't read attribute " + attrName);

  return true;
}

//----------------------------------------------------------------------------//

bool 
writeAttribute(hid_t location, const string& attrName, const string &value)
{
  hid_t attr = -1;
  hid_t attrSpace;
  hid_t attrType;

  bool success = true;

  attrSpace = H5Screate(H5S_SCALAR);
  if (attrSpace == -1)
    success = false;

  attrType = H5Tcopy(H5T_C_S1);
  if (attrType == -1)
    success = false;

  if (value.size()) {
    // if the string is null the following will return error
    // which we don't want.
    if (success && H5Tset_size(attrType, value.size()) == -1){    
      success = false;
    }
  }

  if (success) {
    H5Tset_strpad(attrType, H5T_STR_NULLTERM);
    attr = H5Acreate(location, attrName.c_str(), attrType, attrSpace, 
                     H5P_DEFAULT, H5P_DEFAULT);
  }

  if (attr == -1) {
    Msg::print(Msg::SevWarning, "Error creating attribute: " + attrName);
    success = false;
  }

  if (success && H5Awrite(attr, attrType, value.c_str()) == -1) {
    Msg::print(Msg::SevWarning, "Error writing attribute: " + attrName);
    success = false;
  }

  H5Aclose(attr);
  H5Tclose(attrType);
  H5Sclose(attrSpace);

  return success;

}

//----------------------------------------------------------------------------//

bool 
writeAttribute(hid_t location, const string &attrName, 
             unsigned int attrSize, const int &value)
{
  hid_t attr;
  hid_t attrSpace;
  hsize_t dims[1];

  dims[0] = attrSize;

  attrSpace = H5Screate(H5S_SIMPLE);
  if (attrSpace < 0) 
    return false;

  if (H5Sset_extent_simple(attrSpace, 1, dims, NULL) < 0)
    return false;

  attr = H5Acreate(location, attrName.c_str(), H5T_NATIVE_INT, 
                   attrSpace, H5P_DEFAULT, H5P_DEFAULT);
  if (attr < 0) {
    Msg::print(Msg::SevWarning, "Error creating attribute: " + attrName);
    H5Aclose(attr);
    H5Sclose(attrSpace);
    return false;
  }

  if (H5Awrite(attr, H5T_NATIVE_INT, &value) < 0) {
    Msg::print(Msg::SevWarning, "Error writing attribute: " + attrName);
    H5Aclose(attr);
    H5Sclose(attrSpace);
    return false;
  }

  H5Aclose(attr);
  H5Sclose(attrSpace);

  return true;
}

//----------------------------------------------------------------------------//

bool 
writeAttribute(hid_t location, const string& attrName, 
             unsigned int attrSize, const float &value)
{
  hid_t attr;
  hid_t attrSpace;
  hsize_t dims[1];

  dims[0] = attrSize;

  attrSpace = H5Screate(H5S_SIMPLE);
  if (attrSpace < 0) 
    return false;

  if (H5Sset_extent_simple(attrSpace, 1, dims, NULL) < 0)
    return false;

  attr = H5Acreate(location, attrName.c_str(), H5T_NATIVE_FLOAT, 
                   attrSpace, H5P_DEFAULT, H5P_DEFAULT);
  if (attr < 0) {
    Msg::print(Msg::SevWarning, "Error creating attribute: " + attrName);
    H5Aclose(attr);
    H5Sclose(attrSpace);
    return false;
  }

  if (H5Awrite(attr, H5T_NATIVE_FLOAT, &value) < 0) {
    Msg::print(Msg::SevWarning, "Error writing attribute: " + attrName);
    H5Aclose(attr);
    H5Sclose(attrSpace);
    return false;
  }

  H5Aclose(attr);
  H5Sclose(attrSpace);

  return true;
}

//----------------------------------------------------------------------------//

bool 
writeAttribute(hid_t location, const string& attrName, 
             unsigned int attrSize, const double &value)
{
  hid_t attr;
  hid_t attrSpace;
  hsize_t dims[1];

  dims[0] = attrSize;

  attrSpace = H5Screate(H5S_SIMPLE);
  if (attrSpace < 0) 
    return false;

  if (H5Sset_extent_simple(attrSpace, 1, dims, NULL) < 0)
    return false;

  attr = H5Acreate(location, attrName.c_str(), H5T_NATIVE_DOUBLE, 
                   attrSpace, H5P_DEFAULT, H5P_DEFAULT);
  if (attr < 0) {
    Msg::print(Msg::SevWarning, "Error creating attribute: " + attrName);
    H5Aclose(attr);
    H5Sclose(attrSpace);
    return false;
  }

  if (H5Awrite(attr, H5T_NATIVE_DOUBLE, &value) < 0) {
    Msg::print(Msg::SevWarning, "Error writing attribute: " + attrName);
    H5Aclose(attr);
    H5Sclose(attrSpace);
    return false;
  }

  H5Aclose(attr);
  H5Sclose(attrSpace);

  return true;
}

//----------------------------------------------------------------------------//

bool checkHdf5Gzip()
{
  htri_t avail = H5Zfilter_avail(H5Z_FILTER_DEFLATE);
  if (!avail)
    return false;

  unsigned int filter_info;
  herr_t status = H5Zget_filter_info (H5Z_FILTER_DEFLATE, &filter_info);

  if (status < 0)
    return false;

  if (!(filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED) ||
      !(filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED)) {
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------//

} // namespace Hdf5Util

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
