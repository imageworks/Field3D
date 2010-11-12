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

/*! \file Traits.h
  This file contains the DataTypeTraits struct, with class functions
for converting templatization into strings and enums.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_Traits_H_
#define _INCLUDED_Field3D_Traits_H_

#include <string>

#include <assert.h>

#include <hdf5.h>

#include "Log.h"
#include "Types.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Enums
//----------------------------------------------------------------------------//

enum DataTypeEnum {
  DataTypeHalf=0,
  DataTypeFloat,
  DataTypeDouble,
  DataTypeVecHalf,
  DataTypeVecFloat,
  DataTypeVecDouble,
  DataTypeUnknown
};

//----------------------------------------------------------------------------//
// Data Structures
//----------------------------------------------------------------------------//

template <typename T>
struct DataTypeTraits {
  static std::string name() {
    assert(false && "Unsupported type in DataTypeTraits::name()");
    Msg::print(Msg::SevWarning, "Unsupported type in DataTypeTraits::name()");
    return std::string("ERROR in DataTypeTraits::name()");
  }
  static DataTypeEnum typeEnum() {
    assert(false && "Unsupported type in DataTypeTraits::typeEnum()");
    Msg::print(Msg::SevWarning,
               "Unsupported type in DataTypeTraits::typeEnum()");
    return DataTypeUnknown;
  }
  static hid_t h5type() {
    assert(false && "Unsupported type in DataTypeTraits::h5type()");
    Msg::print(Msg::SevWarning,
               "Unsupported type in DataTypeTraits::h5type()");
    return 0;
  }
  static int h5bits() {
    assert(false && "Unsupported type in DataTypeTraits::h5bits()");
    Msg::print(Msg::SevWarning,
               "Unsupported type in DataTypeTraits::h5bits()");
    return 0;
  }
};

//----------------------------------------------------------------------------//
// Template specializations
//----------------------------------------------------------------------------//

template<>
inline std::string DataTypeTraits<half>::name()
{
  return std::string("half");
}

//----------------------------------------------------------------------------//

template<>
inline std::string DataTypeTraits<float>::name()
{
  return std::string("float");
}

//----------------------------------------------------------------------------//

template<>
inline std::string DataTypeTraits<double>::name()
{
  return std::string("double");
}

//----------------------------------------------------------------------------//

template<>
inline std::string DataTypeTraits<V3h>::name()
{
  return std::string("V3h");
}

//----------------------------------------------------------------------------//

template<>
inline std::string DataTypeTraits<V3f>::name()
{
  return std::string("V3f");
}

//----------------------------------------------------------------------------//

template<>
inline std::string DataTypeTraits<V3d>::name()
{
  return std::string("V3d");
}

//----------------------------------------------------------------------------//

template<>
inline DataTypeEnum DataTypeTraits<half>::typeEnum()
{
  return DataTypeHalf;
}

//----------------------------------------------------------------------------//

template<>
inline DataTypeEnum DataTypeTraits<float>::typeEnum()
{
  return DataTypeFloat;
}

//----------------------------------------------------------------------------//

template<>
inline DataTypeEnum DataTypeTraits<double>::typeEnum()
{
  return DataTypeDouble;
}

//----------------------------------------------------------------------------//

template<>
inline DataTypeEnum DataTypeTraits<V3h>::typeEnum()
{
  return DataTypeVecHalf;
}

//----------------------------------------------------------------------------//

template<>
inline DataTypeEnum DataTypeTraits<V3f>::typeEnum()
{
  return DataTypeVecFloat;
}

//----------------------------------------------------------------------------//

template<>
inline DataTypeEnum DataTypeTraits<V3d>::typeEnum()
{
  return DataTypeVecDouble;
}

template <>
inline hid_t DataTypeTraits<half>::h5type()
{
  return H5T_NATIVE_SHORT;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t DataTypeTraits<float>::h5type()
{
  return H5T_NATIVE_FLOAT;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t DataTypeTraits<double>::h5type()
{
  return H5T_NATIVE_DOUBLE;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t DataTypeTraits<char>::h5type()
{
  return H5T_NATIVE_CHAR;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t DataTypeTraits<V3h>::h5type()
{
  return H5T_NATIVE_SHORT;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t DataTypeTraits<V3f>::h5type()
{
  return H5T_NATIVE_FLOAT;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t DataTypeTraits<V3d>::h5type()
{
  return H5T_NATIVE_DOUBLE;
}

//----------------------------------------------------------------------------//

template <>
inline int DataTypeTraits<half>::h5bits() 
{ 
  return 16; 
}

//----------------------------------------------------------------------------//

template <>
inline int DataTypeTraits<float>::h5bits() 
{ 
  return 32; 
}

//----------------------------------------------------------------------------//

template <>
inline int DataTypeTraits<double>::h5bits() 
{ 
  return 64; 
}

//----------------------------------------------------------------------------//

template <>
inline int DataTypeTraits<V3h>::h5bits() 
{ 
  return 16; 
}

//----------------------------------------------------------------------------//

template <>
inline int DataTypeTraits<V3f>::h5bits() 
{ 
  return 32; 
}

//----------------------------------------------------------------------------//

template <>
inline int DataTypeTraits<V3d>::h5bits() 
{ 
  return 64; 
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
