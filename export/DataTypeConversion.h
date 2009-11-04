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

/*! \file DataTypeConversion.h
  This file contains the dataTypeToString() function. It is used to provide
  compile-time strings corresponding to the typename.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_DataTypeConversion_H_
#define _INCLUDED_Field3D_DataTypeConversion_H_

#include <string>

#include "Types.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Functions
//----------------------------------------------------------------------------//

//! Converts a c++ data type to string
//! \ingroup template_util
template <typename T>
inline std::string dataTypeToString() 
{ return "default"; }

//----------------------------------------------------------------------------//
// Template specializations
//----------------------------------------------------------------------------//

template<>
inline std::string dataTypeToString<half>()
{
  return "half";
}

//----------------------------------------------------------------------------//

template<>
inline std::string dataTypeToString<float>()
{
  return "float";
}

//----------------------------------------------------------------------------//

template<>
inline std::string dataTypeToString<double>()
{
  return "double";
}

//----------------------------------------------------------------------------//

template<>
inline std::string dataTypeToString<V3h>()
{
  return "V3h";
}

//----------------------------------------------------------------------------//

template<>
inline std::string dataTypeToString<V3f>()
{
  return "V3f";
}

//----------------------------------------------------------------------------//

template<>
inline std::string dataTypeToString<V3d>()
{
  return "V3d";
}

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

//! \todo Remove this implementation so compiler will catch non-specialized
//! cases.
template <typename T>
inline DataTypeEnum dataTypeToEnum() { return DataTypeUnknown; }

//----------------------------------------------------------------------------//

// template specializations

template<>
inline DataTypeEnum dataTypeToEnum<SPI::OpenEXR::half>()
{
  return DataTypeHalf;
}

template<>
inline DataTypeEnum dataTypeToEnum<float>()
{
  return DataTypeFloat;
}

template<>
inline DataTypeEnum dataTypeToEnum<double>()
{
  return DataTypeDouble;
}

template<>
inline DataTypeEnum
dataTypeToEnum<SPI::OpenEXR::Imath::Vec3<SPI::OpenEXR::half> >()
{
  return DataTypeVecHalf;
}

template<>
inline DataTypeEnum dataTypeToEnum<SPI::OpenEXR::Imath::V3f>()
{
  return DataTypeVecFloat;
}

template<>
inline DataTypeEnum dataTypeToEnum<SPI::OpenEXR::Imath::V3d>()
{
  return DataTypeVecDouble;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
