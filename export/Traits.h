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

#include <assert.h>
#include <string>

#include <hdf5.h>

#include "Log.h"
#include "Types.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Types
//----------------------------------------------------------------------------//

#if !defined(_MSC_VER)
using ::uint8_t;
using ::int8_t;
using ::uint16_t;
using ::int16_t;
using ::uint32_t;
using ::int32_t;
using ::uint64_t;
using ::int64_t;
#else
typedef unsigned char           uint8_t;
typedef signed char             int8_t;
typedef unsigned short          uint16_t;
typedef signed short            int16_t;
typedef unsigned int            uint32_t;
typedef int                     int32_t;
typedef unsigned long long      uint64_t;
typedef long long               int64_t;
#endif

typedef half                    float16_t;
typedef float                   float32_t;
typedef double                  float64_t;

#ifdef FIELD3D_VERSION_NS
typedef Field3D::V3h            vec16_t;
typedef Field3D::V3f            vec32_t;
typedef Field3D::V3d            vec64_t;
typedef Field3D::V3i            veci32_t;
typedef Field3D::M44d           mtx64_t;
#else
typedef Imath::Vec3<float16_t>  vec16_t;
typedef Imath::Vec3<float32_t>  vec32_t;
typedef Imath::Vec3<float64_t>  vec64_t;
typedef Imath::Vec3<int32_t>    veci32_t;
typedef Imath::M44d             mtx64_t;
#endif

//----------------------------------------------------------------------------//
// Enums
//----------------------------------------------------------------------------//

enum DataTypeEnum {
  DataTypeHalf=0,
  DataTypeUnsignedChar,
  DataTypeInt,
  DataTypeFloat,
  DataTypeDouble,
  DataTypeVecHalf,
  DataTypeVecFloat,
  DataTypeVecDouble,
  DataTypeUnknown
};

//----------------------------------------------------------------------------//

//! Enumerates the various uses for Ogawa-level groups. 
//! \warning Do not under any circumstances alter the order of these! If you
//! need to add more types, append them at the end, before F3DNumDataTypes.
enum OgDataType {

  // Signed and unsigned integers from char to long
  F3DInt8 = 0,
  F3DUint8,

  F3DInt16,
  F3DUint16,

  F3DInt32,
  F3DUint32,

  F3DInt64,
  F3DUint64,

  // Floats
  F3DFloat16,
  F3DFloat32,
  F3DFloat64,

  // Vec3
  F3DVec16,
  F3DVec32,
  F3DVec64,
  F3DVecI32,

  // Matrix
  F3DMtx64,

  // String
  F3DString, 

  F3DNumDataTypes, 

  // Invalid type enum
  F3DInvalidDataType = 127
};

//----------------------------------------------------------------------------//
// FieldTraits
//----------------------------------------------------------------------------//

/*! \class FieldTraits
  \ingroup template_util
  Traits class that lets us answer how many components are in a given data type
*/

template <class Data_T>
class FieldTraits
{
public:
  //! Dimensions of the given data type. i.e. 3 for V3f, 1 for float
  static const int k_dataDims = 3;
  static int dataDims() { return k_dataDims; }
};

template <>
struct FieldTraits<half>
{
public:
  static const int k_dataDims = 1;
  static int dataDims() { return k_dataDims; }
};

template <>
struct FieldTraits<float>
{
public:
  static const int k_dataDims = 1;
  static int dataDims() { return k_dataDims; }
};

template <>
struct FieldTraits<double>
{
public:
  static const int k_dataDims = 1;
  static int dataDims() { return k_dataDims; }
};

template <>
struct FieldTraits<int>
{
public:
  static const int k_dataDims = 1;
  static int dataDims() { return k_dataDims; }
};

template <>
struct FieldTraits<char>
{
public:
  static const int k_dataDims = 1;
  static int dataDims() { return k_dataDims; }
};

template <>
struct FieldTraits<unsigned char>
{
public:
  static const int k_dataDims = 1;
  static int dataDims() { return k_dataDims; }
};

template <>
struct FieldTraits<V3h>
{
public:
  static const int k_dataDims = 3;
  static int dataDims() { return k_dataDims; }
};

template <>
struct FieldTraits<V3f>
{
public:
  static const int k_dataDims = 3;
  static int dataDims() { return k_dataDims; }
};

template <>
struct FieldTraits<V3d>
{
public:
  static const int k_dataDims = 3;
  static int dataDims() { return k_dataDims; }
};

template <>
struct FieldTraits<C3f>
{
public:
  static const int k_dataDims = 3;
  static int dataDims() { return k_dataDims; }
};

//----------------------------------------------------------------------------//
// DataTypeTraits
//----------------------------------------------------------------------------//

template <typename T>
struct DataTypeTraits {
  static std::string name()
  {
    return typeid(T).name();
  }
  static DataTypeEnum typeEnum();
  static hid_t h5type();
  static int h5bits();
};

//----------------------------------------------------------------------------//
// TemplatedFieldType
//----------------------------------------------------------------------------//

//! Used to return a string for the name of a templated field
template <class Field_T>
struct TemplatedFieldType
{
  const char *name()
  {
    return m_name.c_str();
  }
  TemplatedFieldType()
  {
    m_name = Field_T::staticClassName();
    m_name +=
      "<" +
      DataTypeTraits<typename Field_T::value_type>::name() +
      ">";
  }
private:
  std::string m_name;
};

//----------------------------------------------------------------------------//
// NestedFieldType
//----------------------------------------------------------------------------//

//! Used to return a string for the name of a nested templated field
template <typename Field_T>
struct NestedFieldType
{
  const char *name()
  {
    return m_name.c_str();
  }
  NestedFieldType()
  {
    typedef typename Field_T::NestedType    NestedType;
    typedef typename NestedType::value_type value_type;

    m_name = Field_T::staticClassName();
    m_name += 
      std::string("<") + NestedType::staticClassName() + "<" +
      DataTypeTraits<value_type>::name() + ">>";
  }
private:
  std::string m_name;
};

//----------------------------------------------------------------------------//
// Template specializations
//----------------------------------------------------------------------------//

#define FIELD3D_DECL_DATATYPENAME(typeName)               \
  template<>                                              \
  inline std::string DataTypeTraits<typeName>::name()     \
  {                                                       \
    return std::string(#typeName);                        \
  }                                                       \

//----------------------------------------------------------------------------//

FIELD3D_DECL_DATATYPENAME(unsigned char)
FIELD3D_DECL_DATATYPENAME(int)
FIELD3D_DECL_DATATYPENAME(float)
FIELD3D_DECL_DATATYPENAME(half)
FIELD3D_DECL_DATATYPENAME(double)
FIELD3D_DECL_DATATYPENAME(V3h)
FIELD3D_DECL_DATATYPENAME(V3f)
FIELD3D_DECL_DATATYPENAME(V3d)

//----------------------------------------------------------------------------//

template<>
inline DataTypeEnum DataTypeTraits<half>::typeEnum()
{
  return DataTypeHalf;
}

//----------------------------------------------------------------------------//

template<>
inline DataTypeEnum DataTypeTraits<unsigned char>::typeEnum()
{
  return DataTypeUnsignedChar;
}

//----------------------------------------------------------------------------//

template<>
inline DataTypeEnum DataTypeTraits<int>::typeEnum()
{
  return DataTypeInt;
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
inline hid_t DataTypeTraits<unsigned char>::h5type()
{
  return H5T_NATIVE_UCHAR;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t DataTypeTraits<int>::h5type()
{
  return H5T_NATIVE_INT;
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
