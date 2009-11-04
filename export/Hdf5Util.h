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

/*! \file Hdf5Util.h
  \brief Contains various utility functions for Hdf5

  \todo Clean up the attribute read/write functions. Make them throw
  exceptions when failing.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_Hdf5Util_H_
#define _INCLUDED_Field3D_Hdf5Util_H_

//----------------------------------------------------------------------------//

#include <string>
#include <exception>
#include <vector>

#include <boost/lexical_cast.hpp>

#include <hdf5.h>

#include "Exception.h"
#include "Field.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Hdf5Util classes
//----------------------------------------------------------------------------//

//! Contains utility functions and classes for Hdf5 files
//! \ingroup hdf5
namespace Hdf5Util {

//----------------------------------------------------------------------------//

//! Base class for all scoped Hdf5 util classes
//! \ingroup hdf5
class H5Base
{
public:
  // Constructor
  H5Base()
    : m_id(-1)
  { /* Empty */ }
  //! Query the hid_t value
  hid_t id() const
  { return m_id; }
  //! Implicit cast to hid_t
  operator hid_t ()
  { return m_id; }
protected:
  hid_t m_id;
};

//----------------------------------------------------------------------------//

//! Scoped object - Opens attribute by name and closes it on destruction.
//! \ingroup hdf5
class H5ScopedAopen : public H5Base
{
public:
  H5ScopedAopen(hid_t location, const std::string &name)
  {
    m_id = H5Aopen(location, name.c_str(), H5P_DEFAULT);
    if (m_id < 0)
      throw Exc::MissingAttributeException("Couldn't open attribute " + name);
  }
  H5ScopedAopen(hid_t location, const std::string &name, hid_t aapl_id)
  {
    m_id = H5Aopen(location, name.c_str(), aapl_id);
    if (m_id < 0)
      throw Exc::MissingAttributeException("Couldn't open attribute " + name);
  }
  ~H5ScopedAopen()
  {
    if (m_id >= 0)
      H5Aclose(m_id);
  }
};


//----------------------------------------------------------------------------//

//! Scoped object - Opens attribute by index and closes it on destruction.
//! \ingroup hdf5
class H5ScopedAopenIdx : public H5Base
{
public:
  H5ScopedAopenIdx(hid_t location, unsigned idx)
  {
    m_id = H5Aopen_idx(location, idx);
    if (m_id < 0)
      throw Exc::MissingAttributeException("Couldn't open attribute at index: "+boost::lexical_cast<std::string>(idx));
  }
  ~H5ScopedAopenIdx()
  {
    if (m_id >= 0)
      H5Aclose(m_id);
  }
};

//----------------------------------------------------------------------------//

//! Scoped object - creates a group on creation and closes it on destruction.
//! \ingroup hdf5
class H5ScopedGcreate : public H5Base
{
public:
  H5ScopedGcreate(hid_t parentLocation, const std::string &name)
  {
    m_id = H5Gcreate(parentLocation, name.c_str(), 
                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  }
  H5ScopedGcreate(hid_t parentLocation, const std::string &name,              
                  hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id)
  {
    m_id = H5Gcreate(parentLocation, name.c_str(), 
                          lcpl_id, gcpl_id, gapl_id);
  }

  ~H5ScopedGcreate()
  {
    if (m_id >= 0)
      H5Gclose(m_id);
  }
};

//----------------------------------------------------------------------------//

//! Scoped object - opens a group on creation and closes it on destruction.
//! \ingroup hdf5
class H5ScopedGopen : public H5Base
{
public:
  H5ScopedGopen()
    : H5Base()
  {
    // Empty
  }
  H5ScopedGopen(hid_t parentLocation, const std::string &name)
  {
    open(parentLocation, name);
  }
  H5ScopedGopen(hid_t parentLocation, const std::string &name, hid_t gapl_id)
  {
    open(parentLocation, name, gapl_id);
  }
  void open(hid_t parentLocation, const std::string &name)
  {
    m_id = H5Gopen(parentLocation, name.c_str(), H5P_DEFAULT);
  }
  void open(hid_t parentLocation, const std::string &name, hid_t gapl_id)
  {
    m_id = H5Gopen(parentLocation, name.c_str(), gapl_id);
  }

  ~H5ScopedGopen()
  {
    if (m_id >= 0)
      H5Gclose(m_id);
  }
};

//----------------------------------------------------------------------------//

//! Scoped object - creates a dataspace on creation and closes it on 
//! destruction.
//! \ingroup hdf5
class H5ScopedScreate : public H5Base
{
public:
  H5ScopedScreate()
    : H5Base()
  {
    // Empty
  }
  H5ScopedScreate(H5S_class_t type)
  {
    create(type);
  }
  void create(H5S_class_t type)
  {
    m_id = H5Screate(type);
  }
  ~H5ScopedScreate()
  {
    if (m_id >= 0)
      H5Sclose(m_id);
  }
};

//----------------------------------------------------------------------------//

//! Scoped object - creates a dataset on creation and closes it on destruction.
//! \ingroup hdf5
class H5ScopedDcreate : public H5Base
{
public:
  H5ScopedDcreate(hid_t parentLocation, const std::string &name,
                  hid_t dtype_id, hid_t space_id, hid_t lcpl_id, 
                  hid_t dcpl_id, hid_t dapl_id)
  {
    m_id = H5Dcreate(parentLocation, name.c_str(), dtype_id, space_id,
                     lcpl_id, dcpl_id, dapl_id);
  }
  ~H5ScopedDcreate()
  {
    if (m_id >= 0)
      H5Dclose(m_id);
  }
};

//----------------------------------------------------------------------------//

//! Scoped object - opens an attribute data space on creation and closes it on 
//! destruction.
//! \ingroup hdf5
class H5ScopedAget_space : public H5Base
{
public:
  H5ScopedAget_space(hid_t dataset_id)
  {
    m_id = H5Aget_space(dataset_id);
    if (m_id < 0)
      throw Exc::AttrGetSpaceException("Couldn't get attribute space");
  }
  ~H5ScopedAget_space()
  {
    if (m_id >= 0)
      H5Sclose(m_id);
  }
};

//----------------------------------------------------------------------------//

//! Scoped object - opens an attribute data type on creation and closes it on 
//! destruction.
//! \ingroup hdf5
class H5ScopedAget_type : public H5Base
{
public:
  H5ScopedAget_type(hid_t dataset_id)
  {
    m_id = H5Aget_type(dataset_id);
    if (m_id < 0)
      throw Exc::AttrGetTypeException("Couldn't get attribute type");
  }
  ~H5ScopedAget_type()
  {
    if (m_id >= 0)
      H5Tclose(m_id);
  }
};

//----------------------------------------------------------------------------//

//! Scoped object - opens an native type id on creation and closes it on 
//! destruction.
//! \ingroup hdf5
class H5ScopedTget_native_type : public H5Base
{
public:
  H5ScopedTget_native_type(hid_t dataset_id, H5T_direction_t direction)
  {
    m_id = H5Tget_native_type(dataset_id, direction);
    if (m_id < 0)
      throw Exc::AttrGetNativeTypeException("Couldn't get native attribute type");
  }
  ~H5ScopedTget_native_type()
  {
    if (m_id >= 0)
      H5Tclose(m_id);
  }
};

//----------------------------------------------------------------------------//

//! Scoped object - opens a dataset on creation and closes it on 
//! destruction.
//! \ingroup hdf5
class H5ScopedDopen : public H5Base
{
public:
  H5ScopedDopen()
    : H5Base()
  {
    // Empty
  }
  H5ScopedDopen(hid_t parentLocation, const std::string &name, hid_t dapl_id)
  {
    open(parentLocation, name, dapl_id);
  }
  void open(hid_t parentLocation, const std::string &name, hid_t dapl_id)
  {
    m_id = H5Dopen(parentLocation, name.c_str(), dapl_id);
  }
  ~H5ScopedDopen()
  {
    if (m_id >= 0) {
      H5Dclose(m_id);
    }
  }
};

//----------------------------------------------------------------------------//

//! Scoped object - opens a dataset on creation and closes it on destruction.
//! \ingroup hdf5
class H5ScopedDget_space : public H5Base
{
public:
  H5ScopedDget_space()
    : H5Base()
  {
    // Empty
  }
  H5ScopedDget_space(hid_t dataset_id)
  {
    open(dataset_id);
  }
  void open(hid_t dataset_id)
  {
    m_id = H5Dget_space(dataset_id);
  }
  ~H5ScopedDget_space()
  {
    if (m_id >= 0)
      H5Sclose(m_id);
  }
};

//----------------------------------------------------------------------------//

//! Scoped object - opens a dataset on creation and closes it on 
//! destruction.
//! \ingroup hdf5
class H5ScopedDget_type : public H5Base
{
public:
  H5ScopedDget_type()
    : H5Base()
  {
    // Empty
  }
  H5ScopedDget_type(hid_t dataset_id)
  {
    open(dataset_id);
  }
  void open(hid_t dataset_id)
  {
    m_id = H5Dget_type(dataset_id);
  }
  ~H5ScopedDget_type()
  {
    if (m_id >= 0)
      H5Tclose(m_id);
  }
};

//----------------------------------------------------------------------------//
// Hdf5Util functions
//----------------------------------------------------------------------------//

/*! \name Read/write simple data to hdf5 location
  \{
*/

//! Writes a simple linear data set to the given location
//! \ingroup hdf5
template <typename T>
void writeSimpleData(hid_t location, const std::string &name,
                     const std::vector<T> &data);

//! Reads a simple linear data set from the given location
//! \ingroup hdf5
template <typename T>
void readSimpleData(hid_t location, const std::string &name,
                    std::vector<T> &data);

//! \}

//----------------------------------------------------------------------------//

/*! \name Attribute reading
  \{
*/

//! Reads a string attribute
//! \ingroup hdf5
bool readAttribute(hid_t location, const std::string& attrName, 
                  std::string& value); 

//! Reads an int attribute of arbitrary size
//! \ingroup hdf5
bool readAttribute(hid_t location, const std::string& attrName, 
                  unsigned int attrSize, int &value); 

//! Reads a float attribute of arbitrary size
//! \ingroup hdf5
bool readAttribute(hid_t location, const std::string& attrName, 
                  unsigned int attrSize, float &value); 

//! Reads a double attribute of arbitrary size
//! \ingroup hdf5
bool readAttribute(hid_t location, const std::string& attrName, 
                  unsigned int attrSize, double &value); 

//! \}

//----------------------------------------------------------------------------//

/*! \name Attribute writing
  \{
*/

//! Writes a string attribute
//! \ingroup hdf5
bool writeAttribute(hid_t location, const std::string& attrName, 
                  const std::string& value); 

//! Writes an int attribute of arbitrary size
//! \ingroup hdf5
bool writeAttribute(hid_t location, const std::string& attrName, 
                  unsigned int attrSize, const int &value); 

//! Writes a float attribute of arbitrary size
//! \ingroup hdf5
bool writeAttribute(hid_t location, const std::string& attrName, 
                  unsigned int attrSize, const float &value); 

//! Writes a double attribute of arbitrary size
//! \ingroup hdf5
bool writeAttribute(hid_t location, const std::string& attrName, 
                  unsigned int attrSize, const double &value); 

//! \}

//----------------------------------------------------------------------------//

//! Checks whether gzip is available in the current hdf5 library
//! \ingroup hdf5
bool checkHdf5Gzip();

//----------------------------------------------------------------------------//
// Templated functions and classes
//----------------------------------------------------------------------------//

//! Compile-time traits-class for conversion between c++ type and hdf5 type
//! \ingroup hdf5 template_util
template <class CppType_T>
struct TypeToH5Type
{
  static hid_t type();
};

//----------------------------------------------------------------------------//

//! Compile-time traits-class for getting a proper name from type
//! \ingroup hdf5 template_util
//! \todo This shouldn't really be in hdf5 utils?
template <class T>
struct NameForType
{
  static std::string name();
};

//----------------------------------------------------------------------------//

//! Compile-time traits-class for getting bits per component from type
//! \ingroup hdf5 template_util
//! \todo This shouldn't really be in hdf5 utils?
template <class T>
struct BitsForType
{
  static int bits();
};

//----------------------------------------------------------------------------//
// Specializations
//----------------------------------------------------------------------------//

template <>
inline hid_t TypeToH5Type<half>::type()
{
  return H5T_NATIVE_SHORT;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t TypeToH5Type<float>::type()
{
  return H5T_NATIVE_FLOAT;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t TypeToH5Type<double>::type()
{
  return H5T_NATIVE_DOUBLE;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t TypeToH5Type<char>::type()
{
  return H5T_NATIVE_CHAR;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t TypeToH5Type<V3h>::type()
{
  return H5T_NATIVE_SHORT;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t TypeToH5Type<V3f>::type()
{
  return H5T_NATIVE_FLOAT;
}

//----------------------------------------------------------------------------//

template <>
inline hid_t TypeToH5Type<V3d>::type()
{
  return H5T_NATIVE_DOUBLE;
}

//----------------------------------------------------------------------------//

template <>
inline std::string NameForType<half>::name() 
{ 
  return std::string("half"); 
}

//----------------------------------------------------------------------------//

template <>
inline std::string NameForType<float>::name() 
{ 
  return std::string("float"); 
}

//----------------------------------------------------------------------------//

template <>
inline std::string NameForType<double>::name() 
{ 
  return std::string("double"); 
}

//----------------------------------------------------------------------------//

template <>
inline std::string NameForType<V3h>::name() 
{ 
  return std::string("V3h"); 
}

//----------------------------------------------------------------------------//

template <>
inline std::string NameForType<V3f>::name() 
{ 
  return std::string("V3f"); 
}

//----------------------------------------------------------------------------//

template <>
inline std::string NameForType<V3d>::name() 
{ 
  return std::string("V3d"); 
}

//----------------------------------------------------------------------------//

template <>
inline int BitsForType<half>::bits() 
{ 
  return 16; 
}

//----------------------------------------------------------------------------//

template <>
inline int BitsForType<float>::bits() 
{ 
  return 32; 
}

//----------------------------------------------------------------------------//

template <>
inline int BitsForType<double>::bits() 
{ 
  return 64; 
}

//----------------------------------------------------------------------------//

template <>
inline int BitsForType<V3h>::bits() 
{ 
  return 16; 
}

//----------------------------------------------------------------------------//

template <>
inline int BitsForType<V3f>::bits() 
{ 
  return 32; 
}

//----------------------------------------------------------------------------//

template <>
inline int BitsForType<V3d>::bits() 
{ 
  return 64; 
}

//----------------------------------------------------------------------------//
// Implementations
//----------------------------------------------------------------------------//

template <class CppType_T>
hid_t TypeToH5Type<CppType_T>::type()
{
  assert(false && "Unsupported type in TypeToH5Type::type()");
  Msg::print(Msg::SevWarning, "Unsupported type in TypeToH5Type::type()");
  return -1;
}

//----------------------------------------------------------------------------//

template <class T>
std::string NameForType<T>::name()
{
  assert(false && "Unsupported type in NameForType::name()");
  Msg::print(Msg::SevWarning, "Unsupported type in NameForType::name()");
  return std::string("ERROR in NameForType::name()");
}

//----------------------------------------------------------------------------//

template <class T>
int BitsForType<T>::bits()
{
  assert(false && "Unsupported type in BitsForType::bits()");
  Msg::print(Msg::SevWarning, "Unsupported type in BitsForType::bits()");
  return -1;
}

//----------------------------------------------------------------------------//

template <typename T>
void writeSimpleData(hid_t location, const std::string &name,
                     const std::vector<T> &data)
{
  using namespace Exc;

  // Calculate the total number of entries. This factors in that
  // V3f uses 3 components per value, etc.
  hsize_t totalSize[1];
  int components = FieldTraits<T>::dataDims();
  totalSize[0] = data.size() * components;

  // Get the internal data type
  hid_t type = TypeToH5Type<T>::type();

  H5ScopedScreate dataSpace(H5S_SIMPLE);

  if (dataSpace.id() < 0)
    throw WriteSimpleDataException("Couldn't create data space");
  
  H5Sset_extent_simple(dataSpace.id(), 1, totalSize, NULL);

  H5ScopedDcreate dataSet(location, name.c_str(), type, dataSpace.id(), 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  if (dataSet.id() < 0) 
    throw WriteSimpleDataException("Couldn't create data set");
    
  hid_t err = H5Dwrite(dataSet.id(), type, H5S_ALL, H5S_ALL, 
                       H5P_DEFAULT, &data[0]);

  if (err < 0) 
    throw WriteSimpleDataException("Couldn't write data");
}

//----------------------------------------------------------------------------//

template <typename T>
void readSimpleData(hid_t location, const std::string &name,
                     std::vector<T> &data)
{
  using namespace Exc;

  int components = FieldTraits<T>::dataDims();
  hsize_t dims[1];

  H5ScopedDopen dataSet(location, name.c_str(), H5P_DEFAULT);

  if (dataSet.id() < 0) 
    throw OpenDataSetException("Couldn't open data set: " + name);
  
  H5ScopedDget_space dataSpace(dataSet.id());
  H5ScopedDget_type dataType(dataSet.id());
  H5Sget_simple_extent_dims(dataSpace.id(), dims, NULL);

  if (dataSpace.id() < 0) 
    throw GetDataSpaceException("Couldn't get data space");

  if (dataType.id() < 0)
    throw GetDataTypeException("Couldn't get data type");

  int reportedSize = dims[0] / components;

  // Resize target
  data.clear();
  data.resize(reportedSize);
  
  // Get the internal data type
  hid_t type = TypeToH5Type<T>::type();

  if (H5Dread(dataSet.id(), type, H5S_ALL, H5S_ALL, 
              H5P_DEFAULT, &data[0]) < 0) {
    throw Hdf5DataReadException("Couldn't read simple data");
  }
}

//----------------------------------------------------------------------------//

} // namespace Hdf5Util

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
