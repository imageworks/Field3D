//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_OgUtil_H_
#define _INCLUDED_Field3D_OgUtil_H_

//----------------------------------------------------------------------------//
// Includes
//----------------------------------------------------------------------------//

#include <iostream>
#include <string>

#include <OpenEXR/ImathVec.h>

#include "All.h"
#include "UtilFoundation.h"

#include "Traits.h"

//----------------------------------------------------------------------------//
// Defines
//----------------------------------------------------------------------------//

#define OGAWA_THREAD                0
#define OGAWA_START_ID              2
#define OGAWA_DATASET_BASEOFFSET    3
#define OGAWA_INVALID_DATASET_INDEX -1

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Enums
//----------------------------------------------------------------------------//

//! Enumerates the various uses for Ogawa-level groups. 
//! \warning Do not under any circumstances alter the order of these!
enum OgGroupType {
  F3DGroupType = 0,
  F3DAttributeType,
  F3DDatasetType,
  F3DCompressedDatasetType
};

//----------------------------------------------------------------------------//
// OgawaTypeTraits
//----------------------------------------------------------------------------//

//! Declares the OgawaTypeTraits struct, but does not implement it.
template <typename T>
struct OgawaTypeTraits;

//----------------------------------------------------------------------------//

#define F3D_DECLARE_OG_TRAITS(type, enum, name, def) \
  template <>                                        \
  struct OgawaTypeTraits<type>                       \
  {                                                  \
  typedef type value_type;                           \
  static const char* typeName() { return name; }     \
  static OgDataType  typeEnum() { return enum; }     \
  static value_type  defaultValue() { return def; }  \
  };

//----------------------------------------------------------------------------//

F3D_DECLARE_OG_TRAITS(int8_t,      F3DInt8,    "int8_t", 0);
F3D_DECLARE_OG_TRAITS(uint8_t,     F3DUint8,   "uint8_t", 0);
F3D_DECLARE_OG_TRAITS(int16_t,     F3DInt16,   "int16_t", 0);
F3D_DECLARE_OG_TRAITS(uint16_t,    F3DUint16,  "uint16_t", 0);
F3D_DECLARE_OG_TRAITS(int32_t,     F3DInt32,   "int32_t", 0);
F3D_DECLARE_OG_TRAITS(uint32_t,    F3DUint32,  "uint32_t", 0);
F3D_DECLARE_OG_TRAITS(int64_t,     F3DInt64,   "int64_t", 0);
F3D_DECLARE_OG_TRAITS(uint64_t,    F3DUint64,  "uint64_t", 0);
F3D_DECLARE_OG_TRAITS(float16_t,   F3DFloat16, "float16_t", 0);
F3D_DECLARE_OG_TRAITS(float32_t,   F3DFloat32, "float32_t", 0);
F3D_DECLARE_OG_TRAITS(float64_t,   F3DFloat64, "float64_t", 0);
F3D_DECLARE_OG_TRAITS(vec16_t,     F3DVec16,   "vec16_t", vec16_t(0));
F3D_DECLARE_OG_TRAITS(vec32_t,     F3DVec32,   "vec32_t", vec32_t(0));
F3D_DECLARE_OG_TRAITS(vec64_t,     F3DVec64,   "vec64_t", vec64_t(0));
F3D_DECLARE_OG_TRAITS(veci32_t,    F3DVecI32,  "veci32_t", veci32_t(0));
F3D_DECLARE_OG_TRAITS(mtx64_t,     F3DMtx64,   "mtx64_t", mtx64_t());
F3D_DECLARE_OG_TRAITS(std::string, F3DString,  "string", "");

//----------------------------------------------------------------------------//
// Helper functions
//----------------------------------------------------------------------------//

//! Gets a string representation of the OgGroupType enum
const char* ogGroupTypeToString(OgGroupType type);

//----------------------------------------------------------------------------//

//! Reads a string
bool readString(Alembic::Ogawa::IGroupPtr group, const size_t idx, 
                std::string &s);

//----------------------------------------------------------------------------//

//! Reads a single value
template <typename T>
bool readData(Alembic::Ogawa::IGroupPtr group, const size_t idx, 
              T &value) 
{
  // Grab data
  Alembic::Ogawa::IDataPtr data = group->getData(idx, OGAWA_THREAD);
  // Check data length
  const size_t sizeLength = sizeof(T);
  const size_t length = data->getSize();
  if (length != sizeLength) {
    return false;
  }
  // Read the data directly to the input param
  data->read(length, &value, 0, OGAWA_THREAD);
  // Done
  return true;
}

//----------------------------------------------------------------------------//

//! Specialization of readData for strings
template <>
inline bool readData(Alembic::Ogawa::IGroupPtr group, const size_t idx, 
                     std::string &value) 
{
  return readString(group, idx, value);
}

//----------------------------------------------------------------------------//

//! Reads a single data type value
//! \note Not returning a bool here, since OgDataType has an 'invalid' enum.
OgDataType readDataType(Alembic::Ogawa::IGroupPtr group, const size_t idx);

//----------------------------------------------------------------------------//

//! Lowest-level, writes directly to Ogawa Data.
bool writeString(Alembic::Ogawa::OGroupPtr group, const std::string &s);

//----------------------------------------------------------------------------//

//! Lowest-level, writes directly to Ogawa Data.
template <typename T>
bool writeData(Alembic::Ogawa::OGroupPtr group, const T &value)
{
  return group->addData(sizeof(T), &value) != NULL;
}

//----------------------------------------------------------------------------//

//! Specialization of writeData for strings
template <>
inline bool writeData(Alembic::Ogawa::OGroupPtr group, const std::string &value)
{
  return writeString(group, value);
}

//----------------------------------------------------------------------------//

//! Lowest-level, writes directly to Ogawa Data.
template <typename T>
bool writeDataType(Alembic::Ogawa::OGroupPtr group)
{
  OgDataType dataType = OgawaTypeTraits<T>::typeEnum();
  return group->addData(sizeof(OgDataType), &dataType) != NULL;
}

//----------------------------------------------------------------------------//

//! Finds the group name. Really just a wrapper around readString() for the
//! first data set in the group.
bool getGroupName(Alembic::Ogawa::IGroupPtr group, std::string &name);

//----------------------------------------------------------------------------//
// OgIBase
//----------------------------------------------------------------------------//

//! Provides a few standard member functions and data members
class OgIBase
{
public:

  // Ctors, dtor ---------------------------------------------------------------

  //! No initialization - leaves the group pointer null
  OgIBase()
  { /* Empty */ }
  //! Initializes the group pointer
  OgIBase(Alembic::Ogawa::IGroupPtr group)
    : m_group(group)
  { /* Empty */ }

  // Main methods --------------------------------------------------------------

  //! Whether the group is valid
  bool               isValid() const
  { return m_group != NULL; }
  //! Returns the name
  const std::string& name() const
  { return m_name; }

protected:

  // Data members --------------------------------------------------------------

  Alembic::Ogawa::IGroupPtr m_group;
  std::string               m_name;

};

//----------------------------------------------------------------------------//
  
FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // include guard

//----------------------------------------------------------------------------//
