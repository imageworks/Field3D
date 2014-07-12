//----------------------------------------------------------------------------//
// Includes
//----------------------------------------------------------------------------//

#include "OgUtil.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Helper functions
//----------------------------------------------------------------------------//

const char* ogGroupTypeToString(OgGroupType type)
{
  switch(type)
  {
  case F3DGroupType:
    return "group";
  case F3DAttributeType:
    return "attribute";
  case F3DDatasetType:
    return "dataset";
  default:
    return "";
  }
}

//----------------------------------------------------------------------------//

bool readString(Alembic::Ogawa::IGroupPtr group, const size_t idx, 
                std::string &s)
{
  // Grab data
  Alembic::Ogawa::IDataPtr data = group->getData(idx, OGAWA_THREAD);
  // Check that we got something
  if (!data) {
    std::cout << "OgUtil::readString() got null data for index " 
              << idx << std::endl;
    std::cout << "  numChildren(): " << group->getNumChildren() << std::endl;
    return false;
  }
  // Check data length
  const size_t length = data->getSize();
  if (length % sizeof(std::string::value_type) != 0) {
    return false;
  }
  // String length
  const size_t stringLength = length / sizeof(std::string::value_type);
  // Read into temp buffer. Reading straight into std::string is Bad.
  std::vector<std::string::value_type> temp(stringLength + 1);
  // Add null terminator
  temp[stringLength] = 0;
  // Read the data
  data->read(length, &temp[0], 0, OGAWA_THREAD);
  // Construct string. The string see the temp buffer as a const char *.
  s = std::string(&temp[0]);
  // Done
  return true;
}

//----------------------------------------------------------------------------//

OgDataType readDataType(Alembic::Ogawa::IGroupPtr group, const size_t idx) 
{
  // Data type
  OgDataType dataType;;
  // Grab data
  Alembic::Ogawa::IDataPtr data = group->getData(idx, OGAWA_THREAD);
  // Check data length
  const size_t sizeLength = sizeof(OgDataType);
  const size_t length = data->getSize();
  if (length != sizeLength) {
    std::cout << "readDataType() " << sizeLength << " != " << length << std::endl;
    return F3DInvalidDataType;
  }
  // Read the data directly to the input param
  data->read(length, &dataType, 0, OGAWA_THREAD);
  // Done
  return dataType;
}

//----------------------------------------------------------------------------//

bool writeString(Alembic::Ogawa::OGroupPtr group, const std::string &s)
{
  // Strings are written without zero terminator
  Alembic::Ogawa::ODataPtr data = 
    group->addData(s.size() * sizeof(std::string::value_type), s.c_str());
  return data != NULL;
}

//----------------------------------------------------------------------------//

bool getGroupName(Alembic::Ogawa::IGroupPtr group, 
                  std::string &name)
{
  return readString(group, 0, name);
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//

