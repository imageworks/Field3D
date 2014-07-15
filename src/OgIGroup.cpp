//----------------------------------------------------------------------------//
// Includes
//----------------------------------------------------------------------------//

#include "OgIGroup.h"

//----------------------------------------------------------------------------//

using std::cout;
using std::endl;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// OgIGroup implementations
//----------------------------------------------------------------------------//

OgIGroup::OgIGroup(Alembic::Ogawa::IArchive &archive)
  : OgIBase(archive.getGroup())
{
  validate();
  if (m_group) {
    getGroupName(m_group, m_name);
  }
}

//----------------------------------------------------------------------------//

OgGroupType OgIGroup::type() const
{
  return F3DGroupType;
}

//----------------------------------------------------------------------------//

std::vector<std::string> OgIGroup::groupNames() const
{
  return groupNames(F3DGroupType);
}

//----------------------------------------------------------------------------//

std::vector<std::string> OgIGroup::attributeNames() const
{
  return groupNames(F3DAttributeType);
}

//----------------------------------------------------------------------------//

std::vector<std::string> OgIGroup::datasetNames() const
{
  return groupNames(F3DDatasetType);
}

//----------------------------------------------------------------------------//

std::vector<std::string> OgIGroup::compressedDatasetNames() const
{
  return groupNames(F3DCompressedDatasetType);
}

//----------------------------------------------------------------------------//

OgIGroup OgIGroup::findGroup(const std::string &name) const
{
  Alembic::Ogawa::IGroupPtr group = findGroup(name, F3DGroupType);

  if (group) {
    return OgIGroup(group);
  }

  return OgIGroup();
}

//----------------------------------------------------------------------------//

Alembic::Ogawa::IGroupPtr 
OgIGroup::findGroup(const std::string &path,
                    const OgGroupType groupType) const
{
  // If not valid, return non-valid group
  if (!isValid()) {
    return Alembic::Ogawa::IGroupPtr();
  }
  // Check for recursive finding
  if (path.find("/") != std::string::npos) {
    return recursiveFindGroup(path, groupType);
  }
  // Not recursive, just a single group name
  const std::string &name = path;
  // If it's valid, we know we have at least 2 children.
  // Check all children
  for (size_t i = OGAWA_START_ID, end = m_group->getNumChildren(); 
       i < end; ++i) {
    // Is it an Ogawa group? If not, continue
    if (!m_group->isChildGroup(i)) {
      continue;
    }
    // Grab the ogawa group
    Alembic::Ogawa::IGroupPtr group = 
      m_group->getGroup(i, false, OGAWA_THREAD);
    // Data set 0 is the name
    std::string groupName;
    if (!readString(group, 0, groupName)) {
      // This is a bad error. Print details.
      std::cout << "OgIGroup::findGroup() couldn't read subgroup name for "
                << "group name: " << name << std::endl;
      return Alembic::Ogawa::IGroupPtr();
    }
    // Data set 1 is the type
    OgGroupType type;
    if (!readData(group, 1, type)) {
      // This is a bad error. Print details.
      std::cout << "OgIGroup::findGroup() couldn't read subgroup type for "
                << "group name: " << name << std::endl;
      return Alembic::Ogawa::IGroupPtr();
    }
    // Check that group type matches
    if (type != groupType) {
      // This is not an error.
      continue;
    }
    // Check if name matches
    if (groupName == name) {
      return group;
    }
  }
  // Didn't find one
  return Alembic::Ogawa::IGroupPtr();
}

//----------------------------------------------------------------------------//

Alembic::Ogawa::IGroupPtr 
OgIGroup::recursiveFindGroup(const std::string &path,
                             const OgGroupType groupType) const
{
  // If not valid, return non-valid group
  if (!isValid()) {
    return Alembic::Ogawa::IGroupPtr();
  }
  // Find the next group name in the path
  const size_t pos = path.find("/");
  const std::string name = path.substr(0, pos);
  const std::string restOfPath = path.substr(pos + 1);
  // If the group is valid, we know we have at least 2 children.
  // Check all children
  for (size_t i = OGAWA_START_ID, end = m_group->getNumChildren(); 
       i < end; ++i) {
    // Is it an Ogawa group? If not, continue
    if (!m_group->isChildGroup(i)) {
      continue;
    }
    // Grab the ogawa group
    Alembic::Ogawa::IGroupPtr group = 
      m_group->getGroup(i, false, OGAWA_THREAD);
    // Data set 0 is the name
    std::string groupName;
    if (!readString(group, 0, groupName)) {
      // This is a bad error. Print details.
      std::cout << "OgIGroup::recursiveFindGroup() couldn't read subgroup "
                << "name for group name: " << name << std::endl;
      return Alembic::Ogawa::IGroupPtr();
    }
    // Data set 1 is the type
    OgGroupType type;
    if (!readData(group, 1, type)) {
      // This is a bad error. Print details.
      std::cout << "OgIGroup::recursiveFindGroup() couldn't read subgroup "
                << "type for group name: " << name << std::endl;
      return Alembic::Ogawa::IGroupPtr();
    }
    // Check that group type is F3DGroupType
    if (type != F3DGroupType) {
      // This is not an error.
      continue;
    }
    // Check if name matches
    if (groupName == name) {
      OgIGroup subGroup(group);
      return subGroup.findGroup(restOfPath, groupType);
    }
  }
  // Didn't find one
  cout << "Couldn't find group: " << name << endl;
  return Alembic::Ogawa::IGroupPtr();
}

//----------------------------------------------------------------------------//

std::vector<std::string> 
OgIGroup::groupNames(const OgGroupType groupType) const
{
  // If not valid, return non-valid group
  if (!isValid()) {
    return std::vector<std::string>();
  }
  // Check all children
  std::vector<std::string> groups;
  for (size_t i = OGAWA_START_ID, end = m_group->getNumChildren(); 
       i < end; ++i) {
    // Is it an Ogawa group? If not, continue
    if (!m_group->isChildGroup(i)) {
      continue;
    }
    // Grab the ogawa group
    Alembic::Ogawa::IGroupPtr group = 
      m_group->getGroup(i, false, OGAWA_THREAD);
    // Data set 0 is the name
    std::string groupName;
    if (!readString(group, 0, groupName)) {
      continue;
    }
    // Data set 1 is the type
    OgGroupType type;
    if (!readData(group, 1, type)) {
      continue;
    }
    // Check that group type matches
    if (type != groupType) {
      continue;
    }
    // Add group name
    groups.push_back(groupName);
  }
  // Done
  return groups;
}

//----------------------------------------------------------------------------//

OgDataType OgIGroup::attributeType(const std::string &name) const
{
  Alembic::Ogawa::IGroupPtr group = findGroup(name, F3DAttributeType);

  if (group && group->getNumChildren() > 2) {
    return readDataType(group, 2);
  }

  return F3DInvalidDataType;
}

//----------------------------------------------------------------------------//

OgDataType OgIGroup::datasetType(const std::string &name) const
{
  Alembic::Ogawa::IGroupPtr group = findGroup(name, F3DDatasetType);

  if (group && group->getNumChildren() > 2) {
    return readDataType(group, 2);
  }

  return F3DInvalidDataType;
}

//----------------------------------------------------------------------------//

OgDataType OgIGroup::compressedDatasetType(const std::string &name) const
{
  Alembic::Ogawa::IGroupPtr group = findGroup(name, F3DCompressedDatasetType);

  if (group && group->getNumChildren() > 2) {
    return readDataType(group, 2);
  }

  return F3DInvalidDataType;
}

//----------------------------------------------------------------------------//

OgIGroup::OgIGroup()
{
  // Nothing
}

//----------------------------------------------------------------------------//

OgIGroup::OgIGroup(Alembic::Ogawa::IGroupPtr group)
  : OgIBase(group)
{
  validate();
  getGroupName(m_group, m_name);
}

//----------------------------------------------------------------------------//

void OgIGroup::validate()
{
  // If we don't have two children, we're invalid.
  if (m_group && m_group->getNumChildren() < 2) {
    m_group.reset();
    return;
  }
  // If the two first children aren't data sets, we're invalid
  if (m_group && (!m_group->isChildData(0) || !m_group->isChildData(1))) {
    m_group.reset();
  }
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
