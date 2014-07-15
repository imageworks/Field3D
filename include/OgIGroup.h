//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_OgIGroup_H_
#define _INCLUDED_Field3D_OgIGroup_H_

//----------------------------------------------------------------------------//
// Includes
//----------------------------------------------------------------------------//

#include "OgUtil.h"
#include "OgIAttribute.h"
#include "OgIDataset.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// OgIGroup
//----------------------------------------------------------------------------//

/*! \class OgIGroup
  Ogawa input group. Opens a group in an F3D-style Ogawa file.

  Groups can contain other groups, as well as attributes and data sets.
*/

//----------------------------------------------------------------------------//

class OgIGroup : public OgIBase
{

public:

  // Ctor, dtor ----------------------------------------------------------------

  //! Construct from archive
  OgIGroup(Alembic::Ogawa::IArchive &archive);

  // Main methods --------------------------------------------------------------

  //! Returns the type
  OgGroupType              type() const;

  //! Returns a list of F3D group names (always unique)
  std::vector<std::string> groupNames() const;

  //! Returns a list of F3D attribute names (always unique)
  std::vector<std::string> attributeNames() const;

  //! Returns a list of F3D dataset names (always unique)
  std::vector<std::string> datasetNames() const;

  //! Returns a list of compressed F3D dataset names (always unique)
  std::vector<std::string> compressedDatasetNames() const;

  //! Finds an F3D group. The returned OgIGroup will not be valid if the name
  //! wasn't found. The name may be a full path including '/'-separators,
  //! in which case the nested group will be searched for.
  OgIGroup                 findGroup(const std::string &path) const;

  //! Finds an F3D group. The returned OgIAttribute will not be valid if the 
  //! name wasn't found.
  template <typename T>
  OgIAttribute<T>          findAttribute(const std::string &name) const;

  //! Finds an F3D dataset. The returned OgIGroup will not be valid if the name
  //! wasn't found.
  template <typename T>
  OgIDataset<T>            findDataset(const std::string &name) const;

  //! Finds a compressed F3D dataset. The returned OgIGroup will not be valid 
  //! if the name wasn't found.
  template <typename T>
  OgICDataset<T>           findCompressedDataset(const std::string &name) const;

  //! Returns the data type of an attribute
  OgDataType               attributeType(const std::string &name) const;

  //! Returns the data type of a dataset
  OgDataType               datasetType(const std::string &name) const;

  //! Returns the data type of a compressed dataset
  OgDataType               compressedDatasetType(const std::string &name) const;

private:
  
  // Private ctors -------------------------------------------------------------

  //! Construct invalid group
  OgIGroup();

  //! Construct by opening group
  OgIGroup(Alembic::Ogawa::IGroupPtr group);

  // Utility methods -----------------------------------------------------------

  //! Finds an ogawa-level group
  Alembic::Ogawa::IGroupPtr findGroup(const std::string &name,
                                      const OgGroupType groupType) const;

  //! Recursively finds an ogawa-level group
  Alembic::Ogawa::IGroupPtr recursiveFindGroup
  (const std::string &name, const OgGroupType groupType) const;

  //! Retrieves list of ogawa-level groups of the given type
  std::vector<std::string>  groupNames(const OgGroupType groupType) const;

  //! Validates the current group
  void                      validate();

};

//----------------------------------------------------------------------------//
// Template implementations
//----------------------------------------------------------------------------//

template <typename T>
OgIAttribute<T> OgIGroup::findAttribute(const std::string &name) const
{
  Alembic::Ogawa::IGroupPtr group = findGroup(name, F3DAttributeType);

  if (group) {
    return OgIAttribute<T>(group);
  }

  return OgIAttribute<T>();
}

//----------------------------------------------------------------------------//

template <typename T>
OgIDataset<T> OgIGroup::findDataset(const std::string &name) const
{
  Alembic::Ogawa::IGroupPtr group = findGroup(name, F3DDatasetType);

  if (group) {
    return OgIDataset<T>(group);
  }

  return OgIDataset<T>();
}

//----------------------------------------------------------------------------//

template <typename T>
OgICDataset<T> OgIGroup::findCompressedDataset(const std::string &name) const
{
  Alembic::Ogawa::IGroupPtr group = findGroup(name, F3DCompressedDatasetType);

  if (group) {
    return OgICDataset<T>(group);
  }

  return OgICDataset<T>();
}

//----------------------------------------------------------------------------//
  
FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // include guard

//----------------------------------------------------------------------------//
