//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_OgOGroup_H_
#define _INCLUDED_Field3D_OgOGroup_H_

//----------------------------------------------------------------------------//
// Includes
//----------------------------------------------------------------------------//

#include "OgUtil.h"

//----------------------------------------------------------------------------//
// OgOGroup
//----------------------------------------------------------------------------//

/*! \class OgOAttribute
  Ogawa output attribute. Writes a variable as an attribute into an F3D-style
  Ogawa file.
*/

//----------------------------------------------------------------------------//

class OgOGroup : public Alembic::Util::noncopyable
{

public:

  // Ctor, dtor ----------------------------------------------------------------

  //! Constructs the root group in an Ogawa archive
  OgOGroup(Alembic::Ogawa::OArchive &archive)
    : m_name("f3droot") 
  {
    // Reference the root group
    m_group = archive.getGroup();
    // Base data sets
    addBaseData();
  }

  //! Constructs a group as a child to an existing group.
  OgOGroup(OgOGroup &parent, const std::string &name)
    : m_name(name)
  {
    // Construct the group
    m_group = parent.m_group->addGroup();
    // Base data sets
    addBaseData();
  }

  // Main methods --------------------------------------------------------------

  //! Adds an ogawa-level subgroup. Called from OgOAttribute and OgODataset.
  Alembic::Ogawa::OGroupPtr addSubGroup() 
  {
    return m_group->addGroup();
  }

private:

  // Utility methods -----------------------------------------------------------

  void addBaseData()
  {
    // Index 0 is the name
    writeString(m_group, m_name);
    // Index 1 is the type
    writeData(m_group, F3DGroupType);
  }

  // Data members --------------------------------------------------------------

  //! Pointer to the enclosing Ogawa-level group.
  Alembic::Ogawa::OGroupPtr m_group;
  //! Name of the curren
  std::string               m_name;
};

//----------------------------------------------------------------------------//

#endif // include guard

//----------------------------------------------------------------------------//
