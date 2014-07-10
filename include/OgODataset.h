//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_OgODataset_H_
#define _INCLUDED_Field3D_OgODataset_H_

//----------------------------------------------------------------------------//
// Includes
//----------------------------------------------------------------------------//

#include "OgUtil.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// OgODataset
//----------------------------------------------------------------------------//

/*! \class OgOAttribute
  Ogawa output attribute. Writes a variable as an attribute into an F3D-style
  Ogawa file.
*/

//----------------------------------------------------------------------------//

template <typename T>
class OgODataset
{

public:

  // Ctors, dtor ---------------------------------------------------------------

  //! Creates the data set, but does not write any data.
  OgODataset(OgOGroup &parent, const std::string &name)
  {
    // Create a group to store the basic data
    m_group = parent.addSubGroup();
    // Index 0 is the name
    writeString(m_group, name);
    // Index 1 is the type
    writeData(m_group, F3DDatasetType);
    // Index 2 is the data type
    writeDataType<T>(m_group);
    // Index 3 onwards are the actual dataset entries
  }
  
  // Main methods --------------------------------------------------------------

  //! Adds a data element to the data set. Each element may be of different
  //! length
  void addData(const size_t length, const T *data)
  {
    m_group->addData(length * sizeof(T), data);
  }

private:

  //! Pointer to the enclosing group
  Alembic::Ogawa::OGroupPtr m_group;

};

//----------------------------------------------------------------------------//
  
FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // include guard

//----------------------------------------------------------------------------//
