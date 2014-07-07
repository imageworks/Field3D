//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_OgOAttribute_H_
#define _INCLUDED_Field3D_OgOAttribute_H_

//----------------------------------------------------------------------------//
// Includes
//----------------------------------------------------------------------------//

#include "OgUtil.h"
#include "OgOGroup.h"

//----------------------------------------------------------------------------//
// OgOAttribute
//----------------------------------------------------------------------------//

/*! \class OgOAttribute
  Ogawa output attribute. Writes a variable as an attribute into an F3D-style
  Ogawa file.
*/

//----------------------------------------------------------------------------//

template <typename T>
class OgOAttribute
{

public:

  // Ctors, dtor ---------------------------------------------------------------

  //! Creates the attribute and writes the data.
  OgOAttribute(OgOGroup &parent, const std::string &name, const T &value)
  {
    // Create a group to store the attribute data
    Alembic::Ogawa::OGroupPtr group = parent.addSubGroup();
    // Index 0 is the name
    writeString(group, name);
    // Index 1 is the type
    writeData(group, F3DAttributeType);
    // Index 2 is the data type
    writeDataType<T>(group);
    // Index 3 is the data
    writeData<T>(group, value);
  }
};

//----------------------------------------------------------------------------//

#endif // include guard

//----------------------------------------------------------------------------//
