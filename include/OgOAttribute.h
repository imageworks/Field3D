//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_OgOAttribute_H_
#define _INCLUDED_Field3D_OgOAttribute_H_

//----------------------------------------------------------------------------//
// Includes
//----------------------------------------------------------------------------//

#include "OgUtil.h"
#include "OgOGroup.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

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

  // Typedefs ------------------------------------------------------------------

  typedef T value_type;

  // Ctors, dtor ---------------------------------------------------------------

  //! Creates the attribute and writes the data.
  OgOAttribute(OgOGroup &parent, const std::string &name, const T &value)
  {
    using Field3D::Exc::OgOAttributeException;

    // Create a group to store the attribute data
    Alembic::Ogawa::OGroupPtr group = parent.addSubGroup();
    // Index 0 is the name
    if (!writeString(group, name)) {
      throw OgOAttributeException("Couldn't write attribute name for " + name);
    }
    // Index 1 is the type
    if (!writeData(group, F3DAttributeType)) {
      throw OgOAttributeException("Couldn't write attribute group type for " + 
                                  name);
    }
    // Index 2 is the data type
    if (!writeDataType<T>(group)) {
      throw OgOAttributeException("Couldn't write attribute data type for " + 
                                  name);
    }
    // Index 3 is the data
    if (!writeData<T>(group, value)) {
      throw OgOAttributeException("Couldn't write attribute data for " + name);
    }
  }
};

//----------------------------------------------------------------------------//
  
FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // include guard

//----------------------------------------------------------------------------//
