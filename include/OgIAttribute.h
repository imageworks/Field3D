//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_OgIAttribute_H_
#define _INCLUDED_Field3D_OgIAttribute_H_

//----------------------------------------------------------------------------//
// Includes
//----------------------------------------------------------------------------//

#include "OgUtil.h"

#include <OpenEXR/ImathMatrix.h>

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// OgIAttribute
//----------------------------------------------------------------------------//

/*! \class OgIAttribute
  Ogawa input attribute. Reads a single attribute from an F3D-style 
  Ogawa file.

  Attributes should not be relied on for performance input of large data sets,
  use OgIDataset for that.
*/

//----------------------------------------------------------------------------//

template <typename T>
class OgIAttribute : public OgIBase
{

public:

  // Typedefs ------------------------------------------------------------------

  typedef T value_type;

  // Ctor, dtor ----------------------------------------------------------------

  //! The default constructor leaves m_group initialized, which implies it 
  //! is in an invalid state.
  OgIAttribute();

  //! Initialize from an existing IGroup. The constructor will check that the
  //! type enum matches the template parameter, otherwise the attribute is
  //! marked invalid.
  OgIAttribute(Alembic::Ogawa::IGroupPtr group);

  // Main methods --------------------------------------------------------------

  //! Returns the value of the attribute. This will be zero if the attribute
  //! is invalid.
  T value() const;

};

//----------------------------------------------------------------------------//
// Template implementations
//----------------------------------------------------------------------------//

template <typename T>
OgIAttribute<T>::OgIAttribute()
{ 
  // Empty
}

//----------------------------------------------------------------------------//

template <typename T>
OgIAttribute<T>::OgIAttribute(Alembic::Ogawa::IGroupPtr group)
  : OgIBase(group)
{
  // Handle null pointer
  if (!OgIBase::m_group) {
    return;
  }
  // Check data type
  OgDataType dataType = readDataType(group, 2);
  if (dataType != OgawaTypeTraits<T>::typeEnum()) {
    OgIBase::m_group.reset();
    return;
  }
  // Update name
  getGroupName(OgIBase::m_group, OgIBase::m_name);
}

//----------------------------------------------------------------------------//

template <typename T>
T OgIAttribute<T>::value() const
{
  T v;
  if (readData(m_group, 3, v)) {
    return v;
  }
  return OgawaTypeTraits<T>::defaultValue();
}

//----------------------------------------------------------------------------//
  
FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // include guard

//----------------------------------------------------------------------------//
