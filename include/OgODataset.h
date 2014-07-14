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

  // Typedefs ------------------------------------------------------------------

  typedef T value_type;

  // Ctors, dtor ---------------------------------------------------------------

  //! Creates the data set, but does not write any data.
  OgODataset(OgOGroup &parent, const std::string &name);

  // Main methods --------------------------------------------------------------

  //! Adds a data element to the data set. Each element may be of different
  //! length
  void addData(const size_t length, const T *data);

private:

  //! Pointer to the enclosing group
  Alembic::Ogawa::OGroupPtr m_group;

};

//----------------------------------------------------------------------------//
// OgOCDataset
//----------------------------------------------------------------------------//

/*! \class OgOCDataset
  Ogawa output compressed dataset. 
  \note This class is templated to preserve data type, but calls to add
  actual data elements to the data sets deal with uint8_t, since compressed
  data lengths are not necessarily an even multiplier of sizeof(T)
*/

//----------------------------------------------------------------------------//

template <typename T>
class OgOCDataset
{

public:

  // Typedefs ------------------------------------------------------------------

  typedef T value_type;

  // Ctors, dtor ---------------------------------------------------------------

  //! Creates the data set, but does not write any data.
  OgOCDataset(OgOGroup &parent, const std::string &name);

  // Main methods --------------------------------------------------------------

  //! Adds a data element to the data set. Each element may be of different
  //! length
  //! \note The length is measured in bytes (sizeof(uint8_t))
  void addData(const size_t byteLength, const uint8_t *data);

private:

  //! Pointer to the enclosing group
  Alembic::Ogawa::OGroupPtr m_group;

};

//----------------------------------------------------------------------------//
// Template instantiations  
//----------------------------------------------------------------------------//

template <typename T>
OgODataset<T>::OgODataset(OgOGroup &parent, const std::string &name)
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

//----------------------------------------------------------------------------//

template <typename T>
void OgODataset<T>::addData(const size_t length, const T *data)
{
  m_group->addData(length * sizeof(T), data);
}

//----------------------------------------------------------------------------//

template <typename T>
OgOCDataset<T>::OgOCDataset(OgOGroup &parent, const std::string &name)
{
  // Create a group to store the basic data
  m_group = parent.addSubGroup();
  // Index 0 is the name
  writeString(m_group, name);
  // Index 1 is the type
  writeData(m_group, F3DCompressedDatasetType);
  // Index 2 is the data type
  writeDataType<T>(m_group);
  // Index 3 onwards are the actual dataset entries
}

//----------------------------------------------------------------------------//

template <typename T>
void OgOCDataset<T>::addData(const size_t byteLength, const uint8_t *data)
{
  m_group->addData(byteLength * sizeof(uint8_t), data);
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // include guard

//----------------------------------------------------------------------------//
