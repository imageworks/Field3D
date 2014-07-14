//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_OgIDataset_H_
#define _INCLUDED_Field3D_OgIDataset_H_

//----------------------------------------------------------------------------//
// Includes
//----------------------------------------------------------------------------//

#include "OgUtil.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// OgIDataset
//----------------------------------------------------------------------------//

/*! \class OgIDataset
  Ogawa input data set. Reads elements of a dataset from an F3D-style Ogawa 
  file. 

  One dataset can contain multiple elements, each with a different length.

  It only supports linear arrays. Higher-dimensional arrays must be folded
  into a linear array or broken into individual elements.
*/

//----------------------------------------------------------------------------//

template <typename T>
class OgIDataset : public OgIBase
{

public:

  // Typedefs ------------------------------------------------------------------

  typedef T value_type;

  // Ctor, dtor ----------------------------------------------------------------

  //! The default constructor leaves m_group initialized, which implies it 
  //! is in an invalid state.
  OgIDataset();

  //! Initialize from an existing IGroup. The constructor will check that the
  //! type enum matches the template parameter, otherwise the attribute is
  //! marked invalid.
  OgIDataset(Alembic::Ogawa::IGroupPtr group);

  // Main methods --------------------------------------------------------------

  //! Returns the number of elements in the dataset
  size_t                  numDataElements() const;

  //! Returns the size of an individual element.
  //! \return -1 if index provided is not a data set
  Alembic::Util::uint64_t dataSize(const size_t index, 
                                   const size_t threadId) const;

  //! Reads the data. 
  //! \note Assumes that dataSize() has been called to ensure
  //! that the output pointer has enough storage allocated.
  bool                    getData(const size_t index, T *data, 
                                  const size_t threadId) const;

};

//----------------------------------------------------------------------------//
// OgICDataset
//----------------------------------------------------------------------------//

/*! \class OgICDataset
  Ogawa compressed input data set. Reads elements of a dataset from an 
  F3D-style Ogawa file. 

  One dataset can contain multiple elements, each with a different length.

  It only supports linear arrays. Higher-dimensional arrays must be folded
  into a linear array or broken into individual elements.
*/

//----------------------------------------------------------------------------//

template <typename T>
class OgICDataset : public OgIBase
{

public:

  // Typedefs ------------------------------------------------------------------

  typedef T value_type;

  // Ctor, dtor ----------------------------------------------------------------

  //! The default constructor leaves m_group initialized, which implies it 
  //! is in an invalid state.
  OgICDataset();

  //! Initialize from an existing IGroup. The constructor will check that the
  //! type enum matches the template parameter, otherwise the attribute is
  //! marked invalid.
  OgICDataset(Alembic::Ogawa::IGroupPtr group);

  // Main methods --------------------------------------------------------------

  //! Returns the number of elements in the dataset
  size_t                  numDataElements() const;

  //! Returns the size of an individual element.
  //! \return -1 if index provided is not a data set
  Alembic::Util::uint64_t dataSize(const size_t index, 
                                   const size_t threadId) const;

  //! Reads the data. 
  //! \note Assumes that dataSize() has been called to ensure
  //! that the output pointer has enough storage allocated.
  bool                    getData(const size_t index, uint8_t *data, 
                                  const size_t threadId) const;

};

//----------------------------------------------------------------------------//
// Template implementations
//----------------------------------------------------------------------------//

template <typename T>
OgIDataset<T>::OgIDataset()
{
  // Empty
}

//----------------------------------------------------------------------------//

template <typename T>
OgIDataset<T>::OgIDataset(Alembic::Ogawa::IGroupPtr group)
  : OgIBase(group)
{
  // Handle null pointer
  if (!m_group) {
    return;
  }
  // Check data type
  OgDataType dataType = readDataType(group, 2);
  if (dataType != OgawaTypeTraits<T>::typeEnum()) {
    m_group.reset();
    return;
  }
  // Update name
  getGroupName(m_group, m_name);
}

//----------------------------------------------------------------------------//

template <typename T>
size_t OgIDataset<T>::numDataElements() const
{
  return m_group->getNumChildren() - OGAWA_DATASET_BASEOFFSET;
}

//----------------------------------------------------------------------------//

template <typename T>
Alembic::Util::uint64_t 
OgIDataset<T>::dataSize(const size_t index, const size_t threadId) const
{
  // Indices start at OGAWA_DATASET_BASEOFFSET
  const size_t internalIndex = index + OGAWA_DATASET_BASEOFFSET;
  // Check that we have a data set
  if (!m_group->isChildData(internalIndex)) {
    return OGAWA_INVALID_DATASET_INDEX;
  }
  // Grab the data set
  Alembic::Ogawa::IDataPtr idata = m_group->getData(internalIndex, threadId);
  // Compute its length
  return idata->getSize() / sizeof(T);
}

//----------------------------------------------------------------------------//

template <typename T>
bool OgIDataset<T>::getData(const size_t index, T *data, 
                            const size_t threadId) const
{
  // Indices start at OGAWA_DATASET_BASEOFFSET
  const size_t internalIndex = index + OGAWA_DATASET_BASEOFFSET;
  // Check that we have a data set
  if (!m_group->isChildData(internalIndex)) {
    return false;
  }
  // Grab the data set
  Alembic::Ogawa::IDataPtr idata = m_group->getData(internalIndex, threadId);
  // Handle null pointer
  if (!idata) {
    return false;
  }
  // Get the length
  Alembic::Util::uint64_t dataSize = idata->getSize();
  // Read the data
  idata->read(dataSize, data, 0, threadId);
  // Done
  return true;
}

//----------------------------------------------------------------------------//

template <typename T>
OgICDataset<T>::OgICDataset()
{
  // Empty
}

//----------------------------------------------------------------------------//

template <typename T>
OgICDataset<T>::OgICDataset(Alembic::Ogawa::IGroupPtr group)
  : OgIBase(group)
{
  // Handle null pointer
  if (!m_group) {
    return;
  }
  // Check data type
  OgDataType dataType = readDataType(group, 2);
  if (dataType != OgawaTypeTraits<T>::typeEnum()) {
    m_group.reset();
    return;
  }
  // Update name
  getGroupName(m_group, m_name);
}

//----------------------------------------------------------------------------//

template <typename T>
size_t OgICDataset<T>::numDataElements() const
{
  return m_group->getNumChildren() - OGAWA_DATASET_BASEOFFSET;
}

//----------------------------------------------------------------------------//

template <typename T>
Alembic::Util::uint64_t 
OgICDataset<T>::dataSize(const size_t index, const size_t threadId) const
{
  // Indices start at OGAWA_DATASET_BASEOFFSET
  const size_t internalIndex = index + OGAWA_DATASET_BASEOFFSET;
  // Check that we have a data set
  if (!m_group->isChildData(internalIndex)) {
    return OGAWA_INVALID_DATASET_INDEX;
  }
  // Grab the data set
  Alembic::Ogawa::IDataPtr idata = m_group->getData(internalIndex, threadId);
  // Compute its length
  return idata->getSize() / sizeof(uint8_t);
}

//----------------------------------------------------------------------------//

template <typename T>
bool OgICDataset<T>::getData(const size_t index, uint8_t *data, 
                             const size_t threadId) const
{
  // Indices start at OGAWA_DATASET_BASEOFFSET
  const size_t internalIndex = index + OGAWA_DATASET_BASEOFFSET;
  // Check that we have a data set
  if (!m_group->isChildData(internalIndex)) {
    return false;
  }
  // Grab the data set
  Alembic::Ogawa::IDataPtr idata = m_group->getData(internalIndex, threadId);
  // Handle null pointer
  if (!idata) {
    return false;
  }
  // Get the length
  Alembic::Util::uint64_t dataSize = idata->getSize();
  // Read the data
  idata->read(dataSize, data, 0, threadId);
  // Done
  return true;
}

//----------------------------------------------------------------------------//
  
FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // include guard

//----------------------------------------------------------------------------//
