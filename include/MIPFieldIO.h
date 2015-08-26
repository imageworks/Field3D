//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks Inc
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the
 * distribution.  Neither the name of Sony Pictures Imageworks nor the
 * names of its contributors may be used to endorse or promote
 * products derived from this software without specific prior written
 * permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//----------------------------------------------------------------------------//

/*! \file MIPFieldIO.h
  \brief Contains the MIPFieldIO class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_MIPFieldIO_H_
#define _INCLUDED_Field3D_MIPFieldIO_H_

//----------------------------------------------------------------------------//

#include <string>

#include <boost/intrusive_ptr.hpp>
#include <boost/thread/mutex.hpp>

#include <hdf5.h>

#include "MIPField.h"
#include "OgIO.h"
#include "Exception.h"
#include "Field3DFile.h"
#include "Hdf5Util.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Exceptions
//----------------------------------------------------------------------------//

namespace Exc {

DECLARE_FIELD3D_GENERIC_EXCEPTION(MIPFieldIOException, Exception)

} // namespace Exc

//----------------------------------------------------------------------------//
// MIPFieldIO
//----------------------------------------------------------------------------//

/*! \class MIPFieldIO
  \ingroup file_int
  Handles IO for a MIPField object
*/

//----------------------------------------------------------------------------//

class MIPFieldIO : public FieldIO 
{

public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<MIPFieldIO> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef MIPFieldIO class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  static const char* staticClassType()
  {
    return "MIPFieldIO";
  }

  // Constructors --------------------------------------------------------------

  //! Ctor
  MIPFieldIO() 
   : FieldIO()
  { }

  //! Dtor
  virtual ~MIPFieldIO() 
  { /* Empty */ }

  static FieldIO::Ptr create()
  { return Ptr(new MIPFieldIO); }

  // From FieldIO --------------------------------------------------------------

  //! Reads the field at the given location and tries to create a MIPField
  //! object from it. Calls out to readData() for template-specific work.
  //! \returns Null if no object was read
  virtual FieldBase::Ptr read(hid_t layerGroup, 
                              const std::string &filename, 
                              const std::string &layerPath,
                              DataTypeEnum typeEnum);

  //! Reads the field at the given location and tries to create a MIPField
  //! object from it. Calls out to readData() for template-specific work.
  //! \returns Null if no object was read
  virtual FieldBase::Ptr read(const OgIGroup &layerGroup, 
                              const std::string &filename, 
                              const std::string &layerPath,
                              OgDataType typeEnum);

  //! Writes the given field to disk. This function calls out to writeInternal
  //! once the template type has been determined.
  //! \return true if successful, otherwise false
  virtual bool write(hid_t layerGroup, FieldBase::Ptr field);

  //! Writes the given field to disk. This function calls out to writeInternal
  //! once the template type has been determined.
  //! \return true if successful, otherwise false
  virtual bool write(OgOGroup &layerGroup, FieldBase::Ptr field);

  //! Returns the class name
  virtual std::string className() const
  { return "MIPField"; }
  
private:

  // Internal methods ----------------------------------------------------------

  //! This call writes all the attributes and sets up the data space.
  template <template <typename X> class Field_T, class Data_T>
  bool writeInternal(hid_t layerGroup, 
                     typename MIPField<Field_T<Data_T> >::Ptr field);
  //! This call writes all the attributes and sets up the data space.
  template <template <typename X> class Field_T, class Data_T>
  bool writeInternal(OgOGroup &layerGroup, 
                     typename MIPField<Field_T<Data_T> >::Ptr field);

  template <template <typename X> class Field_T, class Data_T>
  typename MIPField<Field_T<Data_T> >::Ptr 
  readInternal(hid_t layerGroup, const std::string &filename, 
               const std::string &layerPath, DataTypeEnum typeEnum);
  template <template <typename X> class Field_T, class Data_T>
  typename MIPField<Field_T<Data_T> >::Ptr 
  readInternal(const OgIGroup &layerGroup, const std::string &filename, 
               const std::string &layerPath, OgDataType typeEnum);

  // Strings -------------------------------------------------------------------

  static const int         k_versionNumber;
  static const std::string k_versionAttrName;
  static const std::string k_extentsStr;
  static const std::string k_extentsMinStr;
  static const std::string k_extentsMaxStr;
  static const std::string k_dataWindowStr;
  static const std::string k_dataWindowMinStr;
  static const std::string k_dataWindowMaxStr;
  static const std::string k_componentsStr;
  static const std::string k_bitsPerComponentStr;
  static const std::string k_mipGroupStr;
  static const std::string k_levelGroupStr;
  static const std::string k_levelsStr;
  static const std::string k_baseTypeStr;
  static const std::string k_dummyDataStr;

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef FieldIO base;

};

//----------------------------------------------------------------------------//
// Implementation details
//----------------------------------------------------------------------------//

namespace detail {

  //! We need to instantiate MIPSparseField rather than MIPField<SparseField>.
  //! This traits class does that.
  template <typename Field_T>
  struct MIPInstance;

  //! Specialization for sparse field
  template <typename Data_T>
  struct MIPInstance<SparseField<Data_T> >
  {
    typedef MIPSparseField<Data_T> type;
  };

  // Specialization for dense field
  template <typename Data_T>
  struct MIPInstance<DenseField<Data_T> >
  {
    typedef MIPDenseField<Data_T> type;
  };

}

//----------------------------------------------------------------------------//
// GenericLazyLoadActionHDF5
//----------------------------------------------------------------------------//

template <class Field_T>
class GenericLazyLoadActionHDF5
  : public LazyLoadAction<Field_T>
{
public:
  
  // Typedefs ------------------------------------------------------------------

  typedef boost::shared_ptr<GenericLazyLoadActionHDF5<Field_T> > Ptr;
  typedef std::vector<Ptr>                                       Vec;

  // Constructor ---------------------------------------------------------------

  GenericLazyLoadActionHDF5(const std::string &filename,
                            const std::string &path,
                            const DataTypeEnum &typeEnum)
    : LazyLoadAction<Field_T>(),
      m_filename(filename), m_path(path), m_typeEnum(typeEnum)
  {
    // Empty
  }

  ~GenericLazyLoadActionHDF5()
  { }

  // From LazyLoadAction -------------------------------------------------------

  virtual typename Field_T::Ptr load() const
  {
    using namespace Hdf5Util;

    hid_t file;
    boost::shared_ptr<H5ScopedGopen> levelGroup;

    {
      // Hold lock while calling H5Fopen
      GlobalLock lock(g_hdf5Mutex);
      
      // Open the HDF5 file
      file = H5Fopen(m_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      if (file < 0) {
        throw Exc::NoSuchFileException(m_filename);
      }
      levelGroup.reset(new H5ScopedGopen(file, m_path));
    }

    // Instantiate I/O
    FieldIO::Ptr io = 
      ClassFactory::singleton().createFieldIO(Field_T::staticClassName());
    FieldBase::Ptr field = io->read(*levelGroup, m_filename, m_path, m_typeEnum);
    if (!field) {
      throw Exc::MIPFieldIOException("Failed to read MIP level from disk.");
    }
    {
      // Hold lock again
      GlobalLock lock(g_hdf5Mutex);
      // Close the file
      if (H5Fclose(file) < 0) {
        Msg::print("Error closing file: " + m_filename);
      } 
    }
    // Done
    return field_dynamic_cast<Field_T>(field);
  }

private:

  // Data members --------------------------------------------------------------

  //! Filename
  const std::string m_filename;
  //! Path in file
  const std::string m_path;
  //! Data type enum
  const DataTypeEnum m_typeEnum;

  //! HDF5 access mutex
  static boost::mutex ms_hdf5Mutex;

};

//----------------------------------------------------------------------------//
// GenericLazyLoadAction
//----------------------------------------------------------------------------//

template <class Field_T>
class GenericLazyLoadAction
  : public LazyLoadAction<Field_T>
{
public:
  
  // Typedefs ------------------------------------------------------------------

  typedef boost::shared_ptr<GenericLazyLoadAction<Field_T> > Ptr;
  typedef std::vector<Ptr>                                   Vec;

  // Constructor ---------------------------------------------------------------

  GenericLazyLoadAction(const std::string &filename,
                        const std::string &path,
                        const OgDataType &typeEnum)
    : LazyLoadAction<Field_T>(),
      m_filename(filename), m_path(path), m_typeEnum(typeEnum)
  {
    // Empty
  }

  ~GenericLazyLoadAction()
  { }

  // From LazyLoadAction -------------------------------------------------------

  virtual typename Field_T::Ptr load() const
  {
    // Lock Ogawa access
    boost::mutex::scoped_lock lock(ms_ogawaMutex);

    Alembic::Ogawa::IArchive archive(m_filename);
    if (!archive.isValid()) {
      throw Exc::NoSuchFileException(m_filename);
    }

    // Open the root level
    OgIGroup root(archive);

    // Traverse all the way down to the MIP level
    OgIGroup levelGroup = root.findGroup(m_path);

    FieldIO::Ptr io = 
      ClassFactory::singleton().createFieldIO(Field_T::staticClassName());
    FieldBase::Ptr field = io->read(levelGroup, m_filename, m_path, m_typeEnum);
    if (!field) {
      throw Exc::MIPFieldIOException("Failed to read MIP level from disk.");
    }
    return field_dynamic_cast<Field_T>(field);
  }

private:

  // Data members --------------------------------------------------------------

  //! Filename
  const std::string m_filename;
  //! Path in file
  const std::string m_path;
  //! Data type enum
  const OgDataType m_typeEnum;

  //! Ogawa access mutex
  static boost::mutex ms_ogawaMutex;

};

//----------------------------------------------------------------------------//
// Templated methods
//----------------------------------------------------------------------------//

template <template <typename X> class Field_T, class Data_T>
bool MIPFieldIO::writeInternal(hid_t layerGroup, 
                               typename MIPField<Field_T<Data_T> >::Ptr field)
{
  using namespace Exc;
  using namespace Hdf5Util;

  Box3i ext(field->extents()), dw(field->dataWindow());

  // Add extents attribute ---

  int extents[6] = 
    { ext.min.x, ext.min.y, ext.min.z, ext.max.x, ext.max.y, ext.max.z };

  if (!writeAttribute(layerGroup, k_extentsStr, 6, extents[0])) {
    throw WriteAttributeException("Couldn't write attribute " + k_extentsStr);
  }

  // Add data window attribute ---

  int dataWindow[6] = 
    { dw.min.x, dw.min.y, dw.min.z, dw.max.x, dw.max.y, dw.max.z };

  if (!writeAttribute(layerGroup, k_dataWindowStr, 6, dataWindow[0])) {
    throw WriteAttributeException("Couldn't write attribute " + k_dataWindowStr);
  }

  // Add components attribute ---

  int components = FieldTraits<Data_T>::dataDims();

  if (!writeAttribute(layerGroup, k_componentsStr, 1, components)) {
    throw WriteAttributeException("Couldn't write attribute " + k_componentsStr);
  }

  // Add the bits per component attribute ---

  int bits = DataTypeTraits<Data_T>::h5bits();
  if (!writeAttribute(layerGroup, k_bitsPerComponentStr, 1, bits)) {
    Msg::print(Msg::SevWarning, "Error adding bits per component attribute.");
    return false;
  }

  // Add the base type attribute ---

  std::string baseType = Field_T<Data_T>::staticClassName();
  if (!writeAttribute(layerGroup, k_baseTypeStr, baseType)) {
    throw WriteAttributeException("Couldn't write attribute " + k_baseTypeStr);
  }

  // Add the mip fields group

  H5ScopedGcreate mipGroup(layerGroup, k_mipGroupStr);

  // Write number of layers
  int numLevels = field->numLevels();
  if (!writeAttribute(mipGroup, k_levelsStr, 1, numLevels)) {
    throw WriteAttributeException("Couldn't write attribute " + k_levelsStr);
  }

  // For each level ---

  for (size_t i = 0; i < field->numLevels(); i++) {

    // Add a named group
    H5ScopedGcreate levelGroup(mipGroup, k_levelGroupStr + "." + 
                               boost::lexical_cast<std::string>(i));
    
    // Add the field to the group
    std::string className = Field_T<Data_T>::staticClassName();
    FieldIO::Ptr io = 
      ClassFactory::singleton().createFieldIO(className);
    io->write(levelGroup, field->mipLevel(i));

  }

  return true; 
}

//----------------------------------------------------------------------------//

template <template <typename X> class Field_T, class Data_T>
bool MIPFieldIO::writeInternal(OgOGroup &layerGroup, 
                               typename MIPField<Field_T<Data_T> >::Ptr field)
{
  using namespace Exc;
  using namespace Hdf5Util;
  using std::string;

  const Box3i ext(field->extents()), dw(field->dataWindow());

  const std::string baseType   = Field_T<Data_T>::staticClassName();
  const int         numLevels  = field->numLevels();
  const int         components = FieldTraits<Data_T>::dataDims();
  const int bits = DataTypeTraits<Data_T>::h5bits();
  
  
  OgOAttribute<veci32_t> extMinAttr(layerGroup, k_extentsMinStr, ext.min);
  OgOAttribute<veci32_t> extMaxAttr(layerGroup, k_extentsMaxStr, ext.max);
  
  OgOAttribute<veci32_t> dwMinAttr(layerGroup, k_dataWindowMinStr, dw.min);
  OgOAttribute<veci32_t> dwMaxAttr(layerGroup, k_dataWindowMaxStr, dw.max);

  OgOAttribute<uint8_t> componentsAttr(layerGroup, k_componentsStr, components);

  OgOAttribute<uint8_t> bitsAttr(layerGroup, k_bitsPerComponentStr, bits);

  OgOAttribute<string> baseTypeAttr(layerGroup, k_baseTypeStr, baseType);

  // This is used to record Data_T cleanly
  OgODataset<Data_T> dummyDataset(layerGroup, k_dummyDataStr);

  OgOGroup mipGroup(layerGroup, k_mipGroupStr);

  // Add the mip fields group

  OgOAttribute<uint32_t> numLevelsAttr(mipGroup, k_levelsStr, numLevels);

  // For each level ---

  for (size_t i = 0; i < field->numLevels(); i++) {

    // Add a named group
    OgOGroup levelGroup(mipGroup, k_levelGroupStr + "." + 
                        boost::lexical_cast<std::string>(i));
    
    // Add the field to the group
    std::string className = Field_T<Data_T>::staticClassName();
    FieldIO::Ptr io = 
      ClassFactory::singleton().createFieldIO(className);
    io->write(levelGroup, field->mipLevel(i));

  }

  return true; 
}

//----------------------------------------------------------------------------//

template <template <typename X> class Field_T, class Data_T>
typename MIPField<Field_T<Data_T> >::Ptr
MIPFieldIO::readInternal(hid_t layerGroup, 
                         const std::string &filename, 
                         const std::string &layerPath, 
                         DataTypeEnum typeEnum)
{
  using namespace detail;
  using namespace Exc;
  using namespace Hdf5Util;

  typedef MIPField<Field_T<Data_T> >                  MIPType;
  typedef GenericLazyLoadActionHDF5<Field_T<Data_T> > Action;

  // Read extents
  Box3i extents;
  if (!readAttribute(layerGroup, k_extentsStr, 6, extents.min.x)) 
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_extentsStr);
  
  // Read data window
  Box3i dataW;
  if (!readAttribute(layerGroup, k_dataWindowStr, 6, dataW.min.x)) 
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_dataWindowStr);
  
  // Read components
  int components;
  if (!readAttribute(layerGroup, k_componentsStr, 1, components)) 
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_componentsStr);
  
  // Construct the resulting MIP field
  typename MIPType::Ptr result(new typename MIPInstance<Field_T<Data_T> >::type);

  // Open the MIP field group
  H5ScopedGopen mipGroup(layerGroup, k_mipGroupStr);

  // Read num levels
  int numLevels;
  if (!readAttribute(mipGroup, k_levelsStr, 1, numLevels)) 
    throw MissingAttributeException("Couldn't find attribute " + k_levelsStr);

  // Read each level ---

  std::vector<typename EmptyField<Data_T>::Ptr>  proxies;
  typename LazyLoadAction<Field_T<Data_T> >::Vec actions;

  for (int i = 0; i < numLevels; i++) {
    // Open level group
    std::string levelGroupStr = 
      k_levelGroupStr + "." + boost::lexical_cast<std::string>(i);
    H5ScopedGopen levelGroup(mipGroup, levelGroupStr);
    // Read the proxy
    typename EmptyField<Data_T>::Ptr proxy(new EmptyField<Data_T>);
    // Read extents
    if (!readAttribute(levelGroup, k_extentsStr, 6, extents.min.x)) 
      throw MissingAttributeException("Couldn't find attribute " + 
                                    k_extentsStr);
    // Read data window
    if (!readAttribute(levelGroup, k_dataWindowStr, 6, dataW.min.x)) 
      throw MissingAttributeException("Couldn't find attribute " + 
                                      k_dataWindowStr);
    // Configure proxy with resolution
    proxy->setSize(extents, dataW);
    proxies.push_back(proxy);
    // Make the lazy load action
    std::string fullPath = 
      layerPath + "/" + k_mipGroupStr + "/" + levelGroupStr;
    typename Action::Ptr action(new Action(filename, fullPath, typeEnum));
    actions.push_back(action);
  }

  result->setupLazyLoad(proxies, actions);

  return result;
}

//----------------------------------------------------------------------------//

template <template <typename X> class Field_T, class Data_T>
typename MIPField<Field_T<Data_T> >::Ptr
MIPFieldIO::readInternal(const OgIGroup &layerGroup, 
                         const std::string &filename, 
                         const std::string &layerPath, 
                         OgDataType typeEnum)
{
  using namespace detail;
  using namespace Exc;

  typedef MIPField<Field_T<Data_T> >              MIPType;
  typedef GenericLazyLoadAction<Field_T<Data_T> > Action;

  Box3i extents, dataW;

#if 0

  // Get extents ---

  OgIAttribute<veci32_t> extMinAttr = 
    layerGroup.findAttribute<veci32_t>(k_extentsMinStr);
  OgIAttribute<veci32_t> extMaxAttr = 
    layerGroup.findAttribute<veci32_t>(k_extentsMaxStr);
  if (!extMinAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_extentsMinStr);
  }
  if (!extMaxAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_extentsMaxStr);
  }

  extents.min = extMinAttr.value();
  extents.max = extMaxAttr.value();

  // Get data window ---

  OgIAttribute<veci32_t> dwMinAttr = 
    layerGroup.findAttribute<veci32_t>(k_dataWindowMinStr);
  OgIAttribute<veci32_t> dwMaxAttr = 
    layerGroup.findAttribute<veci32_t>(k_dataWindowMaxStr);
  if (!dwMinAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_dataWindowMinStr);
  }
  if (!dwMaxAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_dataWindowMaxStr);
  }

  dataW.min = dwMinAttr.value();
  dataW.max = dwMaxAttr.value();

#endif

  // Get num components ---

  OgIAttribute<uint8_t> numComponentsAttr = 
    layerGroup.findAttribute<uint8_t>(k_componentsStr);
  if (!numComponentsAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + 
                                    k_componentsStr);
  }
  
  // Construct the resulting MIP field
  typename MIPType::Ptr result(
    new typename MIPInstance<Field_T<Data_T> >::type);

  // Open the MIP field group
  OgIGroup mipGroup = layerGroup.findGroup(k_mipGroupStr);
  if (!mipGroup.isValid()) {
    MissingAttributeException("Couldn't find group " + k_mipGroupStr);
  }

  // Read num levels
  OgIAttribute<uint32_t> numLevelsAttr = 
    mipGroup.findAttribute<uint32_t>(k_levelsStr);
  if (!numLevelsAttr.isValid()) {
    throw MissingAttributeException("Couldn't find attribute " + k_levelsStr);
  }
  int numLevels = numLevelsAttr.value();

  // Read each level ---

  std::vector<typename EmptyField<Data_T>::Ptr>  proxies;
  typename LazyLoadAction<Field_T<Data_T> >::Vec actions;

  for (int i = 0; i < numLevels; i++) {
    // Open level group
    std::string levelGroupStr = 
      k_levelGroupStr + "." + boost::lexical_cast<std::string>(i);
    OgIGroup levelGroup = mipGroup.findGroup(levelGroupStr);
    // Read the proxy
    typename EmptyField<Data_T>::Ptr proxy(new EmptyField<Data_T>);
    // Read extents
    OgIAttribute<veci32_t> extMinAttr = 
      levelGroup.findAttribute<veci32_t>(k_extentsMinStr);
    OgIAttribute<veci32_t> extMaxAttr = 
      levelGroup.findAttribute<veci32_t>(k_extentsMaxStr);
    if (!extMinAttr.isValid()) {
      throw MissingAttributeException("Couldn't find attribute " + 
                                      k_extentsMinStr);
    }
    if (!extMaxAttr.isValid()) {
      throw MissingAttributeException("Couldn't find attribute " + 
                                      k_extentsMaxStr);
    }
    extents.min = extMinAttr.value();
    extents.max = extMaxAttr.value();
    // Read data window
    OgIAttribute<veci32_t> dwMinAttr = 
      levelGroup.findAttribute<veci32_t>(k_dataWindowMinStr);
    OgIAttribute<veci32_t> dwMaxAttr = 
      levelGroup.findAttribute<veci32_t>(k_dataWindowMaxStr);
    if (!dwMinAttr.isValid()) {
      throw MissingAttributeException("Couldn't find attribute " + 
                                      k_dataWindowMinStr);
    }
    if (!dwMaxAttr.isValid()) {
      throw MissingAttributeException("Couldn't find attribute " + 
                                      k_dataWindowMaxStr);
    }
    dataW.min = dwMinAttr.value();
    dataW.max = dwMaxAttr.value();
    // Configure proxy with resolution
    proxy->setSize(extents, dataW);
    proxies.push_back(proxy);
    // Make the lazy load action
    std::string fullPath = 
      layerPath + "/" + k_mipGroupStr + "/" + levelGroupStr;
    typename Action::Ptr action(new Action(filename, fullPath, typeEnum));
    actions.push_back(action);
  }

  result->setupLazyLoad(proxies, actions);

  return result;
}

//----------------------------------------------------------------------------//
// Template instantiations
//----------------------------------------------------------------------------//

template <typename Field_T> 
boost::mutex GenericLazyLoadActionHDF5<Field_T>::ms_hdf5Mutex;

//----------------------------------------------------------------------------//

template <typename Field_T> 
boost::mutex GenericLazyLoadAction<Field_T>::ms_ogawaMutex;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
