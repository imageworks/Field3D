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
#include "Exception.h"
#include "Field3DFile.h"
#include "Hdf5Util.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

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

  //! Writes the given field to disk. This function calls out to writeInternal
  //! once the template type has been determined.
  //! \return true if successful, otherwise false
  virtual bool write(hid_t layerGroup, FieldBase::Ptr field);

  //! Returns the class name
  virtual std::string className() const
  { return "MIPField"; }
  
private:

  // Internal methods ----------------------------------------------------------

  //! This call writes all the attributes and sets up the data space.
  template <template <typename X> class Field_T, class Data_T>
  bool writeInternal(hid_t layerGroup, 
                     typename MIPField<Field_T<Data_T> >::Ptr field);

  template <template <typename X> class Field_T, class Data_T>
  typename MIPField<Field_T<Data_T> >::Ptr 
  readInternal(hid_t layerGroup, const std::string &filename, 
               const std::string &layerPath, DataTypeEnum typeEnum);

  // Strings -------------------------------------------------------------------

  static const int         k_versionNumber;
  static const std::string k_versionAttrName;
  static const std::string k_extentsStr;
  static const std::string k_dataWindowStr;
  static const std::string k_componentsStr;
  static const std::string k_bitsPerComponentStr;
  static const std::string k_mipGroupStr;
  static const std::string k_levelGroupStr;
  static const std::string k_levelsStr;
  static const std::string k_baseTypeStr;

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
                        const DataTypeEnum &typeEnum)
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
    // Read the data
    FieldBase::Ptr field = io->read(*levelGroup, m_filename, 
                                    m_path, m_typeEnum);
    
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
typename MIPField<Field_T<Data_T> >::Ptr
MIPFieldIO::readInternal(hid_t layerGroup, 
                         const std::string &filename, 
                         const std::string &layerPath, 
                         DataTypeEnum typeEnum)
{
  using namespace detail;
  using namespace Exc;
  using namespace Hdf5Util;

  typedef MIPField<Field_T<Data_T> >              MIPType;
  typedef GenericLazyLoadAction<Field_T<Data_T> > Action;

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
// Template instantiations
//----------------------------------------------------------------------------//

template <typename Field_T> 
boost::mutex GenericLazyLoadAction<Field_T>::ms_hdf5Mutex;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
