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

/*! \file MIPSparseFieldIO.h
  \brief Contains the MIPSparseFieldIO class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_MIPSparseFieldIO_H_
#define _INCLUDED_Field3D_MIPSparseFieldIO_H_

//----------------------------------------------------------------------------//

#include <string>

#include <boost/intrusive_ptr.hpp>

#include <hdf5.h>

#include "MIPSparseField.h"
#include "Exception.h"
#include "SparseFieldIO.h"
#include "Field3DFile.h"
#include "Hdf5Util.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// MIPSparseFieldIO
//----------------------------------------------------------------------------//

/*! \class MIPSparseFieldIO
  \ingroup file_int
  Handles IO for a MIPSparseField object
*/

//----------------------------------------------------------------------------//

class MIPSparseFieldIO : public FieldIO 
{

public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<MIPSparseFieldIO> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef MIPSparseFieldIO class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  static const char* classType()
  {
    return "MIPSparseFieldIO";
  }

  // Constructors --------------------------------------------------------------

  //! Ctor
  MIPSparseFieldIO() 
   : FieldIO()
  { }

  //! Dtor
  virtual ~MIPSparseFieldIO() 
  { /* Empty */ }

  static FieldIO::Ptr create()
  { return Ptr(new MIPSparseFieldIO); }

  // From FieldIO --------------------------------------------------------------

  //! Reads the field at the given location and tries to create a MIPSparseField
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
  { return "MIPSparseField"; }
  
private:

  // Internal methods ----------------------------------------------------------

  //! This call writes all the attributes and sets up the data space.
  template <class Data_T>
  bool writeInternal(hid_t layerGroup, 
                     typename MIPSparseField<Data_T>::Ptr field);
  template <class Data_T>
  typename MIPSparseField<Data_T>::Ptr readInternal(hid_t layerGroup,
                                                   const std::string &filename, 
                                                   const std::string &layerPath,
                                                   DataTypeEnum typeEnum);

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

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef FieldIO base;

};

//----------------------------------------------------------------------------//
// SparseFieldLazyLoadAction
//----------------------------------------------------------------------------//

template <class Field_T>
class SparseFieldLazyLoadAction 
  : public LazyLoadAction<Field_T>
{
public:
  
  // Typedefs ------------------------------------------------------------------

  typedef boost::shared_ptr<SparseFieldLazyLoadAction<Field_T> > Ptr;
  typedef std::vector<Ptr> Vec;

  // Constructor ---------------------------------------------------------------

  SparseFieldLazyLoadAction(const std::string &filename,
                            const std::string &path,
                            const DataTypeEnum &typeEnum)
    : LazyLoadAction<Field_T>(),
      m_filename(filename), m_path(path), m_typeEnum(typeEnum)
  {
    // Empty
  }

  ~SparseFieldLazyLoadAction()
  { }

  // From LazyLoadAction -------------------------------------------------------

  virtual typename Field_T::Ptr load() const
  {
    using namespace Hdf5Util;

    hid_t file = H5Fopen(m_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
      throw Exc::NoSuchFileException(m_filename);
    H5ScopedGopen levelGroup(file, m_path);
    SparseFieldIO io;
    FieldBase::Ptr field = io.read(levelGroup, "", "", m_typeEnum);
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

};

//----------------------------------------------------------------------------//
// Templated methods
//----------------------------------------------------------------------------//

template <class Data_T>
bool MIPSparseFieldIO::writeInternal(hid_t layerGroup, 
                                     typename MIPSparseField<Data_T>::Ptr field)
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
    SparseFieldIO io;
    io.write(levelGroup, field->mipLevel(i));

  }

  return true; 
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MIPSparseField<Data_T>::Ptr
MIPSparseFieldIO::readInternal(hid_t layerGroup, 
                              const std::string &filename, 
                              const std::string &layerPath, 
                              DataTypeEnum typeEnum)
{
  using namespace Exc;
  using namespace Hdf5Util;

  typedef typename SparseField<Data_T>::Ptr SparseFieldPtr;

  typedef SparseFieldLazyLoadAction<SparseField<Data_T> > Action;

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
  typename MIPSparseField<Data_T>::Ptr result(new MIPSparseField<Data_T>);

  // Open the MIP field group
  H5ScopedGopen mipGroup(layerGroup, k_mipGroupStr);

  // Read num levels
  int numLevels;
  if (!readAttribute(mipGroup, k_levelsStr, 1, numLevels)) 
    throw MissingAttributeException("Couldn't find attribute " + k_levelsStr);

  // Read each level ---

  std::vector<typename EmptyField<Data_T>::Ptr> proxies;
  typename LazyLoadAction<SparseField<Data_T> >::Vec actions;

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

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
