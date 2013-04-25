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

/*! \file MIPSparseField.h
  \brief Contains the MIPSparseField class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_MIPSparseField_H_
#define _INCLUDED_MIPSparseField_H_

#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/thread/mutex.hpp>

#include "SparseField.h"
#include "EmptyField.h"
#include "MIPField.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Exceptions
//----------------------------------------------------------------------------//

namespace Exc {

DECLARE_FIELD3D_GENERIC_EXCEPTION(MIPSparseFieldException, Exception)

} // namespace Exc

//----------------------------------------------------------------------------//
// Forward declarations 
//----------------------------------------------------------------------------//

template <class T>
class LinearMIPSparseFieldInterp; 
template <class T>
class CubicMIPSparseFieldInterp; 

//----------------------------------------------------------------------------//
// MIPSparseField
//----------------------------------------------------------------------------//

/*! \class MIPSparseField
  \ingroup field
  \brief This subclass stores a MIP representation of SparseField.

  Each level in the MIPSparseField is stored as a SparseField, and each level
  shares the same FieldMapping definition, even though their resolution is
  different.

  The class is lazy loading, such that no MIP levels are read from disk until
  they are needed. On top of this, standard SparseField caching (memory 
  limiting) is available, and operates the same as normal SparseFields. 

  The class is thread safe, and ensures that data is read from disk from in one
  single thread, using the double-checked locking mechanism.

  Interpolation into a MIP field may be done either directly to a single level,
  or by blending between two MIP levels. When blending, each field is assumed
  to match the other levels only in local-space.
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class MIPSparseField
  : public MIPField<Data_T>, public boost::noncopyable
{
public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<MIPSparseField> Ptr;
  typedef std::vector<Ptr> Vec;

  typedef LinearMIPSparseFieldInterp<Data_T> LinearInterp;
  typedef CubicMIPSparseFieldInterp<Data_T> CubicInterp;

  typedef Data_T value_type;
  typedef SparseField<Data_T> ContainedType;

  typedef SparseField<Data_T> SparseType;
  typedef typename SparseType::Ptr SparsePtr;
  typedef std::vector<SparsePtr> SparseVec;

  typedef EmptyField<Data_T> ProxyField;
  typedef typename ProxyField::Ptr ProxyPtr;
  typedef std::vector<ProxyPtr> ProxyVec;  

  // Constructors --------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  //! Constructs an empty MIP field
  MIPSparseField();

  // \}

  // From Field base class -----------------------------------------------------

  //! \name From Field
  //! \{  
  //! For a MIP field, the common value() call accesses data at level 0 only.
  virtual Data_T value(int i, int j, int k) const;
  //! Returns -current- memory use, rather than the amount used if all
  //! levels were loaded.
  virtual long long int memSize() const;
  //! \}

  // RTTI replacement ----------------------------------------------------------

  typedef MIPSparseField<Data_T> class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS

  static const char *staticClassName()
  {
    return "MIPSparseField";
  }

  static const char *classType()
  {
    return MIPSparseField<Data_T>::ms_classType.name();
  }
    
  // From MIPField -------------------------------------------------------------

  virtual Data_T mipValue(size_t level, int i, int j, int k) const;
  virtual V3i mipResolution(size_t level) const;

  // Concrete voxel access -----------------------------------------------------

  //! Read access to voxel at a given MIP level. 
  Data_T fastMipValue(size_t level, int i, int j, int k) const;

  // From FieldBase ------------------------------------------------------------

  //! \name From FieldBase
  //! \{

  virtual std::string className() const
  { return staticClassName(); }
  
  virtual FieldBase::Ptr clone() const
  { return Ptr(); }

  //! \}

  // Main methods --------------------------------------------------------------

  //! Clears all the levels of the MIP field.
  void clear();
  //! Sets up the MIP field given a set of SparseFields
  //! This call performs sanity checking to ensure that MIP properties are
  //! satisfied for each level.
  //! In this case, all MIP levels are available in memory.
  //! \note The MIP level order is implied to be zero-first.
  void setup(const SparseVec &fields);
  //! Sets up the MIP field in lazy-load mode
  //! \param mipGroupPath Path in F3D file to read data from.
  void setupLazyLoad(const ProxyVec &proxies,
                     const typename LazyLoadAction<SparseType>::Vec &actions);

  //! Returns a pointer to a MIP level
  SparsePtr mipLevel(size_t level);

protected:

  // Typedefs ------------------------------------------------------------------

  typedef MIPField<Data_T> base;

  // Static data members -------------------------------------------------------

  static TemplatedFieldType<MIPSparseField<Data_T> > ms_classType;

  // Data members --------------------------------------------------------------

  //! Storage of all MIP levels. Some or all of the pointers may be NULL.
  //! This is mutable because it needs updating during lazy loading of data.
  mutable std::vector<SparsePtr> m_fields;
  //! Lazy load actions. Only used if setupLazyLoad() has been called.
  mutable typename LazyLoadAction<SparseType>::Vec m_loadActions;
  //! Raw pointers to MIP levels.
  //! \note Important note: This is also used as the double-checked locking
  //! indicator.
  mutable std::vector<SparseField<Data_T>*> m_rawFields;
  //! Resolution of each MIP level.
  mutable std::vector<V3i> m_mipRes;
  //! Mutex lock around IO. Used to make sure only one thread reads MIP 
  //! level data at a time
  mutable boost::mutex m_ioMutex;

  // Utility methods -----------------------------------------------------------

  //! Updates the mapping, extents and data window to match the given field.
  //! Used so that the MIPField will appear to have the same mapping in space
  //! as the level-0 field.
  void updateMapping(FieldRes::Ptr field);
  //! Updates the dependent data members based on m_field
  void updateAuxMembers() const;
  //! Loads the given level from disk
  void loadLevelFromDisk(size_t level) const;
  //! Sanity checks to ensure that the provided Fields are a MIP representation
  template <class Field_T>
  void sanityChecks(const std::vector<typename Field_T::Ptr> &fields);

};

//----------------------------------------------------------------------------//
// Typedefs
//----------------------------------------------------------------------------//

typedef MIPSparseField<half>   MIPSparseFieldh;
typedef MIPSparseField<float>  MIPSparseFieldf;
typedef MIPSparseField<double> MIPSparseFieldd;
typedef MIPSparseField<V3h>    MIPSparseField3h;
typedef MIPSparseField<V3f>    MIPSparseField3f;
typedef MIPSparseField<V3d>    MIPSparseField3d;

//----------------------------------------------------------------------------//
// MIPSparseField implementations
//----------------------------------------------------------------------------//

template <class Data_T>
MIPSparseField<Data_T>::MIPSparseField()
  : base()
{
  m_fields.resize(base::m_numLevels);
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MIPSparseField<Data_T>::clear()
{
  m_fields.clear();
  m_rawFields.clear();
  base::m_numLevels = 0;
  base::m_lowestLevel = 0;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MIPSparseField<Data_T>::setup(const SparseVec &fields)
{
  // Clear existing data
  clear();
  // Run sanity checks. This will throw an exception if the fields are invalid.
  sanityChecks<SparseField<Data_T> >(fields);
  // Update state of object
  m_fields = fields;
  base::m_numLevels = fields.size();
  base::m_lowestLevel = 0;
  updateMapping(fields[0]);
  updateAuxMembers();
  // Update mip res from real fields
  m_mipRes.resize(base::m_numLevels);
  for (size_t i = 0; i < fields.size(); i++) {
    m_mipRes[i] = m_fields[i]->extents().size() + V3i(1);
  } 
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MIPSparseField<Data_T>::setupLazyLoad
(const ProxyVec &proxies,
 const typename LazyLoadAction<SparseType>::Vec &actions)
{
  using namespace Exc;

  // Clear existing data
  clear();
  // Check same number of proxies and actions
  if (proxies.size() != actions.size()) {
    throw MIPSparseFieldException("Incorrect number of lazy load actions");
  }  
  // Run sanity checks. This will throw an exception if the fields are invalid.
  sanityChecks<EmptyField<Data_T> >(proxies);
  // Store the lazy load actions
  m_loadActions = actions;
  // Update state of object
  base::m_numLevels = proxies.size();
  base::m_lowestLevel = 0;
  m_fields.resize(base::m_numLevels);
  updateMapping(proxies[0]);
  updateAuxMembers();
  // Update mip res from proxy fields
  m_mipRes.resize(base::m_numLevels);
  for (size_t i = 0; i < proxies.size(); i++) {
    m_mipRes[i] = proxies[i]->extents().size() + V3i(1);
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MIPSparseField<Data_T>::SparsePtr 
MIPSparseField<Data_T>::mipLevel(size_t level)
{
  assert(level < base::m_numLevels);
  // Ensure level is loaded.
  if (!m_rawFields[level]) {
    loadLevelFromDisk(level);
  } else {
  }
  return m_fields[level];
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T MIPSparseField<Data_T>::value(int i, int j, int k) const
{
  return fastMipValue(0, i, j, k);
}

//----------------------------------------------------------------------------//

template <class Data_T>
long long int MIPSparseField<Data_T>::memSize() const
{ 
  long long int mem = 0;
  for (size_t i = 0; i < m_fields.size(); i++) {
    if (m_fields[i]) {
      mem += m_fields[i]->memSize();
    }
  }
  return mem + sizeof(*this);
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T MIPSparseField<Data_T>::mipValue(size_t level,
                                              int i, int j, int k) const
{
  return fastMipValue(level, i, j, k);
}

//----------------------------------------------------------------------------//

template <class Data_T>
V3i MIPSparseField<Data_T>::mipResolution(size_t level) const
{
  assert(level < base::m_numLevels);
  return m_mipRes[level];
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T MIPSparseField<Data_T>::fastMipValue(size_t level,
                                                  int i, int j, int k) const
{
  assert(level < base::m_numLevels);
  // Ensure level is loaded.
  if (!m_rawFields[level]) {
    loadLevelFromDisk(level);
  }
  // Read from given level
  return m_rawFields[level]->fastValue(i, j, k);
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MIPSparseField<Data_T>::updateAuxMembers() const
{
  m_rawFields.resize(m_fields.size());
  for (size_t i = 0; i < m_fields.size(); i++) {
    m_rawFields[i] = m_fields[i].get();
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MIPSparseField<Data_T>::updateMapping(FieldRes::Ptr field)
{
  base::m_extents = field->extents();
  base::m_dataWindow = field->dataWindow();
  base::setMapping(field->mapping());
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MIPSparseField<Data_T>::loadLevelFromDisk(size_t level) const
{
  // Double-check locking
  if (!m_rawFields[level]) {
    boost::mutex::scoped_lock lock(m_ioMutex);
    if (!m_rawFields[level]) {
      // Execute the lazy load action
      m_fields[level] = m_loadActions[level]->load();
      // Remove lazy load action
      m_loadActions[level].reset();
      // Update aux data
      updateAuxMembers();
    }
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
template <class Field_T>
void
MIPSparseField<Data_T>::
sanityChecks(const std::vector<typename Field_T::Ptr> &fields)
{
  using boost::lexical_cast;
  using std::string;
  using Exc::MIPSparseFieldException;

  // Check zero length
  if (fields.size() == 0) {
    throw MIPSparseFieldException("Zero fields in input");
  }
  // Check all non-null
  for (size_t i = 0; i < fields.size(); i++) {
    if (!fields[i]) {
      throw MIPSparseFieldException("Found null pointer in input");
    }
  }
  // Grab first field
  typename Field_T::Ptr base = fields[0];
  // Check common mapping
  FieldMapping::Ptr baseMapping = base->mapping();
  for (size_t i = 1; i < fields.size(); i++) {
#if 0 // Replace with something less strict
    if (!baseMapping->isIdentical(fields[i]->mapping())) {
      throw MIPSparseFieldException("Field " + 
                                   lexical_cast<string>(i) + 
                                   " has non-matching Mapping");
    }
#endif
  }
  // Check decreasing resolution at higher levels
  V3i prevSize = base->extents().size();
  for (size_t i = 1; i < fields.size(); i++) {
    V3i size = fields[i]->extents().size();
    if (size.x > prevSize.x || 
        size.y > prevSize.y ||
        size.z > prevSize.z) {
      throw MIPSparseFieldException("Field " + 
                                   lexical_cast<string>(i) + 
                                   " had greater resolution than previous"
                                   " level");
    }
    if (size.x >= prevSize.x &&
        size.y >= prevSize.y &&
        size.z >= prevSize.z) {
      throw MIPSparseFieldException("Field " + 
                                   lexical_cast<string>(i) + 
                                   " did not decrease in resolution from "
                                   " previous level");
    }
    prevSize = size;
  }
  // All good.
}

//----------------------------------------------------------------------------//
// Static data member instantiation
//----------------------------------------------------------------------------//

FIELD3D_CLASSTYPE_TEMPL_INSTANTIATION(MIPSparseField);

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
