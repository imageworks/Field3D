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

/*! \file MIPDenseField.h
  \brief Contains the MIPDenseField class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_MIPDenseField_H_
#define _INCLUDED_MIPDenseField_H_

#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/thread/mutex.hpp>

#include "DenseField.h"
#include "EmptyField.h"
#include "MIPField.h"
#include "MIPInterp.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Exceptions
//----------------------------------------------------------------------------//

namespace Exc {

DECLARE_FIELD3D_GENERIC_EXCEPTION(MIPDenseFieldException, Exception)

} // namespace Exc

//----------------------------------------------------------------------------//
// Forward declarations 
//----------------------------------------------------------------------------//

template <class T>
class CubicMIPDenseFieldInterp; 

//----------------------------------------------------------------------------//
// MIPDenseField
//----------------------------------------------------------------------------//

/*! \class MIPDenseField
  \ingroup field
  \brief This subclass stores a MIP representation of DenseField.

  Each level in the MIPDenseField is stored as a DenseField, and each level
  shares the same FieldMapping definition, even though their resolution is
  different.

  The class is lazy loading, such that no MIP level are read from disk until
  they are needed. There is, however, no caching, and once a MIP level is read
  from disk it stays in memory until the entire MIPDenseField is destroyed.

  The class is thread safe, and ensures that data is read from disk from one
  single thread, using the double-checked locking mechanism.

  Interpolation into a MIP field may be done either directly to a single level,
  or by blending between two MIP levels. When blending, each field is assumed
  to match the other levels only in local-space.
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class MIPDenseField
  : public MIPField<Data_T>
{
public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<MIPDenseField> Ptr;
  typedef std::vector<Ptr> Vec;

  typedef MIPLinearInterp<MIPDenseField<Data_T> > LinearInterp;
  typedef CubicMIPDenseFieldInterp<Data_T>        CubicInterp;

  typedef DenseField<Data_T> ContainedType;

  typedef DenseField<Data_T> DenseType;
  typedef typename DenseType::Ptr DensePtr;
  typedef std::vector<DensePtr> DenseVec;

  typedef EmptyField<Data_T> ProxyField;
  typedef typename ProxyField::Ptr ProxyPtr;
  typedef std::vector<ProxyPtr> ProxyVec;  

  // Constructors --------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  //! Constructs an empty MIP field
  MIPDenseField();

  //! Copy constructor. We need this because a) we own a mutex and b) we own
  //! shared pointers and shallow copies are not good enough.
  MIPDenseField(const MIPDenseField &other);

  //! Assignment operator
  const MIPDenseField& operator = (const MIPDenseField &rhs);

  // \}

  // From FieldRes base class --------------------------------------------------

  //! \name From FieldRes
  //! \{  
  //! Returns -current- memory use, rather than the amount used if all
  //! levels were loaded.
  virtual long long int memSize() const;
  //! We need to know if the mapping changed so that we may update the
  //! MIP levels' mappings
  virtual void mappingChanged();
  //! \}
  
  // From Field base class -----------------------------------------------------

  //! \name From Field
  //! \{  
  //! For a MIP field, the common value() call accesses data at level 0 only.
  virtual Data_T value(int i, int j, int k) const;
  //! \}

  // RTTI replacement ----------------------------------------------------------

  typedef MIPDenseField<Data_T> class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS

  static const char *staticClassName()
  {
    return "MIPDenseField";
  }

  static const char *classType()
  {
    return MIPDenseField<Data_T>::ms_classType.name();
  }
    
  // From MIPField -------------------------------------------------------------

  virtual Data_T mipValue(size_t level, int i, int j, int k) const;
  virtual V3i    mipResolution(size_t level) const;
  virtual bool   levelLoaded(const size_t level) const;
  virtual void   getVsMIPCoord(const V3f &vsP, const size_t level, 
                               V3f &outVsP) const;

  // Concrete voxel access -----------------------------------------------------

  //! Read access to voxel at a given MIP level. 
  const Data_T& fastMipValue(size_t level, int i, int j, int k) const;

  // From FieldBase ------------------------------------------------------------

  //! \name From FieldBase
  //! \{

  virtual std::string className() const
  { return staticClassName(); }
  
  virtual FieldBase::Ptr clone() const
  { 
    return Ptr(new MIPDenseField(*this));
  }

  //! \}

  // Main methods --------------------------------------------------------------

  //! Clears all the levels of the MIP field.
  void clear();
  //! Sets up the MIP field given a set of DenseFields
  //! This call performs sanity checking to ensure that MIP properties are
  //! satisfied for each level.
  //! In this case, all MIP levels are available in memory.
  //! \note The MIP level order is implied to be zero-first.
  void setup(const DenseVec &fields);
  //! Sets up the MIP field in lazy-load mode
  //! \param mipGroupPath Path in F3D file to read data from.
  void setupLazyLoad(const ProxyVec &proxies,
                     const typename LazyLoadAction<DenseType>::Vec &actions);

  //! Returns a pointer to a MIP level
  DensePtr mipLevel(const size_t level) const;
  //! Returns a pointer to a MIP level
  const DenseType* rawMipLevel(const size_t level) const;

protected:

  // Typedefs ------------------------------------------------------------------

  typedef MIPField<Data_T> base;

  // Static data members -------------------------------------------------------

  static TemplatedFieldType<MIPDenseField<Data_T> > ms_classType;

  // Data members --------------------------------------------------------------

  //! Storage of all MIP levels. Some or all of the pointers may be NULL.
  //! This is mutable because it needs updating during lazy loading of data.
  mutable std::vector<DensePtr> m_fields;
  //! Lazy load actions. Only used if setupLazyLoad() has been called.
  mutable typename LazyLoadAction<DenseType>::Vec m_loadActions;
  //! Raw pointers to MIP levels.
  //! \note Important note: This is also used as the double-checked locking
  //! indicator.
  mutable std::vector<DenseField<Data_T>*> m_rawFields;
  //! Resolution of each MIP level.
  mutable std::vector<V3i> m_mipRes;
  //! Relative resolution of each MIP level. Pre-computed to avoid
  //! int-to-float conversions
  mutable std::vector<V3f> m_relativeResolution;
  //! Mutex lock around IO. Used to make sure only one thread reads MIP 
  //! level data at a time. When a field is cloned, the two new fields
  //! will share the mutex, since they point to the same file.
  boost::shared_ptr<boost::mutex> m_ioMutex;

  // Utility methods -----------------------------------------------------------

  //! Copies from a second MIPDenseField
  const MIPDenseField& init(const MIPDenseField &rhs);
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

typedef MIPDenseField<half>   MIPDenseFieldh;
typedef MIPDenseField<float>  MIPDenseFieldf;
typedef MIPDenseField<double> MIPDenseFieldd;
typedef MIPDenseField<V3h>    MIPDenseField3h;
typedef MIPDenseField<V3f>    MIPDenseField3f;
typedef MIPDenseField<V3d>    MIPDenseField3d;

//----------------------------------------------------------------------------//
// MIPDenseField implementations
//----------------------------------------------------------------------------//

template <class Data_T>
MIPDenseField<Data_T>::MIPDenseField()
  : base(),
    m_ioMutex(new boost::mutex)
{
  m_fields.resize(base::m_numLevels);
}

//----------------------------------------------------------------------------//

template <class Data_T>
MIPDenseField<Data_T>::MIPDenseField(const MIPDenseField &other)
  : base(other)
{
  init(other);
}

//----------------------------------------------------------------------------//

template <class Data_T>
const MIPDenseField<Data_T>& 
MIPDenseField<Data_T>::operator = (const MIPDenseField &rhs)
{
  base::operator=(rhs);
  return init(rhs);
}

//----------------------------------------------------------------------------//

template <class Data_T>
const MIPDenseField<Data_T>& 
MIPDenseField<Data_T>::init(const MIPDenseField &rhs)
{
  // If any of the fields aren't yet loaded, we can rely on the same load 
  // actions as the other one
  m_loadActions = rhs.m_loadActions;
  // Copy all the regular data members
  m_mipRes = rhs.m_mipRes;
  m_relativeResolution = rhs.m_relativeResolution;
  // The contained fields must be individually cloned if they have already
  // been loaded
  m_fields.resize(rhs.m_fields.size());
  m_rawFields.resize(rhs.m_rawFields.size());
  for (size_t i = 0, end = m_fields.size(); i < end; ++i) {
    // Update the field pointer
    if (rhs.m_fields[i]) {
      FieldBase::Ptr baseClone = rhs.m_fields[i]->clone();
      DensePtr clone = field_dynamic_cast<DenseType>(baseClone);
      if (clone) {
        m_fields[i] = clone;
      } else {
        std::cerr << "MIPDenseField::op=(): Failed to clone." << std::endl;
      }
    }
    // Update the raw pointer
    m_rawFields[i] = m_fields[i].get();
  }
  // New mutex
  m_ioMutex.reset(new boost::mutex);
  // Done
  return *this;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MIPDenseField<Data_T>::clear()
{
  m_fields.clear();
  m_rawFields.clear();
  base::m_numLevels = 0;
  base::m_lowestLevel = 0;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MIPDenseField<Data_T>::setup(const DenseVec &fields)
{
  // Clear existing data
  clear();
  // Run sanity checks. This will throw an exception if the fields are invalid.
  sanityChecks<DenseField<Data_T> >(fields);
  // Update state of object
  m_fields = fields;
  base::m_numLevels = fields.size();
  base::m_lowestLevel = 0;
  updateMapping(fields[0]);
  updateAuxMembers();
  // Resize vectors
  m_mipRes.resize(base::m_numLevels);
  m_relativeResolution.resize(base::m_numLevels);
  // For each MIP level
  for (size_t i = 0; i < fields.size(); i++) {
  // Update MIP res from real fields
    m_mipRes[i] = m_fields[i]->extents().size() + V3i(1);
    // Update relative resolutions
    m_relativeResolution[i] = V3f(m_mipRes[i]) / m_mipRes[0];
  } 
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MIPDenseField<Data_T>::setupLazyLoad
(const ProxyVec &proxies,
 const typename LazyLoadAction<DenseType>::Vec &actions)
{
  using namespace Exc;

  // Clear existing data
  clear();
  // Check same number of proxies and actions
  if (proxies.size() != actions.size()) {
    throw MIPDenseFieldException("Incorrect number of lazy load actions");
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
  // Resize vectors
  m_mipRes.resize(base::m_numLevels);
  m_relativeResolution.resize(base::m_numLevels);
  for (size_t i = 0; i < proxies.size(); i++) {
    // Update mip res from proxy fields
    m_mipRes[i] = proxies[i]->extents().size() + V3i(1);
    // Update relative resolutions
    m_relativeResolution[i] = V3f(m_mipRes[i]) / m_mipRes[0];
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MIPDenseField<Data_T>::DensePtr 
MIPDenseField<Data_T>::mipLevel(const size_t level) const
{
  assert(level < base::m_numLevels);
  // Ensure level is loaded.
  if (!m_rawFields[level]) {
    loadLevelFromDisk(level);
  } 
  return m_fields[level];
}

//----------------------------------------------------------------------------//

template <class Data_T>
const typename MIPDenseField<Data_T>::DenseType* 
MIPDenseField<Data_T>::rawMipLevel(const size_t level) const
{
  assert(level < base::m_numLevels);
  // Ensure level is loaded.
  if (!m_rawFields[level]) {
    loadLevelFromDisk(level);
  } 
  return m_rawFields[level];
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T MIPDenseField<Data_T>::value(int i, int j, int k) const
{
  return fastMipValue(0, i, j, k);
}

//----------------------------------------------------------------------------//

template <class Data_T>
long long int MIPDenseField<Data_T>::memSize() const
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
void MIPDenseField<Data_T>::mappingChanged() 
{ 
  for (size_t i = 0; i < m_fields.size(); i++) {
    if (m_fields[i]) {
      m_fields[i]->setMapping(base::mapping());
    }
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T MIPDenseField<Data_T>::mipValue(size_t level,
                                       int i, int j, int k) const
{
  return fastMipValue(level, i, j, k);
}

//----------------------------------------------------------------------------//

template <class Data_T>
V3i MIPDenseField<Data_T>::mipResolution(size_t level) const
{
  assert(level < base::m_numLevels);
  return m_mipRes[level];
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool MIPDenseField<Data_T>::levelLoaded(const size_t level) const
{
  assert(level < base::m_numLevels);
  return m_rawFields[level] != NULL;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void MIPDenseField<Data_T>::getVsMIPCoord(const V3f &vsP, const size_t level, 
                                          V3f &outVsP) const
{
  outVsP = vsP * m_relativeResolution[level];
}

//----------------------------------------------------------------------------//

template <class Data_T>
const Data_T& MIPDenseField<Data_T>::fastMipValue(size_t level,
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
void MIPDenseField<Data_T>::updateAuxMembers() const
{
  m_rawFields.resize(m_fields.size());
  for (size_t i = 0; i < m_fields.size(); i++) {
    m_rawFields[i] = m_fields[i].get();
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MIPDenseField<Data_T>::updateMapping(FieldRes::Ptr field)
{
  base::m_extents = field->extents();
  base::m_dataWindow = field->dataWindow();
  base::setMapping(field->mapping());
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MIPDenseField<Data_T>::loadLevelFromDisk(size_t level) const
{
  // Double-check locking
  if (!m_rawFields[level]) {
    boost::mutex::scoped_lock lock(*m_ioMutex);
    if (!m_rawFields[level]) {
      // Execute the lazy load action
      m_fields[level] = m_loadActions[level]->load();
      // Remove lazy load action
      m_loadActions[level].reset();
      // Update aux data
      updateAuxMembers();
      // Update the mapping of the loaded field
      m_fields[level]->setMapping(base::mapping());
    }
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
template <class Field_T>
void
MIPDenseField<Data_T>::
sanityChecks(const std::vector<typename Field_T::Ptr> &fields)
{
  using boost::lexical_cast;
  using std::string;
  using Exc::MIPDenseFieldException;

  // Check zero length
  if (fields.size() == 0) {
    throw MIPDenseFieldException("Zero fields in input");
  }
  // Check all non-null
  for (size_t i = 0; i < fields.size(); i++) {
    if (!fields[i]) {
      throw MIPDenseFieldException("Found null pointer in input");
    }
  }
  // Grab first field
  typename Field_T::Ptr base = fields[0];
  // Check common mapping
  FieldMapping::Ptr baseMapping = base->mapping();
  for (size_t i = 1; i < fields.size(); i++) {
#if 0 // This can't be true for different resolutions
    if (!baseMapping->isIdentical(fields[i]->mapping())) {
      throw MIPDenseFieldException("Field " + 
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
      throw MIPDenseFieldException("Field " + 
                                   lexical_cast<string>(i) + 
                                   " had greater resolution than previous"
                                   " level");
    }
    if (size.x >= prevSize.x &&
        size.y >= prevSize.y &&
        size.z >= prevSize.z) {
      throw MIPDenseFieldException("Field " + 
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

FIELD3D_CLASSTYPE_TEMPL_INSTANTIATION(MIPDenseField);

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
