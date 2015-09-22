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

/*! \file MIPField.h
  \brief Contains the MIPField class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_MIPField_H_
#define _INCLUDED_MIPField_H_

#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/thread/mutex.hpp>

#include "EmptyField.h"
#include "MIPBase.h"
#include "MIPInterp.h"
#include "MIPUtil.h"
#include "DenseField.h"
#include "SparseField.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Exceptions
//----------------------------------------------------------------------------//

namespace Exc {

DECLARE_FIELD3D_GENERIC_EXCEPTION(MIPFieldException, Exception)

} // namespace Exc

//----------------------------------------------------------------------------//
// Forward declarations 
//----------------------------------------------------------------------------//

template <class T>
class CubicMIPFieldInterp; 

//----------------------------------------------------------------------------//
// MIPField
//----------------------------------------------------------------------------//

/*! \class MIPField
  \ingroup field
  \brief This subclass stores a MIP representation of a Field_T field

  Each level in the MIPField is stored as a SparseField, and each level
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

template <class Field_T>
class MIPField : public MIPBase<typename Field_T::value_type>
{
public:

  // Typedefs ------------------------------------------------------------------
  
  typedef typename Field_T::value_type        Data_T;
  typedef Field_T                             NestedType;

  typedef boost::intrusive_ptr<MIPField>      Ptr;
  typedef std::vector<Ptr>                    Vec;

  typedef MIPLinearInterp<MIPField<Field_T> > LinearInterp;
  typedef CubicMIPFieldInterp<Data_T>         CubicInterp;

  typedef Data_T                              value_type;

  typedef EmptyField<Data_T>                  ProxyField;
  typedef typename ProxyField::Ptr            ProxyPtr;
  typedef std::vector<ProxyPtr>               ProxyVec;

  typedef typename Field_T::Ptr               FieldPtr;
  typedef std::vector<FieldPtr>               FieldVec;

  // Constructors --------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  //! Constructs an empty MIP field
  MIPField();

  //! Copy constructor. We need this because a) we own a mutex and b) we own
  //! shared pointers and shallow copies are not good enough.
  MIPField(const MIPField &other);

  //! Assignment operator
  const MIPField& operator = (const MIPField &rhs);

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
  virtual size_t voxelCount() const;
  //! \}

  // RTTI replacement ----------------------------------------------------------

  typedef MIPField<Field_T> class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS

  static const char *staticClassName()
  {
    return "MIPField";
  }

  static const char *staticClassType()
  {
    return MIPField<Field_T>::ms_classType.name();
  }
    
  // From MIPField -------------------------------------------------------------

  virtual Data_T mipValue(size_t level, int i, int j, int k) const;
  virtual V3i    mipResolution(size_t level) const;
  virtual bool   levelLoaded(const size_t level) const;
  virtual void   getVsMIPCoord(const V3f &vsP, const size_t level, 
                               V3f &outVsP) const;
  virtual typename Field<Data_T>::Ptr mipLevel(const size_t level) const;

  // Concrete voxel access -----------------------------------------------------

  //! Read access to voxel at a given MIP level. 
  Data_T fastMipValue(size_t level, int i, int j, int k) const;

  // From FieldBase ------------------------------------------------------------

  //! \name From FieldBase
  //! \{

  FIELD3D_CLASSNAME_CLASSTYPE_IMPLEMENTATION;
  
  virtual FieldBase::Ptr clone() const
  { 
    return Ptr(new MIPField(*this));
  }

  //! \}

  // Main methods --------------------------------------------------------------

  //! Clears all the levels of the MIP field.
  void clear();
  //! Sets up the MIP field given a set of non-MIP fields
  //! This call performs sanity checking to ensure that MIP properties are
  //! satisfied for each level.
  //! In this case, all MIP levels are available in memory.
  //! \note The MIP level order is implied to be zero-first.
  void setup(const FieldVec &fields);
  //! Sets up the MIP field in lazy-load mode
  //! \param mipGroupPath Path in F3D file to read data from.
  void setupLazyLoad(const ProxyVec &proxies,
                     const typename LazyLoadAction<Field_T>::Vec &actions);

#if 0
  //! Returns a pointer to a MIP level
  FieldPtr mipLevel(const size_t level) const;
#endif
  //! Returns a raw pointer to a MIP level
  const Field_T* rawMipLevel(const size_t level) const;
  //! Returns a concretely typed pointer to a MIP level
  typename Field_T::Ptr concreteMipLevel(const size_t level) const;

protected:

  // Typedefs ------------------------------------------------------------------

  typedef MIPBase<Data_T> base;

  // Static data members -------------------------------------------------------

  static NestedFieldType<MIPField<Field_T> > ms_classType;

  // Preventing instantiation of this helper class -----------------------------

  

  // Data members --------------------------------------------------------------

  //! Storage of all MIP levels. Some or all of the pointers may be NULL.
  //! This is mutable because it needs updating during lazy loading of data.
  mutable std::vector<FieldPtr> m_fields;
  //! Lazy load actions. Only used if setupLazyLoad() has been called.
  mutable typename LazyLoadAction<Field_T>::Vec m_loadActions;
  //! Raw pointers to MIP levels.
  //! \note Important note: This is also used as the double-checked locking
  //! indicator.
  mutable std::vector<Field_T*> m_rawFields;
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

  //! Copies from a second MIPField
  const MIPField& init(const MIPField &rhs);
  //! Updates the mapping, extents and data window to match the given field.
  //! Used so that the MIPField will appear to have the same mapping in space
  //! as the level-0 field.
  void updateMapping(FieldRes::Ptr field);
  //! Updates the dependent data members based on m_field
  void updateAuxMembers() const;
  //! Updates the name, attribute and metadata for a given level
  void syncLevelInfo(const size_t level) const;
  //! Loads the given level from disk
  void loadLevelFromDisk(size_t level) const;
  //! Sanity checks to ensure that the provided Fields are a MIP representation
  template <typename T>
  void sanityChecks(const T &fields);

};

//----------------------------------------------------------------------------//
// Helper classes
//----------------------------------------------------------------------------//

template <typename Data_T>
class MIPSparseField : public MIPField<SparseField<Data_T> >
{
public:
    virtual FieldBase::Ptr clone() const
  { 
    return FieldBase::Ptr(new MIPSparseField(*this));
  }
};

//----------------------------------------------------------------------------//

template <typename Data_T>
class MIPDenseField : public MIPField<DenseField<Data_T> >
{
  public:
    virtual FieldBase::Ptr clone() const
  { 
    return FieldBase::Ptr(new MIPDenseField(*this));
  }
};

//----------------------------------------------------------------------------//
// MIPField implementations
//----------------------------------------------------------------------------//

template <class Field_T>
MIPField<Field_T>::MIPField()
  : base(),
    m_ioMutex(new boost::mutex)
{
  m_fields.resize(base::m_numLevels);
}

//----------------------------------------------------------------------------//

template <class Field_T>
MIPField<Field_T>::MIPField(const MIPField &other)
  : base(other)
{
  init(other);
}

//----------------------------------------------------------------------------//

template <class Field_T>
const MIPField<Field_T>& 
MIPField<Field_T>::operator = (const MIPField &rhs)
{
  base::operator=(rhs);
  return init(rhs);
}

//----------------------------------------------------------------------------//

template <class Field_T>
const MIPField<Field_T>& 
MIPField<Field_T>::init(const MIPField &rhs)
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
      FieldPtr clone = field_dynamic_cast<Field_T>(baseClone);
      if (clone) {
        m_fields[i] = clone;
      } else {
        std::cerr << "MIPField::op=(): Failed to clone." << std::endl;
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

template <class Field_T>
void MIPField<Field_T>::clear()
{
  m_fields.clear();
  m_rawFields.clear();
  base::m_numLevels = 0;
  base::m_lowestLevel = 0;
}

//----------------------------------------------------------------------------//

template <class Field_T>
void MIPField<Field_T>::setup(const FieldVec &fields)
{
  // Clear existing data
  clear();
  // Run sanity checks. This will throw an exception if the fields are invalid.
  sanityChecks(fields);
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

template <class Field_T>
void MIPField<Field_T>::setupLazyLoad
(const ProxyVec &proxies,
 const typename LazyLoadAction<Field_T>::Vec &actions)
{
  using namespace Exc;

  // Clear existing data
  clear();
  // Check same number of proxies and actions
  if (proxies.size() != actions.size()) {
    throw MIPFieldException("Incorrect number of lazy load actions");
  }  
  // Run sanity checks. This will throw an exception if the fields are invalid.
  sanityChecks(proxies);
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

#if 0

template <class Field_T>
typename MIPField<Field_T>::FieldPtr 
MIPField<Field_T>::mipLevel(size_t level) const
{
  assert(level < base::m_numLevels);
  // Ensure level is loaded.
  if (!m_rawFields[level]) {
    loadLevelFromDisk(level);
  } 
  return m_fields[level];
}
#endif

//----------------------------------------------------------------------------//

template <class Field_T>
const Field_T* 
MIPField<Field_T>::rawMipLevel(size_t level) const
{
  assert(level < base::m_numLevels);
  // Ensure level is loaded.
  if (!m_rawFields[level]) {
    loadLevelFromDisk(level);
  } 
  // Return 
  return m_rawFields[level];
}

//----------------------------------------------------------------------------//

template <class Field_T>
typename Field_T::Ptr
MIPField<Field_T>::concreteMipLevel(size_t level) const
{
  assert(level < base::m_numLevels);
  // Ensure level is loaded.
  if (!m_rawFields[level]) {
    loadLevelFromDisk(level);
  } 
  // Return 
  return m_fields[level];
}

//----------------------------------------------------------------------------//

template <class Field_T>
typename MIPField<Field_T>::Data_T 
MIPField<Field_T>::value(int i, int j, int k) const
{
  return fastMipValue(0, i, j, k);
}

//----------------------------------------------------------------------------//

template <class Field_T>
size_t
MIPField<Field_T>::voxelCount() const
{
  size_t count = 0;
  for (size_t i = 0; i < m_fields.size(); i++) {
    if (m_fields[i]) {
      count += m_fields[i]->voxelCount();
    }
  }
  return count;
}

//----------------------------------------------------------------------------//

template <class Field_T>
long long int MIPField<Field_T>::memSize() const
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

template <class Field_T>
void MIPField<Field_T>::mappingChanged() 
{ 
  // Update MIP offset
  const V3i offset = 
    base::metadata().vecIntMetadata(detail::k_mipOffsetStr, V3i(0));
  base::setMIPOffset(offset);

  V3i baseRes = base::dataWindow().size() + V3i(1);
  if (m_fields[0]) {
    m_fields[0]->setMapping(base::mapping());
  }
  for (size_t i = 1; i < m_fields.size(); i++) {
    if (m_fields[i]) {
      FieldMapping::Ptr mapping = 
        detail::adjustedMIPFieldMapping(this, baseRes,
                                        m_fields[i]->extents(), i);
      m_fields[i]->setMapping(mapping);
    }
  }
}

//----------------------------------------------------------------------------//

template <class Field_T>
typename MIPField<Field_T>::Data_T 
MIPField<Field_T>::mipValue(size_t level, int i, int j, int k) const
{
  return fastMipValue(level, i, j, k);
}

//----------------------------------------------------------------------------//

template <class Field_T>
V3i MIPField<Field_T>::mipResolution(size_t level) const
{
  assert(level < base::m_numLevels);
  return m_mipRes[level];
}

//----------------------------------------------------------------------------//

template <class Field_T>
bool MIPField<Field_T>::levelLoaded(const size_t level) const
{
  assert(level < base::m_numLevels);
  return m_rawFields[level] != NULL;
}

//----------------------------------------------------------------------------//

template <typename Field_T>
void MIPField<Field_T>::getVsMIPCoord(const V3f &vsP, const size_t level, 
                                      V3f &outVsP) const
{
  const V3i &mipOff = base::mipOffset();

  // Compute offset of current level 
  const V3i offset((mipOff.x >> level) << level, 
                   (mipOff.y >> level) << level, 
                   (mipOff.z >> level) << level);

  // Difference between current offset and base offset is num voxels
  // to offset current level by 
  const V3f diff = offset - mipOff;

  // Incorporate shift due to mip offset
  outVsP = (vsP - diff) * pow(2.0, -static_cast<float>(level));
}

//----------------------------------------------------------------------------//

template <typename Field_T>
typename Field<typename MIPField<Field_T>::Data_T>::Ptr 
MIPField<Field_T>::mipLevel(const size_t level) const
{
  assert(level < base::m_numLevels);
  // Ensure level is loaded.
  if (!m_rawFields[level]) {
    loadLevelFromDisk(level);
  } 
  // Return 
  return m_fields[level];
}

//----------------------------------------------------------------------------//

template <class Field_T>
typename MIPField<Field_T>::Data_T 
MIPField<Field_T>::fastMipValue(size_t level, int i, int j, int k) const
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

template <class Field_T>
void MIPField<Field_T>::updateAuxMembers() const
{
  m_rawFields.resize(m_fields.size());
  for (size_t i = 0; i < m_fields.size(); i++) {
    m_rawFields[i] = m_fields[i].get();
  }
}

//----------------------------------------------------------------------------//

template <class Field_T>
void MIPField<Field_T>::syncLevelInfo(const size_t level) const
{
  // At this point, m_fields[level] is guaranteed in memory
  
  // First sync name, attribute
  m_fields[level]->name      = base::name;
  m_fields[level]->attribute = base::attribute;
  // Copy metadata
  m_fields[level]->copyMetadata(*this);
}

//----------------------------------------------------------------------------//

template <class Field_T>
void MIPField<Field_T>::updateMapping(FieldRes::Ptr field)
{
  base::m_extents = field->extents();
  base::m_dataWindow = field->dataWindow();
  base::setMapping(field->mapping());
}

//----------------------------------------------------------------------------//

template <class Field_T>
void MIPField<Field_T>::loadLevelFromDisk(size_t level) const
{
  // Double-check locking
  if (!m_rawFields[level]) {
    boost::mutex::scoped_lock lock(*m_ioMutex);
    if (!m_rawFields[level]) {
      // Execute the lazy load action
      m_fields[level] = m_loadActions[level]->load();
      // Check that field was loaded
      if (!m_fields[level]) {
        throw Exc::MIPFieldException("Couldn't load MIP level: " + 
                                     boost::lexical_cast<std::string>(level));
      }
      // Remove lazy load action
      m_loadActions[level].reset();
      // Update aux data
      updateAuxMembers();
      // Ensure metadata is up to date
      syncLevelInfo(level);
      // Update the mapping of the loaded field
      V3i baseRes = base::dataWindow().size() + V3i(1);
      FieldMapping::Ptr mapping = 
        detail::adjustedMIPFieldMapping(this, baseRes, 
                                        m_fields[level]->extents(), level);
      m_fields[level]->setMapping(mapping);
    }
  }
}

//----------------------------------------------------------------------------//

template <class Field_T>
template <class T>
void
MIPField<Field_T>::sanityChecks(const T &fields)
{
  using boost::lexical_cast;
  using std::string;
  using Exc::MIPFieldException;

  // Check zero length
  if (fields.size() == 0) {
    throw MIPFieldException("Zero fields in input");
  }
  // Check all non-null
  for (size_t i = 0; i < fields.size(); i++) {
    if (!fields[i]) {
      throw MIPFieldException("Found null pointer in input");
    }
  }
  // Check decreasing resolution at higher levels
  V3i prevSize = fields[0]->extents().size();
  for (size_t i = 1; i < fields.size(); i++) {
    V3i size = fields[i]->extents().size();
    if (size.x > prevSize.x || 
        size.y > prevSize.y ||
        size.z > prevSize.z) {
      throw MIPFieldException("Field " + lexical_cast<string>(i) + 
                              " had greater resolution than previous"
                              " level");
    }
    if (size.x >= prevSize.x &&
        size.y >= prevSize.y &&
        size.z >= prevSize.z) {
      throw MIPFieldException("Field " + lexical_cast<string>(i) + 
                              " did not decrease in resolution from "
                              " previous level: " + 
                              lexical_cast<string>(size) + " > " + 
                              lexical_cast<string>(prevSize));
    }
    prevSize = size;
  }
  // All good.
}

//----------------------------------------------------------------------------//
// Static data member instantiation
//----------------------------------------------------------------------------//

template <typename Field_T>
NestedFieldType<MIPField<Field_T> > MIPField<Field_T>::ms_classType =
  NestedFieldType<MIPField<Field_T> >();

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
