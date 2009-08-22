//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks
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

/*! \file Field.h
  \brief Contains Field, WritableField and ResizableField classes.
  \ingroup field
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_Field_H_
#define _INCLUDED_Field3D_Field_H_

#include <cmath>
#include <vector>
#include <map>

#include <boost/intrusive_ptr.hpp> 
#include <boost/thread/mutex.hpp>

#include "DataTypeConversion.h"
#include "Exception.h"
#include "FieldMapping.h"
#include "Log.h"
#include "Types.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Field RTTI Replacement
//----------------------------------------------------------------------------//

#define DEFINE_CHECK_RTTI_CALL \
  virtual bool checkRTTI(const std::string &typenameStr) \
  { return matchRTTI(typenameStr); } \

#define DEFINE_MATCH_RTTI_CALL \
  bool matchRTTI(const std::string &typenameStr) \
  { \
    if (typenameStr == typeid(class_type).name()) { \
      return true; \
    } \
    return base::matchRTTI(typenameStr); \
  }

#define DEFINE_FIELD_RTTI_CONCRETE_CLASS \
  DEFINE_CHECK_RTTI_CALL \
  DEFINE_MATCH_RTTI_CALL

#define DEFINE_FIELD_RTTI_ABSTRACT_CLASS \
  DEFINE_MATCH_RTTI_CALL

//----------------------------------------------------------------------------//
// Exceptions
//----------------------------------------------------------------------------//

namespace Exc {

DECLARE_FIELD3D_GENERIC_EXCEPTION(MemoryException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(ResizeException, Exception)

} // namespace Exc

//----------------------------------------------------------------------------//
// FieldTraits
//----------------------------------------------------------------------------//

/*! \class FieldTraits
  \ingroup template_util
  Traits class that lets us answer how many components are in a given data type
*/

template <class Data_T>
class FieldTraits
{
 public:
  //! Dimensions of the given data type. i.e. 3 for V3f, 1 for float
  static int dataDims();
};

//----------------------------------------------------------------------------//
// FieldBase
//----------------------------------------------------------------------------//

/*! \class FieldBase
  \ingroup field
  This class provides a common base for all Field objects. It serves the 
  purpose of providing the className() virtual function and as a container
  for the metadata map
*/

class FieldBase
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::intrusive_ptr<FieldBase> Ptr;
  typedef std::map<std::string, std::string> StrMetadata;
  typedef std::map<std::string, int> IntMetadata;
  typedef std::map<std::string, float> FloatMetadata;
  typedef std::map<std::string, V3i> VecIntMetadata;
  typedef std::map<std::string, V3f> VecFloatMetadata;

  typedef FieldBase class_type;

  // Constructors --------------------------------------------------------------

  //! \name Constructors, destructors, copying
  //! \{

  //! Constructor
  FieldBase();

  //! Copy constructor
  FieldBase(const FieldBase &src);

  //! Assignment op
  FieldBase & operator = (const FieldBase &src);

  //! Destructor
  virtual ~FieldBase();

  //! \}

  // To be implemented by subclasses -------------------------------------------

  //! \name To be implemented by subclasses
  //! \{

  //! Returns the class name of the object. Used by the class pool and when
  //! writing the data to disk.
  virtual std::string className() const = 0;

  //! Returns a pointer to a copy of the field, pure virtual so ensure
  //! derived classes properly implement it
  virtual Ptr clone() const = 0;

  //! \}
  
  // RTTI replacement ----------------------------------------------------------

  /*! \note A note on why the RTTI replacement is needed:
     RTTI calls fail once the object crosses the dso boundary. We revert
     to using simple string checks which is more expensive but at least works
     once the dso is used in Houdini, etc.
     Use field_dynamic_cast<> for any FieldBase subclass instead of 
     dynamic_cast<>.
  */

  //! \name RTTI replacement
  //! \{

  //! This function is only implemented by concrete classes and triggers
  //! the actual RTTI check through matchRTTI();
  virtual bool checkRTTI(const std::string &typenameStr) = 0;
  
  //! Performs a check to see if the given typename string matches this class'
  //! This needs to be implemented in -all- subclasses, even abstract ones.
  bool matchRTTI(const std::string &typenameStr)
  { return typenameStr == typeid(*this).name(); }

  //! \}

  // Reference counting --------------------------------------------------------

  //! \name Reference counting
  //! \{

  //! Used by boost::intrusive_pointer
  size_t refcnt() 
  { return m_counter; }

  //! Used by boost::intrusive_pointer
  void ref() 
  {         
    boost::mutex::scoped_lock lock(m_refMutex);
    m_counter++; 
  }

  //! Used by boost::intrusive_pointer
  void unref() 
  { 
    boost::mutex::scoped_lock lock(m_refMutex);
    m_counter--; 
  }

  //! \}

  // Access to metadata --------------------------------------------------------

  //! \name Metadata
  //! \{

  //! Tries to retrieve a V3f metadata value. Returns the specified
  //! default value if no metadata was found. 
  V3f vecFloatMetadata(const std::string &name, const V3f &defaultVal) const;

  //! Tries to retrieve a float metadata value. Returns the specified
  //! default value if no metadata was found. 
  float floatMetadata(const std::string &name, const float defaultVal) const;

  //! Tries to retrieve a V3i metadata value. Returns the specified
  //! default value if no metadata was found. 
  V3i vecIntMetadata(const std::string &name, const V3i &defaultVal) const;

  //! Tries to retrieve an int metadata value. Returns the specified
  //! default value if no metadata was found. 
  int intMetadata(const std::string &name, const int defaultVal) const;

  //! Tries to retrieve a string metadata value. Returns the specified
  //! default value if no metadata was found. 
  std::string strMetadata(const std::string &name, 
                          const std::string &defaultVal) const;

  //! Read only access to the m_vecFloatMetadata dictionary
  const VecFloatMetadata& vecFloatMetadata() 
  { return m_vecFloatMetadata; }
    
  //! Read only access to the m_floatMetadata dictionary
  const FloatMetadata& floatMetadata() 
  { return m_floatMetadata; }

  //! Read only access to the m_vecIntMetadata dictionary
  const VecIntMetadata& vecIntMetadata() 
  { return m_vecIntMetadata; }

  //! Read only access to the m_intMetadata dictionary
  const IntMetadata& intMetadata() 
  { return m_intMetadata; }

  //! Read only access to the m_strMetadata dictionary
  const StrMetadata& strMetadata() 
  { return m_strMetadata; }

  //! Set the a V3f value for the given metadata name.
  void setVecFloatMetadata(const std::string &name, const V3f &val);
    
  //! Set the a float value for the given metadata name.
  void setFloatMetadata(const std::string &name, const float val);

  //! Set the a V3i value for the given metadata name.
  void setVecIntMetadata(const std::string &name, const V3i &val);

  //! Set the a int value for the given metadata name.
  void setIntMetadata(const std::string &name, const int val);

  //! Set the a string value for the given metadata name.
  void setStrMetadata(const std::string &name, const std::string &val); 

  //! This function should implemented by concrete classes to  
  //! get the callback when metadata changes
  virtual void metadataHasChanged(const std::string &name) 
  { /* Empty */ }

  //! Copies the metadata from a second field
  void copyMetadata(const FieldBase &field);

  //! \}

  // Public data members -------------------------------------------------------

  //! Optional name of the field
  std::string name;
  //! Optional name of the attribute the field represents
  std::string attribute;

 private:

  // Private data members ------------------------------------------------------

  //! V3f metadata
  VecFloatMetadata m_vecFloatMetadata;
  //! Float metadata
  FloatMetadata m_floatMetadata;
  //! V3i metadata
  VecIntMetadata m_vecIntMetadata;
  //! Int metadata
  IntMetadata m_intMetadata;
  //! String metadata
  StrMetadata m_strMetadata;

  //! For boost intrusive pointer
  mutable int m_counter;
  //! Mutex for ref counting
  mutable boost::mutex m_refMutex;     
 
};

//----------------------------------------------------------------------------//
// Intrusive Pointer reference counting 
//----------------------------------------------------------------------------//

inline void 
intrusive_ptr_add_ref(FieldBase* r)
{
  r->ref();
}

//----------------------------------------------------------------------------//

inline void
intrusive_ptr_release(FieldBase* r)
{
  r->unref();

  if (r->refcnt() == 0)
    delete r;
}

//----------------------------------------------------------------------------//
// FieldRes
//----------------------------------------------------------------------------//

/*! \class FieldRes
  \ingroup field
  This class serves to isolate the extents and data window from its
  templated subclasses. Thus, anything that needs to access the extents or
  data window don't need to know about what data type the subclass is
  templated on.

  It also owns the field's mapping.

  Why do we have both an extent and a data window? The extents are used
  to define which range of voxels define the [0..1] local coordinate system.
  The data window in turn defines the voxels that are legal to read/write
  from. Thus, for optimization we may have a large extents but a small data
  window, or a small extents and a larger data window which would let us
  apply large-kernel filters without having to deal with boundary conditions.
*/

class FieldRes : public FieldBase
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::intrusive_ptr<FieldRes> Ptr;
  typedef std::vector<Ptr> Vec;

  virtual std::string dataTypeString() const = 0;

  // RTTI replacement ----------------------------------------------------------

  typedef FieldRes class_type;
  DEFINE_FIELD_RTTI_ABSTRACT_CLASS;
  
  // Ctor, dtor ----------------------------------------------------------------

  //! This constructor ensures that we have a valid mapping at all times
  FieldRes();

  //! Base class copy constructor
  //! \todo OSS Go over the copy constructing - is it implemented right? 8hr
  FieldRes(const FieldRes &src);

  // Main methods --------------------------------------------------------------

  //! Returns the extents of the data. This signifies the relevant area that
  //! the data exists over. However, the data window (below) may be smaller
  //! than the extents, in which case it is only safe to call value() for
  //! those coordinate inside the data window.
  inline const Box3i& extents() const
  { return m_extents; }
  //! Returns the data window. Any coordinate inside this window is safe to
  //! pass to value() in the Field subclass.
  inline const Box3i& dataWindow() const
  { return m_dataWindow; }

  inline V3i const dataResolution() const
  { return m_dataWindow.max - m_dataWindow.min + V3i(1); }

  //! Sets the field's mapping
  void setMapping(FieldMapping::Ptr mapping);

  //! Returns a pointer to the mapping
  FieldMapping::Ptr mapping()
  { return m_mapping; }

  //! Returns a pointer to the mapping
  const FieldMapping::Ptr mapping() const
  { return m_mapping; }

  //! Returns true is the indicies are in bounds of the data window
  bool isInBounds(int i, int j, int k);

  // To be implemented by subclasses -------------------------------------------

  //! Returns the memory usage (in bytes)
  //! \note This needs to be re-implemented for any subclass that adds data
  //! members. Those classes should also call their superclass and add the 
  //! combined memory use.
  virtual long long int memSize() const
  { return sizeof(*this); }

protected:

  // Typedefs ------------------------------------------------------------------

  typedef MatrixFieldMapping default_mapping;

  // Data members --------------------------------------------------------------

  //! Defines the extents of the the storage. This may be larger or smaller 
  //! than the data window, and in the case where it is larger, care must be
  //! taken not to access voxels outside the data window. This should be treated 
  //! as a closed (i.e. inclusive) interval.
  Box3i m_extents;
  //! Defines the area where data is allocated. This should be treated as a
  //! closed (i.e. inclusive) interval.
  Box3i m_dataWindow;
  //! Pointer to the field's mapping
  FieldMapping::Ptr m_mapping;
  
private:

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef FieldBase base;

};

//----------------------------------------------------------------------------//

inline FieldRes::FieldRes()
  : m_mapping(new default_mapping)
{ 
  m_extents = Box3i(V3i(0), V3i(-1));
  m_dataWindow = m_extents;
  m_mapping->setExtents(m_extents); 
}

//----------------------------------------------------------------------------//

inline FieldRes::FieldRes(const FieldRes &src) 
  : FieldBase(src)
{
  // Call base class first
  // FieldBase(src);
  // Copy self
  *this = src;
  m_mapping = src.mapping()->clone();
}

//----------------------------------------------------------------------------//

inline void FieldRes::setMapping(FieldMapping::Ptr mapping)
{ 
  if (mapping) {
    m_mapping = mapping->clone(); 
    m_mapping->setExtents(m_extents); 
  } else {
    Log::print(Log::SevWarning, 
               "Tried to call FieldRes::setMapping with null pointer");
  }
}

//----------------------------------------------------------------------------//

inline bool FieldRes::isInBounds(int i, int j, int k) 
{
  // Check bounds
  if (i < m_dataWindow.min.x || i > m_dataWindow.max.x ||
      j < m_dataWindow.min.y || j > m_dataWindow.max.y ||
      k < m_dataWindow.min.z || k > m_dataWindow.max.z) {
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------//
// field_dynamic_cast
//----------------------------------------------------------------------------//

//! Dynamic cast that uses string-comparison in order to be safe even
//! after an object crosses a shared library boundary.
//! \ingroup field
template <class Field_T>
typename Field_T::Ptr
field_dynamic_cast(FieldBase::Ptr field)
{
  if (!field) 
    return NULL;
  std::string tgtTypeString = typeid(Field_T).name();
  if (field->checkRTTI(tgtTypeString)) {
    return static_cast<Field_T*>(field.get());
  } else {
    return NULL;
  }
}

//----------------------------------------------------------------------------//
// Field
//----------------------------------------------------------------------------//

/*! \class Field
  \ingroup field
  This class provides read-only access to voxels. A read-only buffer can not be
  resized. Resizing is added by ResizableField. The object still has a 
  size of course, but it can only be set by subclass-specific methods.
  \note Regarding the template type Data_T. This does not necessarily have
  to be the internal data storage format, it only defines the -return type-
  that the particular Field instance provides.
*/

template <class Data_T>
class Field : public FieldRes
{
 public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<Field> Ptr;

  //! Allows us to reference the template class
  typedef Data_T value_type;

  //! This is a convenience typedef for the list that 
  //! Field3DInputFile::readScalarLayers() and 
  //! Field3DInputFile::readVectorLayers() will return its data in
  typedef std::vector<Ptr> Vec;

  // RTTI replacement ----------------------------------------------------------

  typedef Field<Data_T> class_type;
  DEFINE_FIELD_RTTI_ABSTRACT_CLASS;
  
  // Constructors --------------------------------------------------------------

  //! Dtor
  virtual ~Field()
  { /* Empty */ }

  // Iterators -----------------------------------------------------------------

  //! Const iterator for traversing the values in a Field object.
  //! \ingroup field
  class const_iterator;

  //! Const iterator to first element. "cbegin" matches the tr1 c++ standard.
  const_iterator cbegin() const;
  //! Const iterator to first element of specific subset
  const_iterator cbegin(const Box3i &subset) const;
  //! Const iterator pointing one element past the last valid one.
  const_iterator cend() const;
  //! Const iterator pointing one element past the last valid one (for a 
  //! subset)
  const_iterator cend(const Box3i &subset) const;

  // To be implemented by subclasses -------------------------------------------

  //! Read access to a voxel. The coordinates are in integer voxel space . 
  //! \note Before the internal storage is accessed, the subclass must compute 
  //! the data window coordinates by looking at Field::m_dataWindow.
  //! \note Virtual functions are known not to play nice with threading.
  //! Therefor, concrete classes can implement (by convention) fastValue()
  //! as a non-virtual function.
  virtual Data_T value(int i, int j, int k) const = 0;

  // Other member functions ----------------------------------------------------

  virtual std::string dataTypeString() const 
  { return dataTypeToString<Data_T>(); }

private:

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef FieldRes base;

};

//----------------------------------------------------------------------------//

template <class Data_T> 
typename Field<Data_T>::const_iterator 
Field<Data_T>::cbegin() const
{
  if (FieldRes::dataResolution() == V3i(0))
    return cend();
  return const_iterator(*this, m_dataWindow, m_dataWindow.min);
}

//----------------------------------------------------------------------------//

template <class Data_T> 
typename Field<Data_T>::const_iterator
Field<Data_T>::cbegin(const Box3i &subset) const
{
  if (subset.isEmpty())
    return cend(subset);
  return const_iterator(*this, subset, subset.min);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename Field<Data_T>::const_iterator 
Field<Data_T>::cend() const
{ 
  return const_iterator(*this, m_dataWindow, 
                        V3i(m_dataWindow.min.x, 
                            m_dataWindow.min.y,
                            m_dataWindow.max.z + 1));
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename Field<Data_T>::const_iterator 
Field<Data_T>::cend(const Box3i &subset) const
{ 
  return const_iterator(*this, subset, V3i(subset.min.x, 
                                           subset.min.y,
                                           subset.max.z + 1));
}

//----------------------------------------------------------------------------//
// Field::const_iterator
//----------------------------------------------------------------------------//

template <class Data_T>
class Field<Data_T>::const_iterator
{

public:

  // Constructors --------------------------------------------------------------

  const_iterator(const const_iterator &i) 
    :  x(i.x), y(i.y), z(i.z), 
       m_window(i.m_window), m_field(i.m_field) 
  { }

  const_iterator(const Field<Data_T> &field, const Box3i &window,
                 const V3i &currentPos)
    : x(currentPos.x), y(currentPos.y), z(currentPos.z), 
      m_window(window), m_field(field)
  { }

  // Operators -----------------------------------------------------------------

  inline const const_iterator& operator ++ ()
  {
    if (x == m_window.max.x) {
      if (y == m_window.max.y) {
        x = m_window.min.x;
        y = m_window.min.y;
        ++z;
      } else {
        x = m_window.min.x;
        ++y;
      }
    } else {
      ++x;
    }
    return *this;
  }
  template <class Iter_T>
  bool operator == (const Iter_T &rhs) const
  {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }
  template <class Iter_T>
  bool operator != (const Iter_T &rhs) const
  {
    return x != rhs.x || y != rhs.y || z != rhs.z;
  }
  inline Data_T operator * () const
  {
    return m_field.value(x, y, z);
  }
  // Public data members -------------------------------------------------------

  //! Current position
  int x, y, z;

private:

  // Private data members ------------------------------------------------------

  //! Window to traverse
  Box3i m_window;
  //! Reference to field being iterated over
  const Field<Data_T> &m_field;

};

//----------------------------------------------------------------------------//
// WritableField
//----------------------------------------------------------------------------//

/*! \class WritableField
  \ingroup field
  This class brings together both read- and write-access to voxels. The
  buffer can not be resized. Resizing is added by ResizableField.
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class WritableField 
  : public Field<Data_T>
{
public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<WritableField> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef WritableField<Data_T> class_type;
  DEFINE_FIELD_RTTI_ABSTRACT_CLASS;
  
  // Iterators -----------------------------------------------------------------

  //! Non-const iterator for traversing the values in a Field object.
  //! \ingroup field
  class iterator;

  //! Iterator to first element.
  inline iterator begin();
  //! Iterator to first element of specific subset
  inline iterator begin(const Box3i &subset);
  //! Iterator pointing one element past the last valid one.
  inline iterator end();
  //! Iterator pointing one element past the last valid one (for a 
  //! subset)
  inline iterator end(const Box3i &subset);

  // To be implemented by subclasses ------------------------------------------- 

  //! Write access to a voxel. The coordinates are global coordinates. 
  //! \note Before the internal storage is accessed, the subclass must compute 
  //! the crop window coordinates by looking at Field::m_dataWindow.
  //! \note This is named differently from the const value so that non-const
  //! objects still have a clear way of accessing data in a const way.
  //! \note Virtual functions are known not to play nice with threading.
  //! Therefor, concrete classes can implement (by convention) fastLValue()
  //! as a non-virtual function.
  virtual Data_T& lvalue(int i, int j, int k) = 0;

  // Main methods --------------------------------------------------------------

  //! Clears all the voxels in the storage. Should be re-implemented by
  //! subclasses that can provide a more efficient version.
  virtual void clear(const Data_T &value)
  { std::fill(begin(), end(), value); }

private:

  // Typedefs ------------------------------------------------------------------

  typedef Field<Data_T> base;

};

//----------------------------------------------------------------------------//

template <class Data_T>
inline typename WritableField<Data_T>::iterator 
WritableField<Data_T>::begin()
{
  if (FieldRes::dataResolution() == V3i(0))
    return end();
  return iterator(*this, Field<Data_T>::m_dataWindow, 
                  Field<Data_T>::m_dataWindow.min);
}

//----------------------------------------------------------------------------//

template <class Data_T>
inline typename WritableField<Data_T>::iterator 
WritableField<Data_T>::begin(const Box3i &subset)
{
  if (subset.isEmpty())
    return end(subset);
  return iterator(*this, subset, subset.min);
}

//----------------------------------------------------------------------------//

template <class Data_T>
inline typename WritableField<Data_T>::iterator 
WritableField<Data_T>::end()
{ return iterator(*this, Field<Data_T>::m_dataWindow, 
                  V3i(Field<Data_T>::m_dataWindow.min.x, 
                      Field<Data_T>::m_dataWindow.min.y,
                      Field<Data_T>::m_dataWindow.max.z + 1));
}

//----------------------------------------------------------------------------//

template <class Data_T>
inline typename WritableField<Data_T>::iterator 
WritableField<Data_T>::end(const Box3i &subset)
{ return iterator(*this, subset, 
                  V3i(subset.min.x, subset.min.y, subset.max.z + 1));
}

//----------------------------------------------------------------------------//
// WritableField::iterator
//----------------------------------------------------------------------------//

template <class Data_T>
class WritableField<Data_T>::iterator
{

public:

  // Constructors --------------------------------------------------------------

  iterator(WritableField<Data_T> &field, const Box3i &window,
           const V3i &currentPos)
    : x(currentPos.x), y(currentPos.y), z(currentPos.z), 
      m_window(window), m_field(field)
  { }

  // Operators -----------------------------------------------------------------

  inline const iterator& operator ++ ()
  {
    if (x == m_window.max.x) {
      if (y == m_window.max.y) {
        x = m_window.min.x;
        y = m_window.min.y;
        ++z;
      } else {
        x = m_window.min.x;
        ++y;
      }
    } else {
      ++x;
    }
    return *this;
  }

  template <class Iter_T>
  bool operator == (const Iter_T &rhs) const
  {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }

  template <class Iter_T>
  bool operator != (const Iter_T &rhs) const
  {
    return x != rhs.x || y != rhs.y || z != rhs.z;
  }

  inline Data_T& operator * () const
  {
    return m_field.lvalue(x, y, z);
  }

  // Public data members -------------------------------------------------------

  //! Current position
  int x, y, z;

private:

  // Private data members ------------------------------------------------------

  //! Window to traverse
  Box3i m_window;
  //! Reference to field being iterated over
  WritableField<Data_T> &m_field;

};

//----------------------------------------------------------------------------//
// ResizableField
//----------------------------------------------------------------------------//

/*! \class ResizableField
  \ingroup field
  This class adds the ability to resize the data storage object. Most Field
  subclasses will derive from this class. Only classes that cannot implement
  sizeChanged() in a reasonable manner should derive from Field or 
  WritableField.
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class ResizableField
  : public WritableField<Data_T>
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::intrusive_ptr<ResizableField> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef ResizableField<Data_T> class_type;
  DEFINE_FIELD_RTTI_ABSTRACT_CLASS;
  
  // Main methods --------------------------------------------------------------

  //! Resizes the object
  //! \warning Never call this from a constructor. It calls the virtual
  //! function sizeChanged().
  void setSize(const V3i &size);
  //! Resizes the object
  //! \warning Never call this from a constructor. It calls the virtual
  //! function sizeChanged().
  void setSize(const Box3i &extents);
  //! Resizes the object
  //! \warning Never call this from a constructor. It calls the virtual
  //! function sizeChanged().
  void setSize(const Box3i &extents, const Box3i &dataWindow);
  //! Resizes the object with padding
  //! \warning Never call this from a constructor. It calls the virtual
  //! function sizeChanged().
  void setSize(const V3i &size, int padding);

  //! Copies the data from another Field, also resizes
  void copyFrom(typename Field<Data_T>::Ptr other);
  //! Copies the data from another Field of another template class, 
  //! also resizes
  template <class Data_T2>
  void copyFrom(typename Field<Data_T2>::Ptr other);

  //! Sets up this field so that resolution and mapping matches the other
  void matchDefinition(FieldRes::Ptr fieldToMatch);

protected:

  // Typedefs ------------------------------------------------------------------

  typedef WritableField<Data_T> base;

  // To be implemented by subclasses -------------------------------------------

  //! Subclasses should re-implement this if they need to perform memory 
  //! allocations, etc. every time the size of the storage changes.
  //! \note Make sure to call the base class version in subclasses!
  virtual void sizeChanged()
  { base::m_mapping->setExtents(base::m_extents); }

};

//----------------------------------------------------------------------------//

template <class Data_T>
void ResizableField<Data_T>::setSize(const V3i &size)
{
  Field<Data_T>::m_extents.min = V3i(0);
  Field<Data_T>::m_extents.max = size - V3i(1);
  Field<Data_T>::m_dataWindow = Field<Data_T>::m_extents;

  // Tell subclasses that the size changed so they can update themselves.
  sizeChanged();
}

//----------------------------------------------------------------------------//

template <class Data_T>
void ResizableField<Data_T>::setSize(const Box3i &extents)
{ 
  Field<Data_T>::m_extents = extents;
  Field<Data_T>::m_dataWindow = extents;
  // Tell subclasses that the size changed so they can update themselves.
  sizeChanged();
}

//----------------------------------------------------------------------------//

template <class Data_T>
void ResizableField<Data_T>::setSize(const Box3i &extents, 
                                     const Box3i &dataWindow)
{ 
    
  Field<Data_T>::m_extents = extents;
  Field<Data_T>::m_dataWindow = dataWindow;
  // Tell subclasses that the size changed so they can update themselves.
  sizeChanged();
}

//----------------------------------------------------------------------------//

template <class Data_T>
void ResizableField<Data_T>::setSize(const V3i &size, int padding)
{ 
  setSize(Box3i(V3i(0), size - V3i(1)),
          Box3i(V3i(-padding), 
                size + V3i(padding - 1))); 
}

//----------------------------------------------------------------------------//

template <class Data_T>
void ResizableField<Data_T>::copyFrom(typename Field<Data_T>::Ptr other)
{
  // Set mapping
  setMapping(other->mapping());
  // Set size to match
  setSize(other->extents(), other->dataWindow());

  // Copy over the data
  typename base::iterator i = base::begin();
  typename base::iterator end = base::end();
  typename Field<Data_T>::const_iterator c = other->cbegin();
  for (; i != end; ++i, ++c)
    *i = *c;
}

//----------------------------------------------------------------------------//

template <class Data_T>
template <class Data_T2>
void ResizableField<Data_T>::copyFrom(typename Field<Data_T2>::Ptr other) 
{
  // Set mapping
  setMapping(other->mapping());
  // Set size to match
  setSize(other->extents(), other->dataWindow());
  // Copy over the data
  typename base::iterator i = base::begin();
  typename base::iterator end = base::end();
  typename Field<Data_T2>::const_iterator c = other->cbegin();
  for (; i != end; ++i, ++c)
    *i = *c;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void ResizableField<Data_T>::matchDefinition(FieldRes::Ptr fieldToMatch)
{
  setSize(fieldToMatch->extents(), fieldToMatch->dataWindow());
  FieldRes::setMapping(fieldToMatch->mapping());
}

//----------------------------------------------------------------------------//
// Field-related utility functions
//----------------------------------------------------------------------------//

//! Checks whether the mapping and resolution in two different fields are
//! identical
template <class Data_T, class Data_T2>
bool sameDefinition(typename Field<Data_T>::Ptr a, 
                    typename Field<Data_T2>::Ptr b)
{
  if (a->extents() != b->extents()) {
    return false;
  } 
  if (a->dataWindow() != b->dataWindow()) {
    return false;
  }
  if (!a->mapping()->isIdentical(b->mapping())) {
    return false;
  }
  return true;
}

//----------------------------------------------------------------------------//

//! Checks whether the span and data in two different fields are identical
//! \todo This should also check the mapping
template <class Data_T>
bool isIdentical(typename Field<Data_T>::Ptr a, typename Field<Data_T>::Ptr b)
{
  if (!sameDefinition<Data_T, Data_T>(a, b)) {
    return false;
  }
  // If data window is the same, we can safely assume that the range of
  // both fields' iterators are the same.
  typename Field<Data_T>::const_iterator is1 = a->cbegin();
  typename Field<Data_T>::const_iterator is2 = b->cbegin();
  typename Field<Data_T>::const_iterator ie1 = a->cend();
  bool same = true;
  for (; is1 != ie1; ++is1, ++is2) {
    if (*is1 != *is2) {
      same = false;
      break;
    }
  }
  return same;
}

//----------------------------------------------------------------------------//

//! Goes from continuous coordinates to discrete coordinates
//! See Graphics Gems - What is a pixel
inline int contToDisc(double contCoord)
{
  return static_cast<int>(std::floor(contCoord));
}

//----------------------------------------------------------------------------//

//! Goes from discrete coordinates to continuous coordinates
//! See Graphics Gems - What is a pixel
inline double discToCont(int discCoord)
{
  return static_cast<double>(discCoord) + 0.5;
}

//----------------------------------------------------------------------------//

//! Goes from continuous coords to discrete for a 2-vector
inline V2i contToDisc(const V2d &contCoord)
{
  return V2i(contToDisc(contCoord.x), contToDisc(contCoord.y));  
}

//----------------------------------------------------------------------------//

//! Goes from discrete coords to continuous for a 2-vector
inline V2d discToCont(const V2i &discCoord)
{
  return V2d(discToCont(discCoord.x), discToCont(discCoord.y));  
}

//----------------------------------------------------------------------------//

//! Goes from continuous coords to discrete for a 3-vector
inline V3i contToDisc(const V3d &contCoord)
{
  return V3i(contToDisc(contCoord.x), contToDisc(contCoord.y),
             contToDisc(contCoord.z));
}

//----------------------------------------------------------------------------//

//! Goes from discrete coords to continuous for a 3-vector
inline V3d discToCont(const V3i &discCoord)
{
  return V3d(discToCont(discCoord.x), discToCont(discCoord.y),
             discToCont(discCoord.z));  
}

//----------------------------------------------------------------------------//

//! \ingroup template_util
template <class Iter_T>
void advance(Iter_T &iter, int num) 
{
  if (num <= 0) return;
  for (int i=0; i<num; ++i, ++iter);
}

//----------------------------------------------------------------------------//

//! \ingroup template_util
template <class Iter_T>
void advance(Iter_T &iter, int num, const Iter_T &end) 
{
  if (num <= 0) 
    return;
  for (int i=0; i<num && iter != end; ++i, ++iter);
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard

