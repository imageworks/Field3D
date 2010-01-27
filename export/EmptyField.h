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

/*! \file EmptyField.h
  \brief Contains the EmptyField class
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_EmptyField_H_
#define _INCLUDED_Field3D_EmptyField_H_

#include <vector>

#include <boost/lexical_cast.hpp>

#include "Field.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//

// gets rid of warnings
#define UNUSED(p) ((p)=(p))

//----------------------------------------------------------------------------//

/*! \class EmptyField
  \ingroup field
  \brief This subclass of Field does not store any data.

  Its primary purpose is to be used as a proxy field. It can carry the same
  resolution, metadata and mapping as a regular field.

  Usage is similar to a DenseField, except that it does not contain
  any data.  It stores a default value that may be set and queried, so
  it can be treated as a constant field. 
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class EmptyField
  : public ResizableField<Data_T>
{
public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<EmptyField> Ptr;
  typedef std::vector<Ptr> Vec;

  // Constructors --------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  //! Constructs an empty buffer
  EmptyField();

#if 0 // These should be removed

  //! Constructs a buffer of a given size
  explicit EmptyField(const V3i &size);
  //! Constructs a buffer of a given size plus padding. The padding will
  //! be added to the data window but not to the extents
  explicit EmptyField(const V3i &size, int padding);
  //! Constructs a buffer with explicit extents
  explicit EmptyField(const Box3i &extents);
  //! Constructs a buffer with explicit extents and data window
  explicit EmptyField(const Box3i &extents, const Box3i &dataWindow);

#endif

  // \}

  // Main methods --------------------------------------------------------------

  //! Clears all the voxels in the storage
  virtual void clear(const Data_T &value);

  //! Returns the constant value
  const Data_T& constantvalue() const;
  //! Sets the constant value
  void setConstantvalue(const Data_T &val);

  // From Field base class -------------------------------------------------

  //! \name From Field
  //! \{  
  virtual Data_T value(int i, int j, int k) const;
  virtual long long int memSize() const;
  //! \}

  // RTTI replacement ----------------------------------------------------------

  typedef EmptyField<Data_T> class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS

  // From WritableField base class -----------------------------------------

  //! \name From WritableField
  //! \{
  virtual Data_T& lvalue(int i, int j, int k);
  //! \}
  
  // From FieldBase ------------------------------------------------------------

  //! \name From FieldBase
  //! \{
  virtual std::string className() const
  { return std::string("EmptyField"); }

  virtual FieldBase::Ptr clone() const
  { return Ptr(new EmptyField(*this)); }

  //! \}

 protected:

  // Data members --------------------------------------------------------------

  //! Field default value
  Data_T m_default;
  //! Dummy variable for assignment
  Data_T m_ignoredData;
  //! Field constant value
  Data_T m_constantData;

 private:

  // Typedefs ------------------------------------------------------------------

  typedef ResizableField<Data_T> base;

};

//----------------------------------------------------------------------------//
// EmptyField implementations
//----------------------------------------------------------------------------//

template <class Data_T>
EmptyField<Data_T>::EmptyField()
  : base()
{ 
  // Empty
}

#if 0

template <class Data_T>
EmptyField<Data_T>::EmptyField()
{ 
  base::setSize(Box3i(V3i(0), V3i(-1))); 
}

//----------------------------------------------------------------------------//

template <class Data_T>
EmptyField<Data_T>::EmptyField(const V3i &size)
{ 
  base::setSize(size); 
}

//----------------------------------------------------------------------------//

template <class Data_T>
EmptyField<Data_T>::EmptyField(const V3i &size, int padding)
{ 
  base::setSize(Box3i(V3i(0), size - V3i(1)),
                Box3i(V3i(-padding), 
                             size + V3i(padding - 1))); 
}

//----------------------------------------------------------------------------//

template <class Data_T>
EmptyField<Data_T>::EmptyField(const Box3i &extents)
{ 
  base::setSize(extents); 
}

//----------------------------------------------------------------------------//

template <class Data_T>
EmptyField<Data_T>::EmptyField(const Box3i &extents,
                               const Box3i &dataWindow)
{ 
  base::setSize(extents, dataWindow); 
}

#endif

//----------------------------------------------------------------------------//

template <class Data_T>
void EmptyField<Data_T>::clear(const Data_T &value)
{
  m_constantData = m_default = value;
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T EmptyField<Data_T>::value(int i, int j, int k) const
{
  assert (i >= base::m_dataWindow.min.x);
  assert (i <= base::m_dataWindow.max.x);
  assert (j >= base::m_dataWindow.min.y);
  assert (j <= base::m_dataWindow.max.y);
  assert (k >= base::m_dataWindow.min.z);
  assert (k <= base::m_dataWindow.max.z);

  UNUSED(i);
  UNUSED(j);
  UNUSED(k);

  // Access data
  return m_default;
}

//----------------------------------------------------------------------------//

template <class Data_T>
long long int EmptyField<Data_T>::memSize() const
{ 
  long long int superClassMemSize = base::memSize();
  return sizeof(*this) + superClassMemSize; 
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T& EmptyField<Data_T>::lvalue(int i, int j, int k)
{
  assert (i >= base::m_dataWindow.min.x);
  assert (i <= base::m_dataWindow.max.x);
  assert (j >= base::m_dataWindow.min.y);
  assert (j <= base::m_dataWindow.max.y);
  assert (k >= base::m_dataWindow.min.z);
  assert (k <= base::m_dataWindow.max.z);

  UNUSED(i);
  UNUSED(j);
  UNUSED(k);

  // Access data
  return m_ignoredData;
}

//----------------------------------------------------------------------------//

template <class Data_T>
inline void EmptyField<Data_T>::setConstantvalue(const Data_T &val)
{
  m_constantData = val;
}

//----------------------------------------------------------------------------//

template <class Data_T>
inline const Data_T& EmptyField<Data_T>::constantvalue() const
{
  return m_constantData;
}

//----------------------------------------------------------------------------//
// Typedefs
//----------------------------------------------------------------------------//

typedef EmptyField<float> Proxy;
typedef EmptyField<float>::Ptr ProxyPtr;
typedef std::vector<ProxyPtr> Proxies;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#undef UNUSED

//----------------------------------------------------------------------------//

#endif // Include guard
