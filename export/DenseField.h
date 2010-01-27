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

/*! \file DenseField.h
  \brief Contains the DenseField class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_DenseField_H_
#define _INCLUDED_Field3D_DenseField_H_

#include <vector>

#include <boost/lexical_cast.hpp>

#include "Field.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Forward declarations 
//----------------------------------------------------------------------------//

template <class Field_T>
class LinearGenericFieldInterp;
template <class Field_T>
class CubicGenericFieldInterp; 


//----------------------------------------------------------------------------//
// DenseField
//----------------------------------------------------------------------------//

/*! \class DenseField
  \ingroup field
  \brief This subclass of Field stores data in a contiguous std::vector.

  Refer to \ref using_fields for examples of how to use this in your code.
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class DenseField
  : public ResizableField<Data_T>
{
public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<DenseField> Ptr;
  typedef std::vector<Ptr> Vec;

  typedef LinearGenericFieldInterp<DenseField<Data_T> > LinearInterp;
  typedef CubicGenericFieldInterp<DenseField<Data_T> > CubicInterp;

  typedef ResizableField<Data_T> base;

  // Constructors --------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  //! Constructs an empty buffer
  DenseField();

  // \}

  // Main methods --------------------------------------------------------------

  //! Clears all the voxels in the storage
  virtual void clear(const Data_T &value);

  // From Field base class -----------------------------------------------------

  //! \name From Field
  //! \{  
  virtual Data_T value(int i, int j, int k) const;
  virtual long long int memSize() const;
  //! \}

  // RTTI replacement ----------------------------------------------------------

  typedef DenseField<Data_T> class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS

  // From WritableField base class ---------------------------------------------

  //! \name From WritableField
  //! \{
  virtual Data_T& lvalue(int i, int j, int k);
  //! \}
  
  // Concrete voxel access -----------------------------------------------------

  //! Read access to voxel. Notice that this is non-virtual.
  const Data_T& fastValue(int i, int j, int k) const;
  //! Write access to voxel. Notice that this is non-virtual.
  Data_T& fastLValue(int i, int j, int k);

  // Iterators -----------------------------------------------------------------

  //! \name Iterators
  //! \{

  //! Const iterator for traversing the values in a Field object.
  class const_iterator;

  //! Non-const iterator for traversing the values in a Field object.
  class iterator;

  //! Const iterator to first element. "cbegin" matches the tr1 c++ standard.
  const_iterator cbegin() const;
  //! Const iterator to first element of specific subset
  const_iterator cbegin(const Box3i &subset) const;
  //! Const iterator pointing one element past the last valid one.
  const_iterator cend() const;
  //! Const iterator pointing one element past the last valid one (for a 
  //! subset)
  const_iterator cend(const Box3i &subset) const;
  //! Iterator to first element.
  iterator begin();
  //! Iterator to first element of specific subset
  iterator begin(const Box3i &subset);
  //! Iterator pointing one element past the last valid one.
  iterator end();
  //! Iterator pointing one element past the last valid one (for a 
  //! subset)
  iterator end(const Box3i &subset);

  //! \}

  // Utility methods -----------------------------------------------------------

  //! Returns the internal memory size in each dimension. This is used for
  //! example in LinearInterpolator, where it optimizes random access to
  //! voxels.
  const V3i &internalMemSize() const
  { return m_memSize; }

  // From FieldBase ------------------------------------------------------------

  //! \name From FieldBase
  //! \{
  virtual std::string className() const
  { return std::string("DenseField"); }

  virtual FieldBase::Ptr clone() const
  { return Ptr(new DenseField(*this)); }

  //! \}

 protected:

  // From ResizableField class ---------------------------------------------

  virtual void sizeChanged();

  // Data members --------------------------------------------------------------

  //! Memory allocation size in each dimension
  V3i m_memSize;
  //! X scanline * Y scanline size
  int m_memSizeXY;
  //! Field storage
  std::vector<Data_T> m_data;

 private:

  // Direct access to memory for iterators -------------------------------------

  //! Returns a pointer to a given element. Used by the iterators mainly.
  inline Data_T* ptr(int i, int j, int k);
  //! Returns a pointer to a given element. Used by the iterators mainly.
  inline const Data_T* ptr(int i, int j, int k) const;

};

//----------------------------------------------------------------------------//
// Typedefs
//----------------------------------------------------------------------------//

typedef DenseField<half>   DenseFieldh;
typedef DenseField<float>  DenseFieldf;
typedef DenseField<double> DenseFieldd;
typedef DenseField<V3h>    DenseField3h;
typedef DenseField<V3f>    DenseField3f;
typedef DenseField<V3d>    DenseField3d;

//----------------------------------------------------------------------------//
// DenseField::const_iterator
//----------------------------------------------------------------------------//

template <class Data_T>
class DenseField<Data_T>::const_iterator
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef DenseField<Data_T> class_type;

  // Constructors --------------------------------------------------------------

  const_iterator(const class_type &field, const Box3i &window,
                 const V3i &currentPos)
    : x(currentPos.x), y(currentPos.y), z(currentPos.z), 
      m_window(window), m_field(field)
  { m_p = m_field.ptr(x, y, z); }

  // Operators -----------------------------------------------------------------

  const const_iterator& operator ++ ()
  {
    if (x == m_window.max.x) {
      if (y == m_window.max.y) {
        m_p = m_field.ptr(x = m_window.min.x, y = m_window.min.y, ++z);
      } else {
        m_p = m_field.ptr(x = m_window.min.x, ++y, z);
      }
    } else {
      ++x;
      ++m_p;
    }
    return *this;
  }

  template <class Iter_T>
  inline bool operator == (const Iter_T &rhs) const
  {
    return m_p == &(*rhs);
  }

  template <class Iter_T>
  inline bool operator != (const Iter_T &rhs) const
  {
    return m_p != &(*rhs);
  }

  inline const Data_T& operator * () const
  {
    return *m_p;
  }

  inline const Data_T* operator -> () const
  {
    return m_p;
  }

  // Public data members -------------------------------------------------------

  //! Current position
  int x, y, z;

private:

  // Private data members ------------------------------------------------------

  //! Pointer to current element
  const Data_T *m_p;
  //! Window to traverse
  Box3i m_window;
  //! Reference to field being iterated over
  const class_type &m_field;

};

//----------------------------------------------------------------------------//
// DenseField::iterator
//----------------------------------------------------------------------------//

template <class Data_T>
class DenseField<Data_T>::iterator
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef DenseField<Data_T> class_type;

  // Constructors --------------------------------------------------------------

  iterator(class_type &field, const Box3i &window,
           const V3i &currentPos)
    : x(currentPos.x), y(currentPos.y), z(currentPos.z), 
      m_window(window), m_field(field)
  { m_p = m_field.ptr(x, y, z); }

  // Operators -----------------------------------------------------------------

  const iterator& operator ++ ()
  {
    if (x == m_window.max.x) {
      if (y == m_window.max.y) {
        m_p = m_field.ptr(x = m_window.min.x, y = m_window.min.y, ++z);
      } else {
        m_p = m_field.ptr(x = m_window.min.x, ++y, z);
      }
    } else {
      ++x;
      ++m_p;
    }
    return *this;
  }

  template <class Iter_T>
  inline bool operator == (const Iter_T &rhs) const
  {
    return m_p == &(*rhs);
  }

  template <class Iter_T>
  inline bool operator != (const Iter_T &rhs) const
  {
    return m_p != &(*rhs);
  }

  inline Data_T& operator * () const
  {
    return *m_p;
  }

  inline Data_T* operator -> () const
  {
    return m_p;
  }

  // Public data members -------------------------------------------------------

  //! Current position
  int x, y, z;

private:

  // Private data members ------------------------------------------------------

  //! Pointer to current element
  Data_T *m_p;
  //! Window to traverse
  Box3i m_window;
  //! Reference to field being iterated over
  class_type &m_field;
};

//----------------------------------------------------------------------------//
// DenseField implementations
//----------------------------------------------------------------------------//

template <class Data_T>
DenseField<Data_T>::DenseField()
  : base(),
    m_memSize(0), m_memSizeXY(0)
{
  // Empty
}

//----------------------------------------------------------------------------//

template <class Data_T>
void DenseField<Data_T>::clear(const Data_T &value)
{
  std::fill(m_data.begin(), m_data.end(), value);
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T DenseField<Data_T>::value(int i, int j, int k) const
{
  return fastValue(i, j, k);
}

//----------------------------------------------------------------------------//

template <class Data_T>
long long int DenseField<Data_T>::memSize() const
{ 
  long long int superClassMemSize = base::memSize();
  long long int vectorMemSize = m_data.capacity() * sizeof(Data_T);
  return sizeof(*this) + vectorMemSize + superClassMemSize; 
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T& DenseField<Data_T>::lvalue(int i, int j, int k)
{
  return fastLValue(i, j, k);
}

//----------------------------------------------------------------------------//

template <class Data_T>
const Data_T& DenseField<Data_T>::fastValue(int i, int j, int k) const
{
  assert (i >= base::m_dataWindow.min.x);
  assert (i <= base::m_dataWindow.max.x);
  assert (j >= base::m_dataWindow.min.y);
  assert (j <= base::m_dataWindow.max.y);
  assert (k >= base::m_dataWindow.min.z);
  assert (k <= base::m_dataWindow.max.z);
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  // Access data
  return m_data[i + j * m_memSize.x + k * m_memSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T& DenseField<Data_T>::fastLValue(int i, int j, int k)
{
  assert (i >= base::m_dataWindow.min.x);
  assert (i <= base::m_dataWindow.max.x);
  assert (j >= base::m_dataWindow.min.y);
  assert (j <= base::m_dataWindow.max.y);
  assert (k >= base::m_dataWindow.min.z);
  assert (k <= base::m_dataWindow.max.z);
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  // Access data
  return m_data[i + j * m_memSize.x + k * m_memSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename DenseField<Data_T>::const_iterator 
DenseField<Data_T>::cbegin() const
{ 
  if (FieldRes::dataResolution() == V3i(0))
    return cend();
  return const_iterator(*this, base::m_dataWindow, base::m_dataWindow.min); 
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename DenseField<Data_T>::const_iterator 
DenseField<Data_T>::cbegin(const Box3i &subset) const
{ 
  if (subset.isEmpty())
    return cend(subset);
  return const_iterator(*this, subset, subset.min); 
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename DenseField<Data_T>::const_iterator 
DenseField<Data_T>::cend() const
{ 
  return const_iterator(*this, base::m_dataWindow, 
                        V3i(base::m_dataWindow.min.x, 
                            base::m_dataWindow.min.y,
                            base::m_dataWindow.max.z + 1));
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename DenseField<Data_T>::const_iterator 
DenseField<Data_T>::cend(const Box3i &subset) const
{ 
  return const_iterator(*this, subset, 
                        V3i(subset.min.x, subset.min.y, subset.max.z + 1));
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename DenseField<Data_T>::iterator 
DenseField<Data_T>::begin()
{ 
  if (FieldRes::dataResolution() == V3i(0))
    return end();
  return iterator(*this, base::m_dataWindow, base::m_dataWindow.min); }

//----------------------------------------------------------------------------//

template <class Data_T>
typename DenseField<Data_T>::iterator 
DenseField<Data_T>::begin(const Box3i &subset)
{ 
  if (subset.isEmpty())
    return end(subset);
  return iterator(*this, subset, subset.min); 
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename DenseField<Data_T>::iterator 
DenseField<Data_T>::end()
{ 
  return iterator(*this, base::m_dataWindow, 
                  V3i(base::m_dataWindow.min.x, 
                      base::m_dataWindow.min.y,
                      base::m_dataWindow.max.z + 1));
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename DenseField<Data_T>::iterator 
DenseField<Data_T>::end(const Box3i &subset)
{ 
  return iterator(*this, subset, 
                  V3i(subset.min.x, subset.min.y, subset.max.z + 1));
}

//----------------------------------------------------------------------------//

template <class Data_T>
void DenseField<Data_T>::sizeChanged() 
{
  // Call base class
  base::sizeChanged();

  // Calculate offsets
  m_memSize = base::m_dataWindow.max - base::m_dataWindow.min + V3i(1);
  m_memSizeXY = m_memSize.x * m_memSize.y;

  // Check that mem size is >= 0 in all dimensions
  if (std::min(std::min(m_memSize.x, m_memSize.y), m_memSize.z) < 0)
    throw Exc::ResizeException("Attempt to resize ResizableField object "
                               "using negative size. Data window was: " +
                               boost::lexical_cast<std::string>(m_memSize));

  // Allocate memory
  try {
    m_data.resize(m_memSize.x * m_memSize.y * m_memSize.z);
  }
  catch (std::bad_alloc &e) {
    throw Exc::MemoryException("Couldn't allocate DenseField of size " + 
                               boost::lexical_cast<std::string>(m_memSize));
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
inline Data_T* DenseField<Data_T>::ptr(int i, int j, int k)
{
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  // Access data
  return &m_data[i + j * m_memSize.x + k * m_memSizeXY];      
}

//----------------------------------------------------------------------------//

template <class Data_T>
inline const Data_T* DenseField<Data_T>::ptr(int i, int j, int k) const
{
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  // Access data
  return &m_data[i + j * m_memSize.x + k * m_memSizeXY];
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
