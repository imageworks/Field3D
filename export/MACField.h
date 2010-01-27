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

/*! \file MACField.h
  \brief Contains the MACField class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_MACField_H_
#define _INCLUDED_Field3D_MACField_H_

#include <vector>
#include <boost/lexical_cast.hpp>

#include "Field.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Forward declarations 
//----------------------------------------------------------------------------//

template <class T>
class LinearMACFieldInterp; 
template <class T>
class CubicMACFieldInterp; 

//----------------------------------------------------------------------------//
// Enumerators 
//----------------------------------------------------------------------------//

enum MACComponent {
  MACCompU = 0,
  MACCompV,
  MACCompW
};

//----------------------------------------------------------------------------//
// MACField
//----------------------------------------------------------------------------//

/*! \class MACField
  \ingroup field
  \brief This subclass of Field implements a standard MAC field. Refer to
  your favorite fluid simulations book for an explanation.

  The definition for u,v,w indexing used can be found on page 25 in
  Bridson - Fluid Simulation for Computer Graphics

  \note This class can only be templated on Vec3 instances.
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class MACField : public ResizableField<Data_T>
{
public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<MACField> Ptr;
  typedef std::vector<Ptr> Vec;

  //! This typedef is used to refer to the scalar component type of the vectors
  typedef typename Data_T::BaseType real_t;

  typedef LinearMACFieldInterp<Data_T> LinearInterp;
  typedef CubicMACFieldInterp<Data_T> CubicInterp;

  // Constructors --------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  //! Constructs an empty buffer
  MACField();

  // \}

  // Main methods --------------------------------------------------------------

  //! Clears all the voxels in the storage
  virtual void clear(const Data_T &value);

  // From Field base class -----------------------------------------------------

  //! \name From Field
  //! \{  

  //! \note This returns the voxel-centered interpolated value
  virtual Data_T value(int i, int j, int k) const;
  virtual long long int memSize() const;

  //! \}

  // RTTI replacement ----------------------------------------------------------

  typedef MACField<Data_T> class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  // From WritableField base class ---------------------------------------------

  //! \name From WritableField
  //! \{

  //! This will return the appropriate interpolated value but
  //! setting that to something else does not change the MAC field.
  //! \warning See description
  virtual Data_T& lvalue(int i, int j, int k);

  //! \}
  
  // Concrete component access -------------------------------------------------

  //! \name MAC-component access
  //! \{

  //! Read access to value on u-facing wall
  //! \note i coordinate represents i-1/2!
  const real_t& u(int i, int j, int k) const;
  //! Write access to value on u-facing wall
  //! \note i coordinate represents i-1/2!
  real_t& u(int i, int j, int k);
  //! Read access to value on v-facing wall
  //! \note j coordinate represents j-1/2!
  const real_t& v(int i, int j, int k) const;
  //! Write access to value on v-facing wall
  //! \note j coordinate represents j-1/2!
  real_t& v(int i, int j, int k);
  //! Read access to value on w-facing wall
  //! \note k coordinate represents k-1/2!
  const real_t& w(int i, int j, int k) const;
  //! Write access to value on w-facing wall
  //! \note k coordinate represents k-1/2!
  real_t& w(int i, int j, int k);

  //! \}

  // Iterators -----------------------------------------------------------------

  /*! A note about MAC iterators
     The subset that we choose to iterate over is defined by voxels, not
     MAC face coordinates. Thus, iterator from (0, 0, 0) to (0, 0, 0) will
     actually visit (0, 0, 0) and (1, 0, 0) for the u component, and
     (0, 0, 0) to (0, 1, 0) for the v component...
  */

  //! \name MAC-component iterators
  //! \{

  //! Const iterator for traversing the u values.
  class const_mac_comp_iterator;
  //! Non-const iterator for traversing the u values.
  class mac_comp_iterator;

  //! Const iterator to first element. "cbegin" matches the tr1 c++ standard.
  const_mac_comp_iterator cbegin_comp(MACComponent comp) const;
  //! Const iterator to first element of specific subset
  const_mac_comp_iterator cbegin_comp(MACComponent comp, 
                                      const Box3i &subset) const;
  //! Const iterator to first element. "cbegin" matches the tr1 c++ standard.
  const_mac_comp_iterator cend_comp(MACComponent comp) const;
  //! Const iterator to first element of specific subset
  const_mac_comp_iterator cend_comp(MACComponent comp, 
                                    const Box3i &subset) const;
  
  //! Iterator to first element. 
  mac_comp_iterator begin_comp(MACComponent comp);
  //! Iterator to first element of specific subset
  mac_comp_iterator begin_comp(MACComponent comp, 
                               const Box3i &subset);
  //! Iterator to first element. 
  mac_comp_iterator end_comp(MACComponent comp);
  //! Iterator to first element of specific subset
  mac_comp_iterator end_comp(MACComponent comp, 
                             const Box3i &subset);

  //! \}

  // Utility methods -----------------------------------------------------------

  //! Returns the u-component interpolated to the cell center
  real_t uCenter(int i, int j, int k) const
  {
    return (u(i, j, k) + u(i + 1, j, k)) * 0.5;
  }
  //! Returns the v-component interpolated to the cell center
  real_t vCenter(int i, int j, int k) const
  {
    return (v(i, j, k) + v(i, j + 1, k)) * 0.5;
  }
  //! Returns the w-component interpolated to the cell center
  real_t wCenter(int i, int j, int k) const
  {
    return (w(i, j, k) + w(i, j, k + 1)) * 0.5;
  }

  //! TEMP: Copies the MAC field data from another MAC field. This should
  //! be re-implemented using proper iterators over the field
  void copyMAC(MACField::Ptr other)
  {
    matchDefinition(other);
    std::copy(other->m_u.begin(), other->m_u.end(), m_u.begin());
    std::copy(other->m_v.begin(), other->m_v.end(), m_v.begin());
    std::copy(other->m_w.begin(), other->m_w.end(), m_w.begin());
  }

  // Utility methods -----------------------------------------------------------

  //! Returns the size of U,V,W components 
  V3i getComponentSize() const
  { return V3i(m_u.size(), m_v.size(), m_w.size()); }
  
  // From FieldBase ------------------------------------------------------------

  //! \name From FieldBase
  //! \{
  virtual std::string className() const
  { return std::string("MACField"); }

  virtual FieldBase::Ptr clone() const
  { return Ptr(new MACField(*this)); }

  //! \}

 protected:

  // From ResizableField class ---------------------------------------------

  virtual void sizeChanged();

  // Concrete component access -------------------------------------------------

  //! Direct access to value on u-facing wall
  //! \note i coordinate represents i-1/2!
  const real_t* uPtr(int i, int j, int k) const;
  //! Direct access to value on u-facing wall
  //! \note i coordinate represents i-1/2!
  real_t* uPtr(int i, int j, int k);
  //! Direct access to value on v-facing wall
  //! \note j coordinate represents j-1/2!
  const real_t* vPtr(int i, int j, int k) const;
  //! Direct access to value on v-facing wall
  //! \note j coordinate represents j-1/2!
  real_t* vPtr(int i, int j, int k);
  //! Direct access to value on w-facing wall
  //! \note k coordinate represents k-1/2!
  const real_t* wPtr(int i, int j, int k) const;
  //! Direct access to value on w-facing wall
  //! \note k coordinate represents k-1/2!
  real_t* wPtr(int i, int j, int k);

  // Data members --------------------------------------------------------------

  //! U component storage
  std::vector<real_t> m_u;
  //! V component storage
  std::vector<real_t> m_v;
  //! W component storage
  std::vector<real_t> m_w;

  //! Size of U grid along each axis
  V3i m_uSize;
  //! Size of xy slice for u component
  int m_uSizeXY;
  //! Size of V grid along each axis
  V3i m_vSize;
  //! Size of xy slice for v component
  int m_vSizeXY;
  //! Size of W grid along each axis
  V3i m_wSize;
  //! Size of xy slice for w component
  int m_wSizeXY;

  //! Dummy storage of a temp value that lvalue() can write to
  mutable Data_T m_dummy;

 private:

  // Typedefs ------------------------------------------------------------------

  typedef ResizableField<Data_T> base;

};

//----------------------------------------------------------------------------//
// Typedefs
//----------------------------------------------------------------------------//

typedef MACField<V3h> MACField3h;
typedef MACField<V3f> MACField3f;
typedef MACField<V3d> MACField3d;

//----------------------------------------------------------------------------//
// MACField::const_mac_comp_iterator
//----------------------------------------------------------------------------//

template <class Data_T>
class MACField<Data_T>::const_mac_comp_iterator
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef MACField<Data_T> class_type;
  typedef typename MACField<Data_T>::real_t real_t;

  // Constructors --------------------------------------------------------------

  const_mac_comp_iterator(MACComponent comp, 
                          const class_type &field, 
                          const Box3i &window, 
                          const V3i &currentPos)
    : x(currentPos.x), y(currentPos.y), z(currentPos.z), 
      m_p(NULL), m_window(window), m_comp(comp), 
      m_field(field)
  { 
    updatePointer();
  }

  // Operators -----------------------------------------------------------------

  const const_mac_comp_iterator& operator ++ ()
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
      updatePointer();
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

  inline const real_t& operator * () const
  {
    return *m_p;
  }

  inline const real_t* operator -> () const
  {
    return m_p;
  }

  // Public data members -------------------------------------------------------

  //! Current position
  int x, y, z;

private:

  // Convenience methods -------------------------------------------------------

  void updatePointer()
  {
    switch (m_comp) {
    case MACCompU:
      m_p = m_field.uPtr(x, y, z);
      break;
    case MACCompV:
      m_p = m_field.vPtr(x, y, z);
      break;
    case MACCompW:
      m_p = m_field.wPtr(x, y, z);
      break;
    default:
      assert(false && "Illegal MACComponent in const_mac_comp_iterator");
    }    
  }

  // Private data members ------------------------------------------------------

  //! Pointer to current element
  const real_t *m_p;
  //! Window to traverse
  Box3i m_window;
  //! Component to look up
  MACComponent m_comp;
  //! Reference to field being iterated over
  const class_type &m_field;

};

//----------------------------------------------------------------------------//

template <class Data_T>
class MACField<Data_T>::mac_comp_iterator
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef MACField<Data_T> class_type;
  typedef typename MACField<Data_T>::real_t real_t;

  // Constructors --------------------------------------------------------------

  mac_comp_iterator(MACComponent comp, class_type &field, 
                    const Box3i &window, const V3i &currentPos)
    : x(currentPos.x), y(currentPos.y), z(currentPos.z), 
      m_p(NULL), m_window(window), m_comp(comp), 
      m_field(field)
  { 
    updatePointer();
  }

  // Operators -----------------------------------------------------------------

  mac_comp_iterator& operator ++ ()
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
      updatePointer();
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

  inline real_t& operator * () const
  {
    return *m_p;
  }

  inline real_t* operator -> () const
  {
    return m_p;
  }

  // Public data members -------------------------------------------------------

  //! Current position
  int x, y, z;

private:

  // Convenience methods -------------------------------------------------------

  void updatePointer()
  {
    switch (m_comp) {
    case MACCompU:
      m_p = m_field.uPtr(x, y, z);
      break;
    case MACCompV:
      m_p = m_field.vPtr(x, y, z);
      break;
    case MACCompW:
      m_p = m_field.wPtr(x, y, z);
      break;
    default:
      assert(false && "Illegal MACComponent in const_mac_comp_iterator");
    }    
  }

  // Private data members ------------------------------------------------------

  //! Pointer to current element
  real_t *m_p;
  //! Window to traverse
  Box3i m_window;
  //! Component to look up
  MACComponent m_comp;
  //! Reference to field being iterated over
  class_type &m_field;

};

//----------------------------------------------------------------------------//
// Implementation specific helpers 
//----------------------------------------------------------------------------//

namespace MACFieldUtil {

  inline Box3i makeDataWindowForComponent(Box3i dataWindow, MACComponent comp)
  {
    switch (comp) {
    case MACCompU:
      dataWindow.max += V3i(1, 0, 0);
      break;
    case MACCompV:
      dataWindow.max += V3i(0, 1, 0);
      break;
    case MACCompW:
      dataWindow.max += V3i(0, 0, 1);
      break;
    default:
      assert(false && "Illegal MACComponent in makeDataWindowForComponent");
    } 
    return dataWindow;
  }

}

//----------------------------------------------------------------------------//
// MACField implementations
//----------------------------------------------------------------------------//

template <class Data_T>
MACField<Data_T>::MACField()
  : base()
{
  
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MACField<Data_T>::clear(const Data_T &value)
{
  std::fill(m_u.begin(), m_u.end(), value.x);
  std::fill(m_v.begin(), m_v.end(), value.y);
  std::fill(m_w.begin(), m_w.end(), value.z);
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T MACField<Data_T>::value(int i, int j, int k) const
{
  return Data_T(uCenter(i, j, k), vCenter(i, j, k), wCenter(i, j, k));
}

//----------------------------------------------------------------------------//

template <class Data_T>
long long int MACField<Data_T>::memSize() const
{ 
  long long int superClassMemSize = base::memSize();
  long long int vectorMemSize = 
    (m_u.capacity() + m_v.capacity() + m_w.capacity()) * sizeof(real_t);
  return sizeof(*this) + vectorMemSize + superClassMemSize; 
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T& MACField<Data_T>::lvalue(int i, int j, int k)
{
  m_dummy = value(i, j, k);
  return m_dummy;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void MACField<Data_T>::sizeChanged() 
{
  // Call base class
  base::sizeChanged();

  V3i baseSize = 
    base::m_dataWindow.max - base::m_dataWindow.min + V3i(1);

  if (std::min(std::min(baseSize.x, baseSize.y), baseSize.z) < 0)
    throw Exc::ResizeException("Attempt to resize ResizableField object "
                               "using negative size. Data window was: " +
                               boost::lexical_cast<std::string>(baseSize));

  // Calculate the size for each component of the MAC field
  m_uSize = baseSize + V3i(1, 0, 0);
  m_vSize = baseSize + V3i(0, 1, 0);
  m_wSize = baseSize + V3i(0, 0, 1);

  // Calculate the size of each z slice
  m_uSizeXY = m_uSize.x * m_uSize.y;
  m_vSizeXY = m_vSize.x * m_vSize.y;
  m_wSizeXY = m_wSize.x * m_wSize.y;

  // Allocate memory
  try {
    m_u.resize(m_uSize.x * m_uSize.y * m_uSize.z);
    m_v.resize(m_vSize.x * m_vSize.y * m_vSize.z);
    m_w.resize(m_wSize.x * m_wSize.y * m_wSize.z);
  }
  catch (std::bad_alloc &e) {
    throw Exc::MemoryException("Couldn't allocate MACField of size " + 
                               boost::lexical_cast<std::string>(baseSize));
  }

}

//----------------------------------------------------------------------------//

template <class Data_T>
const typename MACField<Data_T>::real_t& 
MACField<Data_T>::u(int i, int j, int k) const
{
  assert (i >= base::m_dataWindow.min.x);
  assert (i <= base::m_dataWindow.max.x + 1);
  assert (j >= base::m_dataWindow.min.y);
  assert (j <= base::m_dataWindow.max.y);
  assert (k >= base::m_dataWindow.min.z);
  assert (k <= base::m_dataWindow.max.z);
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  return m_u[i + j * m_uSize.x + k * m_uSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::real_t& 
MACField<Data_T>::u(int i, int j, int k)
{
  assert (i >= base::m_dataWindow.min.x);
  assert (i <= base::m_dataWindow.max.x + 1);
  assert (j >= base::m_dataWindow.min.y);
  assert (j <= base::m_dataWindow.max.y);
  assert (k >= base::m_dataWindow.min.z);
  assert (k <= base::m_dataWindow.max.z);
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  return m_u[i + j * m_uSize.x + k * m_uSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
const typename MACField<Data_T>::real_t& 
MACField<Data_T>::v(int i, int j, int k) const
{
  assert (i >= base::m_dataWindow.min.x);
  assert (i <= base::m_dataWindow.max.x);
  assert (j >= base::m_dataWindow.min.y);
  assert (j <= base::m_dataWindow.max.y + 1);
  assert (k >= base::m_dataWindow.min.z);
  assert (k <= base::m_dataWindow.max.z);
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  return m_v[i + j * m_vSize.x + k * m_vSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::real_t& 
MACField<Data_T>::v(int i, int j, int k)
{
  assert (i >= base::m_dataWindow.min.x);
  assert (i <= base::m_dataWindow.max.x);
  assert (j >= base::m_dataWindow.min.y);
  assert (j <= base::m_dataWindow.max.y + 1);
  assert (k >= base::m_dataWindow.min.z);
  assert (k <= base::m_dataWindow.max.z);
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  return m_v[i + j * m_vSize.x + k * m_vSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
const typename MACField<Data_T>::real_t& 
MACField<Data_T>::w(int i, int j, int k) const
{
  assert (i >= base::m_dataWindow.min.x);
  assert (i <= base::m_dataWindow.max.x);
  assert (j >= base::m_dataWindow.min.y);
  assert (j <= base::m_dataWindow.max.y);
  assert (k >= base::m_dataWindow.min.z);
  assert (k <= base::m_dataWindow.max.z + 1);
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  return m_w[i + j * m_wSize.x + k * m_wSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::real_t& 
MACField<Data_T>::w(int i, int j, int k)
{
  assert (i >= base::m_dataWindow.min.x);
  assert (i <= base::m_dataWindow.max.x);
  assert (j >= base::m_dataWindow.min.y);
  assert (j <= base::m_dataWindow.max.y);
  assert (k >= base::m_dataWindow.min.z);
  assert (k <= base::m_dataWindow.max.z + 1);
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  return m_w[i + j * m_wSize.x + k * m_wSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::const_mac_comp_iterator 
MACField<Data_T>::cbegin_comp(MACComponent comp) const
{
  using namespace MACFieldUtil;
  if (FieldRes::dataResolution() == V3i(0))
    return cend_comp(comp);
  Box3i dataWindow = 
    makeDataWindowForComponent(base::m_dataWindow, comp);
  return const_mac_comp_iterator(comp, *this, dataWindow, dataWindow.min);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::const_mac_comp_iterator 
MACField<Data_T>::cbegin_comp(MACComponent comp, const Box3i &subset) const
{
  using namespace MACFieldUtil;
  if (subset.isEmpty())
    return cend_comp(comp, subset);
  Box3i dataWindow = makeDataWindowForComponent(subset, comp);
  return const_mac_comp_iterator(comp, *this, dataWindow, subset.min);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::const_mac_comp_iterator 
MACField<Data_T>::cend_comp(MACComponent comp) const
{
  using namespace MACFieldUtil;
  Box3i dataWindow = 
    makeDataWindowForComponent(base::m_dataWindow, comp);
  return const_mac_comp_iterator(comp, *this, dataWindow, 
                                 V3i(dataWindow.min.x,
                                     dataWindow.min.y,
                                     dataWindow.max.z + 1));
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::const_mac_comp_iterator 
MACField<Data_T>::cend_comp(MACComponent comp, const Box3i &subset) const
{
  using namespace MACFieldUtil;
  Box3i dataWindow = makeDataWindowForComponent(subset, comp);
  return const_mac_comp_iterator(comp, *this, dataWindow, 
                                 V3i(dataWindow.min.x,
                                     dataWindow.min.y,
                                     dataWindow.max.z + 1));
}
  
//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::mac_comp_iterator 
MACField<Data_T>::begin_comp(MACComponent comp)
{
  using namespace MACFieldUtil;
  if (FieldRes::dataResolution() == V3i(0))
    return end_comp(comp);
  Box3i dataWindow = makeDataWindowForComponent(base::m_dataWindow, comp);
  return mac_comp_iterator(comp, *this, dataWindow, dataWindow.min);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::mac_comp_iterator 
MACField<Data_T>::begin_comp(MACComponent comp, const Box3i &subset)
{
  using namespace MACFieldUtil;
  if (subset.isEmpty())
    return end_comp(comp, subset);
  Box3i dataWindow = makeDataWindowForComponent(subset, comp);
  return mac_comp_iterator(comp, *this, dataWindow, subset.min);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::mac_comp_iterator 
MACField<Data_T>::end_comp(MACComponent comp)
{
  using namespace MACFieldUtil;
  Box3i dataWindow = makeDataWindowForComponent(base::m_dataWindow, comp);
  return mac_comp_iterator(comp, *this, dataWindow, V3i(dataWindow.min.x,
                                                        dataWindow.min.y,
                                                        dataWindow.max.z + 1));
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::mac_comp_iterator 
MACField<Data_T>::end_comp(MACComponent comp, const Box3i &subset)
{
  using namespace MACFieldUtil;
  Box3i dataWindow = makeDataWindowForComponent(subset, comp);
  return mac_comp_iterator(comp, *this, dataWindow, V3i(dataWindow.min.x,
                                                        dataWindow.min.y,
                                                        dataWindow.max.z + 1));
}

//----------------------------------------------------------------------------//

template <class Data_T>
const typename MACField<Data_T>::real_t*
MACField<Data_T>::uPtr(int i, int j, int k) const
{
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  return &m_u[i + j * m_uSize.x + k * m_uSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::real_t*
MACField<Data_T>::uPtr(int i, int j, int k)
{
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  return &m_u[i + j * m_uSize.x + k * m_uSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
const typename MACField<Data_T>::real_t* 
MACField<Data_T>::vPtr(int i, int j, int k) const
{
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  return &m_v[i + j * m_vSize.x + k * m_vSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::real_t* 
MACField<Data_T>::vPtr(int i, int j, int k)
{
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  return &m_v[i + j * m_vSize.x + k * m_vSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
const typename MACField<Data_T>::real_t* 
MACField<Data_T>::wPtr(int i, int j, int k) const
{
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  return &m_w[i + j * m_wSize.x + k * m_wSizeXY];
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename MACField<Data_T>::real_t* 
MACField<Data_T>::wPtr(int i, int j, int k)
{
  // Add crop window offset
  i -= base::m_dataWindow.min.x;
  j -= base::m_dataWindow.min.y;
  k -= base::m_dataWindow.min.z;
  return &m_w[i + j * m_wSize.x + k * m_wSizeXY];
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
