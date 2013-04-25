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
  \brief Contains MIPField class
  \ingroup field
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_MIPField_H_
#define _INCLUDED_Field3D_MIPField_H_

#include "Field.h"
#include "RefCount.h"
#include "Types.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// LazyLoadAction
//----------------------------------------------------------------------------//

/*! \class LazyLoadAction
  This run-time functor holds an action that executes the loading of a field.
  The template argument is the return type of the functor;
 */

//----------------------------------------------------------------------------//

template <class Field_T>
class LazyLoadAction
{
public:
  
  // Typedefs ------------------------------------------------------------------

  typedef boost::shared_ptr<LazyLoadAction<Field_T> > Ptr;
  typedef std::vector<Ptr> Vec;
  
  virtual ~LazyLoadAction()
  { }

  // To be implemented by subclasses -------------------------------------------
  
  //! Performs the loading of the pre-determined field and returns a pointer 
  //! to it
  virtual typename Field_T::Ptr load() const = 0;

};

//----------------------------------------------------------------------------//
// MIPField
//----------------------------------------------------------------------------//

/*! \class MIPField

  Some assumptions:

  MIP fields are neither Resizable nor Writable. They are constructed from an
  existing field such as DenseField, SparseField, etc.

  The highest resolution representation is level 0.

  All MIPField subclasses are delayed-read, so as not to touch high res
  data unless needed.

  The base class provides mipValue() and mipLValue(). It is assumed that
  concrete subclasses provide fastMipValue() and fastMipLValue().


 */

//----------------------------------------------------------------------------//

template <class Data_T>
class MIPField : public Field<Data_T>
{

public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::intrusive_ptr<MIPField> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef MIPField<Data_T> class_type;
  DEFINE_FIELD_RTTI_ABSTRACT_CLASS;

  static const char *staticClassName()
  {
    return "MIPField";
  }
  
  static const char* classType()
  {
    return MIPField<Data_T>::ms_classType.name();
  }
  
  // Constructors --------------------------------------------------------------

  MIPField();
  
  // To be implemented by subclasses -------------------------------------------

  //! Read access to a voxel in a given MIP level
  //! \param level The MIP level to read from
  virtual Data_T mipValue(size_t level, int i, int j, int k) const = 0;

  //! Returns the resolution of a given MIP level
  virtual V3i mipResolution(size_t level) const = 0;

  // Main methods --------------------------------------------------------------

  //! Sets the lowest MIP level to use. Defaults to zero, but can be set higher 
  //! to prevent high resolution levels from being accessed.
  void setLowestLevel(size_t level);
  //! Lowest MIP level to use.
  size_t lowestLevel() const
  { return m_lowestLevel; }
  //! Number of MIP levels
  size_t numLevels() const
  { return m_numLevels; }

protected:

  // Static data members -------------------------------------------------------

  static TemplatedFieldType<MIPField<Data_T> > ms_classType;

  // Typedefs ------------------------------------------------------------------

  typedef Field<Data_T> base;

  // Data members --------------------------------------------------------------

  //! Number of MIP levels. The default is 1.
  size_t m_numLevels;
  //! The lowest MIP level to use. Defaults to 0, but can be set higher to
  //! prevent high resolution levels from being accessed.
  size_t m_lowestLevel;

};

//----------------------------------------------------------------------------//
// MIPField implementations
//----------------------------------------------------------------------------//

template <typename Data_T>
MIPField<Data_T>::MIPField()
  : m_numLevels(1), m_lowestLevel(0)
{
  
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void MIPField<Data_T>::setLowestLevel(size_t level)
{
  m_lowestLevel = level;
}

//----------------------------------------------------------------------------//
// Static member initialization
//----------------------------------------------------------------------------//

FIELD3D_CLASSTYPE_TEMPL_INSTANTIATION(MIPField);

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard

