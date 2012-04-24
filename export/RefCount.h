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

/*! \file RefCount.h
  \brief Contains base class for reference counting with Mutex
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_REF_COUNT_H_
#define _INCLUDED_Field3D_REF_COUNT_H_

#define FIELD3D_USE_ATOMIC_COUNT

//----------------------------------------------------------------------------//
#include <boost/intrusive_ptr.hpp> 

#ifdef FIELD3D_USE_ATOMIC_COUNT
#include <boost/detail/atomic_count.hpp>
#else
#include <boost/thread/mutex.hpp>
#endif

#include <string.h>
#include "Traits.h"
#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Field RTTI Replacement
//----------------------------------------------------------------------------//

#define DEFINE_CHECK_RTTI_CALL                    \
  virtual bool checkRTTI(const char *typenameStr) \
  { return matchRTTI(typenameStr); }              \
  
#define DEFINE_MATCH_RTTI_CALL                        \
  bool matchRTTI(const char *typenameStr)             \
  {                                                   \
    if (strcmp(typenameStr,classType()) == 0) {       \
      return true;                                    \
    }                                                 \
    return base::matchRTTI(typenameStr);              \
  }                                                   \

#define DEFINE_FIELD_RTTI_CONCRETE_CLASS        \
  DEFINE_CHECK_RTTI_CALL                        \
  DEFINE_MATCH_RTTI_CALL                        \

#define DEFINE_FIELD_RTTI_ABSTRACT_CLASS        \
  DEFINE_MATCH_RTTI_CALL                        \

//----------------------------------------------------------------------------//

class FIELD3D_API RefBase 
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::intrusive_ptr<RefBase> Ptr;

  // Constructors --------------------------------------------------------------

  //! \name Constructors, destructors, copying
  //! \{

  RefBase() 
    : m_counter(0) 
  {}
    
  //! Copy constructor
  RefBase(const RefBase&) 
    : m_counter(0) 
  {}

  //! Assignment operator
  RefBase& operator= (const RefBase&)
  { return *this; }

  //! Destructor
  virtual ~RefBase() 
  {}

  //! \}

  // Reference counting --------------------------------------------------------

  //! Used by boost::intrusive_pointer
  size_t refcnt() 
  { return m_counter; }
    
  //! Used by boost::intrusive_pointer
  void ref() const
  { 
#ifndef FIELD3D_USE_ATOMIC_COUNT
    boost::mutex::scoped_lock lock(m_refMutex);
#endif
    ++m_counter; 
  }

  //! Used by boost::intrusive_pointer
  void unref() const
  {
#ifndef FIELD3D_USE_ATOMIC_COUNT
    boost::mutex::scoped_lock lock(m_refMutex);
#endif
    --m_counter; 
    // since we use intrusive_pointer no need
    // to delete the object ourselves.
  }
  
  // RTTI replacement ----------------------------------------------------------

  /*! \note A note on why the RTTI replacement is needed:
     RTTI calls fail once the object crosses the dso boundary. We revert
     to using simple string checks which is more expensive but at least works
     once the dso is used in Houdini, etc.
     Use field_dynamic_cast<> for any RefBase subclass instead of 
     dynamic_cast<>.
  */

  //! \name RTTI replacement
  //! \{

  //! This function is only implemented by concrete classes and triggers
  //! the actual RTTI check through matchRTTI();
  virtual bool checkRTTI(const char *typenameStr) = 0;
  
  //! Performs a check to see if the given typename string matches this class'
  //! This needs to be implemented in -all- subclasses, even abstract ones.
  bool matchRTTI(const char *typenameStr)
  {
    if (strcmp(classType(), typenameStr) == 0)
      return true;
    return false;
  }

  static const char *classType()
  {
    return "RefBase";
  }

  //! \}

private:

  //! For boost intrusive pointer
#ifdef FIELD3D_USE_ATOMIC_COUNT
  mutable boost::detail::atomic_count m_counter;
#else
  mutable long m_counter;
  //! Mutex for ref counting
  mutable boost::mutex m_refMutex;     
#endif

};

//----------------------------------------------------------------------------//
// Intrusive Pointer reference counting 
//----------------------------------------------------------------------------//

inline void 
intrusive_ptr_add_ref(RefBase* r)
{
  r->ref();
}

//----------------------------------------------------------------------------//

inline void
intrusive_ptr_release(RefBase* r)
{
  r->unref();

  if (r->refcnt() == 0)
    delete r;
}

//----------------------------------------------------------------------------//
// field_dynamic_cast
//----------------------------------------------------------------------------//

//! Dynamic cast that uses string-comparison in order to be safe even
//! after an object crosses a shared library boundary.
//! \ingroup field
template <class Field_T>
typename Field_T::Ptr
field_dynamic_cast(RefBase::Ptr field)
{
  if (!field) 
    return NULL;

  const char *tgtTypeString =  Field_T::classType();
  
  if (field->checkRTTI(tgtTypeString)) {
    return static_cast<Field_T*>(field.get());
  } else {
    return NULL;
  }
}

//#define FIELD_DYNAMIC_CAST boost::dynamic_pointer_cast
#define FIELD_DYNAMIC_CAST field_dynamic_cast

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard

