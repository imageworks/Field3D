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

/*! \file RefCount.h
  \brief Contains base class for reference counting with Mutex
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_REF_COUNT_H_
#define _INCLUDED_Field3D_REF_COUNT_H_

//#define FIELD3D_USE_ATOMIC_COUNT

//----------------------------------------------------------------------------//
#include <boost/intrusive_ptr.hpp> 

#ifdef FIELD3D_USE_ATOMIC_COUNT
#include <boost/detail/atomic_count.hpp>
#else
#include <boost/thread/mutex.hpp>
#endif

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//

class RefBase 
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

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard

