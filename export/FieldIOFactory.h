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

/*! \file FieldIOFactory.h
  \brief Contains FieldIOFactory class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_FieldIOFactory_H_
#define _INCLUDED_Field3D_FieldIOFactory_H_

//----------------------------------------------------------------------------//

#include <boost/intrusive_ptr.hpp>

#include <string>
#include <map>
#include <list>

#include <hdf5.h>
#include <typeinfo>

#include "Field.h"
#include "Log.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// FieldIO
//----------------------------------------------------------------------------//

/*! \class FieldIO
  \ingroup file_int
   A creation class.  The application needs to derive from this class
   for any new voxel field data structions.  Within the read and write methods 
   it is expected that the derived object knows how to read and write to an 
   hdf5 file through the layerGroup id.  

   \todo Merge this into ClassFactory.
*/

//----------------------------------------------------------------------------//

class FieldIO 
{

public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::intrusive_ptr<FieldIO> Ptr;

  // Ctors, dtor ---------------------------------------------------------------

  //! Ctor
  FieldIO() 
  { m_counter = 0; }
  
  //! Dtor
  virtual ~FieldIO() {}

  // Methods to be implemented by subclasses -----------------------------------

  //! Read the field at the given hdf5 group
  //! \returns Pointer to the created field, or a null pointer if the field
  //! couldn't be read.
  virtual FieldBase::Ptr read(hid_t layerGroup, const std::string &filename, 
                              const std::string &layerPath) = 0;

  //! Write the field to the given layer group
  //! \returns Whether the operation was successful
  virtual bool write(hid_t layerGroup, FieldBase::Ptr field) = 0;

  //! Returns the class name. This is used when registering the class to the
  //! FieldIOFactory object.
  virtual std::string className() const = 0;

  //! routines for boost::intrusive_pointer
  size_t refcnt(void) 
  {
    return m_counter;
  }

  void ref(void) 
  {
    m_counter++;
  }

  void unref(void) 
  {
    m_counter--;
  }

  // Strings used when reading/writing -----------------------------------------

  static std::string classNameAttrName;
  static std::string versionAttrName;

 private:
  //! for boost intrusive pointer
  mutable int m_counter;

};


//----------------------------------------------------------------------------//
// Intrusive Pointer reference counting 
//----------------------------------------------------------------------------//

inline void
intrusive_ptr_add_ref(FieldIO* r)
{
    r->ref();
}

//----------------------------------------------------------------------------//

inline void
intrusive_ptr_release(FieldIO* r)
{
    r->unref();

    if (r->refcnt() == 0)
      delete r;
}
 
//----------------------------------------------------------------------------//
// FieldIOFactory
//----------------------------------------------------------------------------//

/*! \class FieldIOFactory
  \ingroup file_int
   A voxel field factory. The application registers its own create functions
   based on the class type. Currently this class knows about hdf5 structures.

   \note This is a Singleton class
   
   \note Marshall: It might be cool to have the FieldIOFactory loadup it's own FieldIO
   objects through DSOs.  Just have some kind of path variable and some standard
   naming conventions for the DSOs that can be gathered through attributes or
   the class name.  Then load up the dsos on demand.
 */

//----------------------------------------------------------------------------//

class FieldIOFactory {

 public:

  // Ctors, dtor ---------------------------------------------------------------

  ~FieldIOFactory() 
  { clear(); }

  // Main interface ------------------------------------------------------------

  //! Returns a reference to the singleton instance
  static FieldIOFactory& fieldIOFactoryInstance();

  //! Clears all the registered IO classes
  void clear();

  //! Registers the given object instance with the factory.
  bool registerClass(FieldIO::Ptr io);

  //! Reads the given field from disk and returns it in its native data 
  //! structure
  template <class Data_T>
  typename Field<Data_T>::Ptr 
  read(const std::string &className, hid_t layerGroup,
       const std::string &filename, const std::string &layerPath) const;

  //! Writes the given field to disk in its native format.
  bool write(hid_t layerGroup, FieldBase::Ptr field) const;

 protected:

  //! Protected destructor to prevent instantiation
  FieldIOFactory() 
  { /* Empty */ }

 private:

  // Data members --------------------------------------------------------------

  typedef std::map<std::string, FieldIO::Ptr> IoClassMap;
  
  //! Map of class name and their respective IO object
  IoClassMap m_ioClasses;

  //! Singleton instance
  static FieldIOFactory* m_fieldIOFactory;

};

//----------------------------------------------------------------------------//

//! \todo Call separate read function based on template type.
template <class Data_T>
typename Field<Data_T>::Ptr 
FieldIOFactory::read(const std::string &className, 
                     hid_t layerGroup,
                     const std::string &filename, 
                     const std::string &layerPath) const
{
  using namespace boost;
  using namespace std;

  typedef typename Field<Data_T>::Ptr FieldPtr;
 
  IoClassMap::const_iterator i = m_ioClasses.find(className.c_str());
  if (i == m_ioClasses.end()) {
    Log::print(Log::SevWarning, "Unable to find class type: " + className);
    return FieldPtr();
  }

  FieldIO::Ptr io = i->second;

  assert(io != 0);

  FieldBase::Ptr field = io->read(layerGroup, filename, layerPath);
  
  if (!field) {
    Log::print(Log::SevWarning, "Couldn't read layer");
    return FieldPtr();
  }

  FieldPtr result = field_dynamic_cast<Field<Data_T> >(field);

  string typeId1(typeid(*field).name());
  string typeId2(typeid(Field<Data_T>).name());

  if (result)
    return result;

  return FieldPtr();
}

//----------------------------------------------------------------------------//

//! Convenience define
#define g_fieldIOFactory (FieldIOFactory::fieldIOFactoryInstance())

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
