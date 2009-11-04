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

/*! \file FieldIO.h
  \brief Contains FieldIO class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_FieldIO_H_
#define _INCLUDED_Field3D_FieldIO_H_

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

class FieldIO : public RefBase
{

public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::intrusive_ptr<FieldIO> Ptr;

  // Ctors, dtor ---------------------------------------------------------------

  //! Ctor
  FieldIO() : RefBase() {}
  
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

  // Strings used when reading/writing -----------------------------------------

 private:


};



//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
