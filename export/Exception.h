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

/*! \file Exception.h
  \brief Contains Exception base class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_Exception_H_
#define _INCLUDED_Field3D_Exception_H_

#include <stdexcept>

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//! Namespace for Exception objects
//! \ingroup exc
namespace Exc {

//----------------------------------------------------------------------------//
// Exception
//----------------------------------------------------------------------------//

/*! \class Exception
  \ingroup exc
  Base class for Exceptions. Currently very minimal. Really only makes a 
  convenient class to subclass from. 
  \note This is only intended for use within the SpComp2 - don't ever pass
  an Exception class across the .so boundary.
*/

//----------------------------------------------------------------------------//

class Exception : public std::exception
{
 public:

  // Ctor, dtor ----------------------------------------------------------------

  //! Construct from string
  Exception(const std::string &what) throw()
    : std::exception(), m_what(what)
  { }

  //! Destructor. 
  virtual ~Exception() throw()
    { }

  // std::exception ------------------------------------------------------------

  virtual const char* what() const throw()
    { return m_what.c_str(); }

 protected:

  // Data members --------------------------------------------------------------

  //! What string for the expection
  std::string m_what;

};

//----------------------------------------------------------------------------//
// Exceptions used in Field3D
//----------------------------------------------------------------------------//

//! Used to declare a generic but named exception
#define DECLARE_FIELD3D_GENERIC_EXCEPTION(name, base_class) \
class name : public base_class \
{ \
 public: \
  explicit name(const std::string &what = "") throw() \
    : base_class(what) \
  { } \
  ~name() throw() \
  { } \
}; \

//----------------------------------------------------------------------------//

DECLARE_FIELD3D_GENERIC_EXCEPTION(AttrGetNativeTypeException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(AttrGetSpaceException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(AttrGetTypeException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(BadFileHierarchyException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(BadHdf5IdException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(CreateDataSetException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(CreateDataSpaceException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(CreateDataTypeException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(CreateGroupException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(ErrorCreatingFileException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(FileIntegrityException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(GetDataSpaceException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(GetDataTypeException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(Hdf5DataReadException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(MissingAttributeException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(MissingGroupException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(NoSuchFileException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(OpenDataSetException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(ReadHyperSlabException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(ReadMappingException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(UnsupportedVersionException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(WriteAttributeException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(WriteHyperSlabException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(WriteLayerException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(WriteMACFieldDataException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(WriteMappingException, Exception)
DECLARE_FIELD3D_GENERIC_EXCEPTION(WriteSimpleDataException, Exception)

//----------------------------------------------------------------------------//

} // namespace Exc

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
