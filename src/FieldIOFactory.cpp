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

/*! \file FieldIOFactory.cpp
  \brief Contains implementation of FieldIOFactory class
*/

//----------------------------------------------------------------------------//

#include "FieldIOFactory.h"

#include "Hdf5Util.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Field3D namespaces
//----------------------------------------------------------------------------//

using namespace Exc;
using namespace Hdf5Util;

//----------------------------------------------------------------------------//

FieldIOFactory* FieldIOFactory::m_fieldIOFactory = 0;

//----------------------------------------------------------------------------//
// Static data members
//----------------------------------------------------------------------------//

std::string FieldIO::classNameAttrName("class_name");
std::string FieldIO::versionAttrName("version");

//----------------------------------------------------------------------------//

bool
FieldIOFactory::write(hid_t layerGroup, FieldBase::Ptr field) const
{
  if (m_ioClasses.size() == 0) {
    Log::print(Log::SevWarning, 
               "No I/O classes registered. Did you forget to call initIO()?");
    return false;
  }

  IoClassMap::const_iterator i = m_ioClasses.find(field->className());
  if (i == m_ioClasses.end()) {
    Log::print(Log::SevWarning, "Unable to find class type: " + 
              field->className());
    return false;
  }

  // Add class name attribute
  if (!writeAttribute(layerGroup, FieldIO::classNameAttrName, 
                      field->className())) {
    Log::print(Log::SevWarning, "Error adding class name attribute.");
    return false;
  }


  FieldIO::Ptr io = i->second;

  assert(io != 0);

  return io->write(layerGroup, field);
}


//----------------------------------------------------------------------------//

FieldIOFactory& 
FieldIOFactory::fieldIOFactoryInstance()
{

  if (!FieldIOFactory::m_fieldIOFactory) {
    FieldIOFactory::m_fieldIOFactory = new FieldIOFactory;
  }

  return *FieldIOFactory::m_fieldIOFactory;

}

//----------------------------------------------------------------------------//

void
FieldIOFactory::clear()
{
  m_ioClasses.clear();
}

//----------------------------------------------------------------------------//

bool
FieldIOFactory::registerClass(FieldIO::Ptr io)
{
  if (!io) 
    return false;

  if (m_ioClasses.find(io->className()) != m_ioClasses.end()) {
    return false;
  }

  m_ioClasses[io->className()] = io;

  return true;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
