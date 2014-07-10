//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2014 Sony Pictures Imageworks Inc., 
 *                    Pixar Animation Studios Inc.
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

/*! \file FieldMetadata.cpp
  \brief Contains implementations of FieldMetadata member functions
  \ingroup field
*/

//----------------------------------------------------------------------------//

#include "FieldMetadata.h"

#include "Field.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// FieldMetadata implementations
//----------------------------------------------------------------------------//

void FieldMetadata::setVecFloatMetadata(const std::string &name, 
                                        const V3f &val)
{ 
  m_vecFloatMetadata[name] = val; 
  if (m_owner) {
    m_owner->metadataHasChanged(name);
  }
}
    
//----------------------------------------------------------------------------//

void FieldMetadata::setFloatMetadata(const std::string &name, 
                                     const float val)
{ 
  m_floatMetadata[name] = val; 
  if (m_owner) {
    m_owner->metadataHasChanged(name);
  }
}

//----------------------------------------------------------------------------//

void FieldMetadata::setVecIntMetadata(const std::string &name, 
                                      const V3i &val)
{ 
  m_vecIntMetadata[name] = val; 
  if (m_owner) {
    m_owner->metadataHasChanged(name);
  }
}

//----------------------------------------------------------------------------//

void FieldMetadata::setIntMetadata(const std::string &name, 
                                   const int val)
{ 
  m_intMetadata[name] = val; 
  if (m_owner) {
    m_owner->metadataHasChanged(name);
  }
}

//----------------------------------------------------------------------------//

void FieldMetadata::setStrMetadata(const std::string &name, 
                                   const std::string &val)
{ 
  m_strMetadata[name] = val; 
  if (m_owner) {
    m_owner->metadataHasChanged(name);
  }
}

//----------------------------------------------------------------------------//

V3f FieldMetadata::vecFloatMetadata(const std::string &name,
                                    const V3f& defaultVal) const
{
  V3f retVal = defaultVal;
  
  VecFloatMetadata::const_iterator i = m_vecFloatMetadata.find(name);
  if (i != m_vecFloatMetadata.end()) {
    retVal = i->second;
  } 

  return retVal;
}

//----------------------------------------------------------------------------//

float FieldMetadata::floatMetadata(const std::string &name, 
                                   const float defaultVal) const
{
  float retVal = defaultVal;

  FloatMetadata::const_iterator i = m_floatMetadata.find(name);
  if (i != m_floatMetadata.end()) {
    retVal = i->second;
  } 

  return retVal;
}

//----------------------------------------------------------------------------//

V3i FieldMetadata::vecIntMetadata(const std::string &name,
                                  const V3i& defaultVal) const
{
  V3i retVal = defaultVal;

  VecIntMetadata::const_iterator i = m_vecIntMetadata.find(name);
  if (i != m_vecIntMetadata.end()) {
    retVal = i->second;
  } 

  return retVal;
}

//----------------------------------------------------------------------------//

int FieldMetadata::intMetadata(const std::string &name, 
                               const int defaultVal) const
{
  int retVal = defaultVal;

  IntMetadata::const_iterator i = m_intMetadata.find(name);
  if (i != m_intMetadata.end()) {
    retVal = i->second;
  } 

  return retVal;
}

//----------------------------------------------------------------------------//

std::string FieldMetadata::strMetadata(const std::string &name, 
                                       const std::string &defaultVal) const
{
  std::string retVal = defaultVal;

  StrMetadata::const_iterator i = m_strMetadata.find(name);
  if (i != m_strMetadata.end()) {
    retVal = i->second;
  } 

  return retVal;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
