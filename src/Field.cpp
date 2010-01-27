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

/*! \file Field.cpp
  Contains some template specializations for FieldTraits.
*/

//----------------------------------------------------------------------------//

#include "Field.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// FieldBase
//----------------------------------------------------------------------------//

FieldBase::FieldBase()
  :RefBase()
{ 
  // Empty
}

//----------------------------------------------------------------------------//

FieldBase::~FieldBase()
{ 
  // Empty
}

//----------------------------------------------------------------------------//

void FieldBase::setVecFloatMetadata(const std::string &name, const V3f &val)
{ 
  m_vecFloatMetadata[name] = val; 
  this->metadataHasChanged(name);
}
    
//----------------------------------------------------------------------------//

void FieldBase::setFloatMetadata(const std::string &name, const float val)
{ 
  m_floatMetadata[name] = val; 
  this->metadataHasChanged(name);
}

//----------------------------------------------------------------------------//

void FieldBase::setVecIntMetadata(const std::string &name, const V3i &val)
{ 
  m_vecIntMetadata[name] = val; 
  this->metadataHasChanged(name);
}

//----------------------------------------------------------------------------//

void FieldBase::setIntMetadata(const std::string &name, const int val)
{ 
  m_intMetadata[name] = val; 
  this->metadataHasChanged(name);
}

//----------------------------------------------------------------------------//

void FieldBase::setStrMetadata(const std::string &name, const std::string &val)
{ 
  m_strMetadata[name] = val; 
  this->metadataHasChanged(name);
}

//----------------------------------------------------------------------------//

void FieldBase::copyMetadata(const FieldBase &field)
{
  m_vecFloatMetadata = field.m_vecFloatMetadata;
  m_floatMetadata = field.m_floatMetadata;
  m_vecIntMetadata = field.m_vecIntMetadata;
  m_intMetadata = field.m_intMetadata;
  m_strMetadata = field.m_strMetadata;    
}

//----------------------------------------------------------------------------//

V3f FieldBase::vecFloatMetadata(const std::string &name,
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

float FieldBase::floatMetadata(const std::string &name, 
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

V3i FieldBase::vecIntMetadata(const std::string &name,
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

int FieldBase::intMetadata(const std::string &name, const int defaultVal) const
{
  int retVal = defaultVal;

  IntMetadata::const_iterator i = m_intMetadata.find(name);
  if (i != m_intMetadata.end()) {
    retVal = i->second;
  } 

  return retVal;
}

//----------------------------------------------------------------------------//

std::string FieldBase::strMetadata(const std::string &name, 
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
// FieldTraits template specializations
//----------------------------------------------------------------------------//

template <>
int FieldTraits<half>::dataDims()
{
  return 1;
}

//----------------------------------------------------------------------------//

template <>
int FieldTraits<float>::dataDims()
{
  return 1;
}

//----------------------------------------------------------------------------//

template <>
int FieldTraits<double>::dataDims()
{
  return 1;
}

//----------------------------------------------------------------------------//

template <>
int FieldTraits<int>::dataDims()
{
  return 1;
}

//----------------------------------------------------------------------------//

template <>
int FieldTraits<char>::dataDims()
{
  return 1;
}

//----------------------------------------------------------------------------//

template <>
int FieldTraits<V3h>::dataDims()
{
  return 3;
}

//----------------------------------------------------------------------------//

template <>
int FieldTraits<V3f>::dataDims()
{
  return 3;
}

//----------------------------------------------------------------------------//

template <>
int FieldTraits<V3d>::dataDims()
{
  return 3;
}

//----------------------------------------------------------------------------//

template <>
int FieldTraits<C3f>::dataDims()
{
  return 3;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
