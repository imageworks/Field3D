
//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2010 Sony Pictures Imageworks Inc
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

/**
 * @file   FieldMetadata.h
 * @brief  Basic container for metedata
 * @ingroup field
 *
 */

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_FieldMetadata_H_
#define _INCLUDED_Field3D_FieldMetadata_H_

//----------------------------------------------------------------------------//

#include <list>
#include <string>
#include <vector>
#include <map>

#include "Types.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// MetadataCallback
//----------------------------------------------------------------------------//

class MetadataCallback
{
public:

  //! Alerts the callback holder that the metadata has changed.
  virtual void metadataHasChanged(const std::string &)
  { /* Default does nothing. */ }
  
};

//----------------------------------------------------------------------------//
// FieldMetadata
//----------------------------------------------------------------------------//

class FieldMetadata
{
 public:

  // Typedefs ------------------------------------------------------------------

  typedef std::map<std::string, std::string> StrMetadata;
  typedef std::map<std::string, int> IntMetadata;
  typedef std::map<std::string, float> FloatMetadata;
  typedef std::map<std::string, V3i> VecIntMetadata;
  typedef std::map<std::string, V3f> VecFloatMetadata;

  // Ctor, dtor ----------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  FieldMetadata(MetadataCallback *owner) 
    : m_owner(owner)
  { } 
  
  virtual ~FieldMetadata() {} 

  //! \}

  // Operators -----------------------------------------------------------------

  void operator = (const FieldMetadata &other) 
  { 
    m_vecFloatMetadata = other.m_vecFloatMetadata;
    m_floatMetadata    = other.m_floatMetadata;
    m_vecIntMetadata   = other.m_vecIntMetadata;
    m_intMetadata      = other.m_intMetadata;
    m_strMetadata      = other.m_strMetadata;
  }

  // Access to metadata --------------------------------------------------------

  //! \name Metadata
  //! \{

  //! Tries to retrieve a V3f metadata value. Returns the specified
  //! default value if no metadata was found. 
  V3f vecFloatMetadata(const std::string &name, const V3f &defaultVal) const;

  //! Tries to retrieve a float metadata value. Returns the specified
  //! default value if no metadata was found. 
  float floatMetadata(const std::string &name, const float defaultVal) const;

  //! Tries to retrieve a V3i metadata value. Returns the specified
  //! default value if no metadata was found. 
  V3i vecIntMetadata(const std::string &name, const V3i &defaultVal) const;

  //! Tries to retrieve an int metadata value. Returns the specified
  //! default value if no metadata was found. 
  int intMetadata(const std::string &name, const int defaultVal) const;

  //! Tries to retrieve a string metadata value. Returns the specified
  //! default value if no metadata was found. 
  std::string strMetadata(const std::string &name, 
                          const std::string &defaultVal) const;

  //! Read only access to the m_vecFloatMetadata dictionary
  const VecFloatMetadata& vecFloatMetadata() const
  { return m_vecFloatMetadata; }
    
  //! Read only access to the m_floatMetadata dictionary
  const FloatMetadata& floatMetadata() const
  { return m_floatMetadata; }

  //! Read only access to the m_vecIntMetadata dictionary
  const VecIntMetadata& vecIntMetadata() const
  { return m_vecIntMetadata; }

  //! Read only access to the m_intMetadata dictionary
  const IntMetadata& intMetadata() const
  { return m_intMetadata; }

  //! Read only access to the m_strMetadata dictionary
  const StrMetadata& strMetadata() const
  { return m_strMetadata; }

  //! Set the a V3f value for the given metadata name.
  void setVecFloatMetadata(const std::string &name, const V3f &val);
    
  //! Set the a float value for the given metadata name.
  void setFloatMetadata(const std::string &name, const float val);

  //! Set the a V3i value for the given metadata name.
  void setVecIntMetadata(const std::string &name, const V3i &val);

  //! Set the a int value for the given metadata name.
  void setIntMetadata(const std::string &name, const int val);

  //! Set the a string value for the given metadata name.
  void setStrMetadata(const std::string &name, const std::string &val); 

  //! \}

 private:

  // Private member functions --------------------------------------------------

  FieldMetadata(const FieldMetadata &);

  // Private data members ------------------------------------------------------

  //! V3f metadata
  VecFloatMetadata m_vecFloatMetadata;
  //! Float metadata
  FloatMetadata m_floatMetadata;
  //! V3i metadata
  VecIntMetadata m_vecIntMetadata;
  //! Int metadata
  IntMetadata m_intMetadata;
  //! String metadata
  StrMetadata m_strMetadata;

  //! Pointer to owner. It is assumed that this has a lifetime at least as
  //! long as the Metadata instance.
  MetadataCallback *m_owner;

};

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif /* _INCLUDED_Field3D_FieldMetadata_H_ */

