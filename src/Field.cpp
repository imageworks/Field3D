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
  : RefBase(),
    m_metadata(this)
{ 
  // Empty
}

//----------------------------------------------------------------------------//

FieldBase::FieldBase(const FieldBase &other)
  : RefBase(),
    name(other.name),
    attribute(other.attribute),
    m_metadata(this)
{ 
  m_metadata = other.m_metadata;  
}


//----------------------------------------------------------------------------//

FieldBase::~FieldBase()
{ 
  // Empty
}

//----------------------------------------------------------------------------//
// FieldRes
//----------------------------------------------------------------------------//

size_t FieldRes::numGrains() const
{
  // Grain resolution
  const V3i res = m_dataWindow.size() + V3i(1);
  // Num grains is Y * Z
  return res.y * res.z;
}

//----------------------------------------------------------------------------//

bool FieldRes::getGrainBounds(const size_t idx, Box3i &bounds) const
{
  // Grain resolution
  const V3i res   = m_dataWindow.size() + V3i(1);
  // Compute coordinate
  const int y     = idx % res.y;
  const int z     = idx / res.y;
  // Build bbox
  const V3i start = m_dataWindow.min + V3i(0, y, z);
  const V3i end   = m_dataWindow.min + V3i(res.x, y, z);
  bounds = Field3D::clipBounds(Box3i(start, end), m_dataWindow);
  // Done. We return false since we don't know the underlying memory layout.
  return false;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
