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

/*! \file SparseFile.cpp
  \brief Contains implementations relating to reading of sparse field files.
*/

//----------------------------------------------------------------------------//

#include "SparseFile.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Static instances
//----------------------------------------------------------------------------//

SparseFileManager *SparseFileManager::ms_singleton = 0;

//----------------------------------------------------------------------------//
// SparseFileManager
//----------------------------------------------------------------------------//

SparseFileManager & SparseFileManager::singleton()
{ 
  if (!ms_singleton) {
    ms_singleton = new SparseFileManager;
  }
  return *ms_singleton;
}

//----------------------------------------------------------------------------//

void SparseFileManager::setLimitMemUse(bool enabled) 
{
  m_limitMemUse = enabled;
}

//----------------------------------------------------------------------------//

bool SparseFileManager::doLimitMemUse() const
{ 
  return m_limitMemUse; 
}

//----------------------------------------------------------------------------//

void SparseFileManager::setMaxMemUse(float maxMemUse) 
{
  m_maxMemUse = maxMemUse;
}

//----------------------------------------------------------------------------//

SparseFileManager::SparseFileManager()
  : m_maxMemUse(1000.0), 
    m_limitMemUse(false)
{
  // Empty
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

