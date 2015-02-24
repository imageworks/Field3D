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

/*! \file OgawaFwd.h
  \brief Contains forward declarations for Ogawa classes
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_OgawaFwd_H_
#define _INCLUDED_Field3D_OgawaFwd_H_

#include <boost/shared_ptr.hpp>

#include "ns.h"

//----------------------------------------------------------------------------//
// Forward declarations
//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

class OgIGroup;
class OgOGroup;

template <typename T>
class OgIAttribute;
template <typename T>
class OgOAttribute;

template <typename T>
class OgIDataset;
template <typename T>
class OgODataset;

typedef boost::shared_ptr<OgIGroup> OgIGroupPtr;

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

namespace Alembic {
  namespace Ogawa {
    namespace v7 {
      class OArchive;
      class IArchive;
    }
    using namespace v7;
  }
}

typedef boost::shared_ptr<Alembic::Ogawa::IArchive> IArchivePtr;

//----------------------------------------------------------------------------//

#endif // include guard
