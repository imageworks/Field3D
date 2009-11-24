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

#define FIELD3D_MAJOR_VER 1
#define FIELD3D_MINOR_VER 1
#define FIELD3D_MICRO_VER 3

#define FIELD3D_VERSION_NS v1_1

#ifdef FIELD3D_EXTRA_NAMESPACE

#  define FIELD3D_NAMESPACE_OPEN \
  namespace FIELD3D_EXTRA_NAMESPACE { \
    namespace Field3D { namespace FIELD3D_VERSION_NS {
#  define FIELD3D_NAMESPACE_HEADER_CLOSE \
  } using namespace FIELD3D_VERSION_NS; } }
#  define FIELD3D_NAMESPACE_SOURCE_CLOSE \
  } } }

#else

#  define FIELD3D_NAMESPACE_OPEN \
  namespace Field3D { namespace FIELD3D_VERSION_NS {
#  define FIELD3D_NAMESPACE_HEADER_CLOSE \
  } using namespace FIELD3D_VERSION_NS; } 
#  define FIELD3D_NAMESPACE_SOURCE_CLOSE \
  } } 

#endif

//----------------------------------------------------------------------------//
