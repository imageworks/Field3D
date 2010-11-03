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

#ifndef _INCLUDED_Field3D_SpiMathLib_H_
#define _INCLUDED_Field3D_SpiMathLib_H_

#include <OpenEXR/half.h>
#include <OpenEXR/ImathHalfLimits.h>

#include <OpenEXR/ImathBox.h>
#include <OpenEXR/ImathBoxAlgo.h>
#include <OpenEXR/ImathColor.h>
#include <OpenEXR/ImathMatrix.h>
#include <OpenEXR/ImathVec.h>

#include <OpenEXR/ImathRoots.h>
#include <OpenEXR/ImathMatrixAlgo.h>
#include <OpenEXR/ImathRandom.h>
#include <OpenEXR/ImathPlane.h>
#include <OpenEXR/ImathQuat.h>

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

#ifndef OPENEXR_VERSION_NS

typedef ::half               half;
typedef Imath::V2i           V2i;
typedef Imath::V2f           V2f;
typedef Imath::V2d           V2d;
typedef Imath::C3f           C3f;
typedef Imath::Color3<half>  C3h;
typedef Imath::C4f           C4f;
typedef Imath::V3i           V3i;
typedef Imath::Vec3<half>    V3h;
typedef Imath::V3f           V3f;
typedef Imath::V3d           V3d;
typedef Imath::Box2f         Box2f;
typedef Imath::Box2i         Box2i;
typedef Imath::Box3i         Box3i;
typedef Imath::Box3f         Box3f;
typedef Imath::Box3d         Box3d;
typedef Imath::M33f          M33f;
typedef Imath::M44f          M44f;
typedef Imath::M44d          M44d;
typedef Imath::Plane3d       Plane3d;
typedef Imath::Line3d        Line3d;
typedef Imath::Quatd         Quatd;


#define FIELD3D_VEC3_T       Imath::Vec3

#define FIELD3D_CLIP         Imath::clip
#define FIELD3D_LERP         Imath::lerp
#define FIELD3D_LERPFACTOR   Imath::lerpfactor
#define FIELD3D_EXTRACT_SHRT Imath::extractSHRT

#define FIELD3D_RAND48       Imath::Rand48
#define FIELD3D_RAND32       Imath::Rand32
#define FIELD3D_SOLIDSPHERERAND Imath::solidSphereRand
#define FIELD3D_HALF_LIMITS Imath::limits<SPI::Field3D::half>

// default random number generator
#define FIELD3D_RAND         Imath::Rand48

#else 



typedef SPI::OpenEXR::half                            half;
typedef SPI::OpenEXR::Imath::V2i                      V2i;
typedef SPI::OpenEXR::Imath::V2f                      V2f;
typedef SPI::OpenEXR::Imath::V2d                      V2d;
typedef SPI::OpenEXR::Imath::C3f                      C3f;
typedef SPI::OpenEXR::Imath::C4f                      C4f;
typedef SPI::OpenEXR::Imath::Color3<SPI::OpenEXR::half> C3h;
typedef SPI::OpenEXR::Imath::V3i                      V3i;
typedef SPI::OpenEXR::Imath::Vec3<SPI::OpenEXR::half> V3h;
typedef SPI::OpenEXR::Imath::V3f                      V3f;
typedef SPI::OpenEXR::Imath::V3d                      V3d;
typedef SPI::OpenEXR::Imath::Box2i                    Box2i;
typedef SPI::OpenEXR::Imath::Box2f                    Box2f;
typedef SPI::OpenEXR::Imath::Box3i                    Box3i;
typedef SPI::OpenEXR::Imath::Box3f                    Box3f;
typedef SPI::OpenEXR::Imath::Box3d                    Box3d;
typedef SPI::OpenEXR::Imath::M33f                     M33f;
typedef SPI::OpenEXR::Imath::M44f                     M44f;
typedef SPI::OpenEXR::Imath::M44d                     M44d;
typedef SPI::OpenEXR::Imath::Plane3d                  Plane3d;
typedef SPI::OpenEXR::Imath::Line3d                   Line3d;
typedef SPI::OpenEXR::Imath::Quatd                    Quatd;


#define FIELD3D_VEC3_T       SPI::OpenEXR::Imath::Vec3

#define FIELD3D_CLIP         SPI::OpenEXR::Imath::clip
#define FIELD3D_LERP         SPI::OpenEXR::Imath::lerp
#define FIELD3D_LERPFACTOR   SPI::OpenEXR::Imath::lerpfactor
#define FIELD3D_EXTRACT_SHRT SPI::OpenEXR::Imath::extractSHRT

// default random number generator
#define FIELD3D_RAND         SPI::OpenEXR::Imath::Rand48

#define FIELD3D_RAND48       SPI::OpenEXR::Imath::Rand48
#define FIELD3D_RAND32       SPI::OpenEXR::Imath::Rand32
#define FIELD3D_SOLIDSPHERERAND SPI::OpenEXR::Imath::solidSphereRand
#define FIELD3D_HALF_LIMITS  SPI::OpenEXR::Imath::limits<SPI::OpenEXR::half>

//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
#endif // OPENEXR_VERSION
FIELD3D_NAMESPACE_HEADER_CLOSE
#endif // Include guard
