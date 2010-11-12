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

#ifndef _INCLUDED_Field3D_gpu_FieldInterpCuda_H_
#define _INCLUDED_Field3D_gpu_FieldInterpCuda_H_

// field3d includes
#include "Field3D/gpu/ns.h"
#include "Field3D/gpu/Traits.h"

// cuda includes
#include <cuda.h>
#include <cutil_math.h>

// std includes
#include <stdexcept>

FIELD3D_GPU_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// LinearFieldInterp
//----------------------------------------------------------------------------//
//! continuous (interpolated) sampling via a uniform grid sampler
//  (ie providing i,j,k access)
//----------------------------------------------------------------------------//

template <typename Sampler_T>
struct LinearFieldInterp
{
  typedef LinearFieldInterp<Sampler_T> interp_type;
  typedef Sampler_T sampler_type;
  typedef typename Sampler_T::sample_type sample_type;
  typedef typename Sampler_T::value_type value_type;

  // Constructors --------------------------------------------------------------

  LinearFieldInterp(Sampler_T _sampler)
  : m_sampler(_sampler)
  {}

  // Main methods --------------------------------------------------------------

  //! sample at a voxel space location
  template<typename Accessor_T>
  __host__ __device__
  void sample(Accessor_T& ac, const float3& vsP, sample_type& dst) const;

  //! const access to sampler
  __host__ __device__
  const Sampler_T& getSampler()
  {
    return m_sampler;
  }

  template <typename TexAccessor_T>
  void bindTex(TexAccessor_T& tex) const;

  template <typename TexAccessor_T>
  void unbindTex(TexAccessor_T& tex) const;

  // Data members --------------------------------------------------------------
private:
  //! member vars
  const Sampler_T m_sampler;
};

//----------------------------------------------------------------------------//
// LinearFieldInterp implementations
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
//! sample at a voxel space location
template <typename Sampler_T>
template<typename Accessor_T>
void LinearFieldInterp<Sampler_T>::sample(Accessor_T& ac,
                                          const float3& vsP,
                                          sample_type& dst) const
{
  // Voxel centers are at .5 coordinates
  // NOTE: Don't use contToDisc for this, we're looking for sample
  // point locations, not coordinate shifts.
  float3 p = vsP - make_float3(0.5f);

  // Lower left corner
  int3 c1 = make_int3(int(floorf(p.x)), int(floorf(p.y)), int(floorf(p.z)));
  // Upper right corner
  int3 c2(c1 + make_int3(1));
  // C1 fractions
  float3 f1(make_float3(c2) - p);
  // C2 fraction
  float3 f2(make_float3(1.0f) - f1);

  // Clamp the indexing coordinates
  if( true )
  {
    c1.x = max(m_sampler.getDataWindowMin().x
         , min(c1.x, m_sampler.getDataWindowMax().x));
    c2.x = max(m_sampler.getDataWindowMin().x
         , min(c2.x, m_sampler.getDataWindowMax().x));
    c1.y = max(m_sampler.getDataWindowMin().y
         , min(c1.y, m_sampler.getDataWindowMax().y));
    c2.y = max(m_sampler.getDataWindowMin().y
         , min(c2.y, m_sampler.getDataWindowMax().y));
    c1.z = max(m_sampler.getDataWindowMin().z
         , min(c1.z, m_sampler.getDataWindowMax().z));
    c2.z = max(m_sampler.getDataWindowMin().z
         , min(c2.z, m_sampler.getDataWindowMax().z ) );
  }

  dst = (f1.x * (f1.y * (f1.z * m_sampler.getValue(ac, c1.x, c1.y, c1.z) +
                         f2.z * m_sampler.getValue(ac, c1.x, c1.y, c2.z)) +
                 f2.y * (f1.z * m_sampler.getValue(ac, c1.x, c2.y, c1.z) +
                         f2.z * m_sampler.getValue(ac, c1.x, c2.y, c2.z))) +
         f2.x * (f1.y * (f1.z * m_sampler.getValue(ac, c2.x, c1.y, c1.z) +
                         f2.z * m_sampler.getValue(ac, c2.x, c1.y, c2.z)) +
                 f2.y * (f1.z * m_sampler.getValue(ac, c2.x, c2.y, c1.z) +
                         f2.z * m_sampler.getValue(ac, c2.x, c2.y, c2.z))));
}

//----------------------------------------------------------------------------//

#ifdef NVCC
template< typename Sampler_T>
template <typename TexAccessor_T>
void LinearFieldInterp<Sampler_T>::bindTex(TexAccessor_T& tex) const
{
  cudaChannelFormatDesc channelDesc
  = cudaCreateChannelDesc< typename TexAccessor_T::cuda_tex_value_type >();
  tex.getTex().filterMode = cudaFilterModePoint;
  tex.getTex().normalized = false;
  tex.getTex().channelDesc = channelDesc;
  if (cudaBindTexture(NULL,
                      &tex.getTex(),
                      m_sampler.dataPtr(),
                      &channelDesc,
                      m_sampler.texMemSize())
      != (unsigned int) CUDA_SUCCESS)
  {
    throw std::runtime_error("failed to bind texture");
  }
}

//----------------------------------------------------------------------------//

template< typename Sampler_T>
template <typename TexAccessor_T>
void LinearFieldInterp<Sampler_T>::unbindTex(TexAccessor_T& tex) const
{
  cudaUnbindTexture(&tex.getTex());
}
#endif


FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
