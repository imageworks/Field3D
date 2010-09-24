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

#ifndef _INCLUDED_Field3D_gpu_FieldSamplerCuda_H_
#define _INCLUDED_Field3D_gpu_FieldSamplerCuda_H_

#include "Field3D/gpu/ns.h"

FIELD3D_GPU_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
//! Base class for sampling from a voxel grid
struct FieldSampler
{
	FieldSampler( Field3D::V3i _dataResolution, Field3D::Box3i _dataWindow )
	: m_dataResolution( make_int3(_dataResolution.x, _dataResolution.y, _dataResolution.z) )
	, ystride( _dataResolution.x )
	, zstride( _dataResolution.x * _dataResolution.y )
	, nxnynz( _dataResolution.x * _dataResolution.y * _dataResolution.z )
	, dataWindowMin( make_int3( _dataWindow.min.x, _dataWindow.min.y, _dataWindow.min.z ) )
	, dataWindowMax( make_int3( _dataWindow.max.x, _dataWindow.max.y, _dataWindow.max.z ) )
	{}

	//! get voxel resolution
	inline __host__  __device__
	const int3& dataResolution() const
	{
		return m_dataResolution;
	}

	/*! The total number of voxels in the data window as if the field was dense
	 * \note For sparse fields, not all voxels are allocated
	 */
	int dataWindowVoxelCount() const
	{
		return m_dataResolution.x * m_dataResolution.y * m_dataResolution.z;
	}

	/*! 1d to 3d index mapping as if the field was dense
	 * \note For sparse fields, the returned voxel index may not be for an allocated voxel
	 */
	inline __host__  __device__
	int3 denseVoxelIndex( int i ) const
	{
		int3 id;
		id.x = i % ystride + dataWindowMin.x;
		id.y = ( i / ystride ) % m_dataResolution.y + dataWindowMin.y;
		id.z = i / zstride + dataWindowMin.z;
		return id;
	}

	inline __host__  __device__
	const int3& getDataWindowMin() const
	{
		return dataWindowMin;
	}

	inline __host__  __device__
	const int3& getDataWindowMax() const
	{
		return dataWindowMax;
	}

	inline __host__   __device__
	void applyDataWindowOffset( int &i,
								int &j,
								int &k ) const
	{
		i -= dataWindowMin.x;
		j -= dataWindowMin.y;
		k -= dataWindowMin.z;
	}

protected:
	const int ystride;
	const int zstride;
	const int nxnynz;
	const int3 m_dataResolution;
	const int3 dataWindowMin;
	const int3 dataWindowMax;
};

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//



#endif // Include guard

