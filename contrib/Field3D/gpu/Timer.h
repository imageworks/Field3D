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

#ifndef _INCLUDED_Field3D_gpu_Timer_H_
#define _INCLUDED_Field3D_gpu_Timer_H_

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <cuda_runtime_api.h>

#include "Field3D/gpu/ns.h"

FIELD3D_GPU_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
//! Wall clock timer for profiling CPU code
struct CpuTimer
{
	CpuTimer() :
		m_start( boost::posix_time::microsec_clock::local_time() )
	{}

	//! elapsed time in seconds
	float elapsed() const
	{
		boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
		boost::posix_time::time_duration diff = now - m_start;
		return diff.total_milliseconds() / 1000.0f;
	}

private:
	boost::posix_time::ptime m_start;
};

//----------------------------------------------------------------------------//
//! Wall clock timer for profiling GPU code
struct GpuTimer
{
	GpuTimer()
	{
		m_start = new cudaEvent_t;
		m_stop = new cudaEvent_t;

		cudaEventCreate( (cudaEvent_t *) m_start );
		cudaEventCreate( (cudaEvent_t *) m_stop );

		// start the timer
		cudaEventRecord( *( (cudaEvent_t *) m_start ), 0 );
	}

	~GpuTimer()
	{
		cudaEventDestroy( *( (cudaEvent_t *) m_start ) );
		cudaEventDestroy( *( (cudaEvent_t *) m_stop ) );

		delete (cudaEvent_t *) m_start;
		delete (cudaEvent_t *) m_stop;
	}

	//! elapsed time in seconds
	float elapsed()
	{
		// stop the timer
		cudaEventRecord( *( (cudaEvent_t *) m_stop ), 0 );

		cudaEventSynchronize( *( (cudaEvent_t *) m_stop ) );
		float ms;
		cudaEventElapsedTime( &ms, *( (cudaEvent_t *) m_start ), *( (cudaEvent_t *) m_stop ) );
		return ms / 1000.0f;
	}

private:
	void *m_start;
	void *m_stop;
};

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
