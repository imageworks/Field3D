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

#ifndef _INCLUDED_Field3D_gpu_BufferCL_H_
#define _INCLUDED_Field3D_gpu_BufferCL_H_

#define __CL_ENABLE_EXCEPTIONS

#include "Field3D/gpu/CL/cl.hpp"
#include "Field3D/gpu/buffer/Buffer.h"
#include "Field3D/gpu/ns.h"

FIELD3D_GPU_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// IteratorCL
//----------------------------------------------------------------------------//
//! An iterator for OpenCL buffers
//----------------------------------------------------------------------------//

template< typename BufferPtr >
class IteratorCL
{
private:
  BufferPtr m_bufferPtr;
  int m_index;
  bool m_initialized;
public:
  IteratorCL(void)
  : m_index(-1)
  , m_initialized(false)
  , m_bufferPtr( NULL )
  {
    m_index = -1;
    m_initialized = false;
  }

  ~IteratorCL(void)
  {}

  static IteratorCL begin( BufferPtr vec )
  {
    IteratorCL i;

    if (!vec->empty())
    {
      i.m_index = 0;
    }

    i.m_bufferPtr = vec;
    i.m_initialized = true;
    return i;
  }

  static IteratorCL end( BufferPtr vec )
  {
    IteratorCL i;

    if (!vec->empty())
    {
      i.m_index = vec->size();
    }
    i.m_bufferPtr = vec;
    i.m_initialized = true;
    return i;
  }

  bool operator==(IteratorCL i)
  {
    return ((m_bufferPtr == i.m_bufferPtr)
        && (m_index == i.m_index)
        && (m_initialized == i.m_initialized));
  }

  bool operator!=(IteratorCL i)
  {
    return (!(*this==i));
  }

  IteratorCL operator+( size_t i )
  {
    IteratorCL result(*this);
    result.m_index += i;
    return result;
  }

  void operator++()
  {
    m_index++;
  }

  void operator++(int x)
  {
    m_index += x;
  }

  void operator--()
  {
    m_index--;
  }

  void operator--(int x)
  {
    m_index -= x;
  }

  int index()
  {
    return m_index;
  }

  BufferPtr vec() const
  {
    return m_bufferPtr;
  }

  BufferPtr vec()
  {
    return m_bufferPtr;
  }
};

//----------------------------------------------------------------------------//
// BufferCL
//----------------------------------------------------------------------------//
//! An OpenCL Buffer
//----------------------------------------------------------------------------//

template< typename Data_T >
struct BufferCL: public Buffer<Data_T>
{
  friend class IteratorCL< BufferCL* >;
  typedef Data_T value_type;
  typedef IteratorCL< BufferCL<Data_T>* > iterator;
  typedef IteratorCL< const BufferCL<Data_T>* > const_iterator;

  BufferCL( cl::Context& ctx, cl::CommandQueue& q )
  : m_context( ctx )
  , m_queue( q )
  , m_empty( true )
  {}

  iterator begin()
  {
    return iterator::begin(this);
  };

  const_iterator begin() const
  {
    return const_iterator::begin(this);
  };

  iterator end()
  {
    return iterator::end(this);
  };

  const_iterator end() const
  {
    return const_iterator::end(this);
  };

  bool empty() const
  {
    return m_empty;
  }

  void resize( size_t new_size )
  {
    cl_int err = CL_SUCCESS;
    try
    {
      m_data = cl::Buffer( m_context,
          CL_MEM_READ_ONLY,
          sizeof(Data_T) * new_size,
          NULL,
          &err );
    }
    catch ( cl::Error err )
    {
      std::cerr << "ERROR: " << err.what()
      << "(" << err.err() << ")" << std::endl;
    }
    m_size = new_size;
    m_empty = false;
  }

  size_t size() const
  {
    return m_size;
  }

  template< typename InputIterator >
  void setValue( int offset,
      InputIterator first,
      InputIterator last )
  {
    size_t n = std::distance( first, last );
    m_queue.enqueueWriteBuffer( m_data,
        CL_TRUE,
        sizeof(Data_T) * offset,
        sizeof(Data_T) * n,
        &( *first ) );
  }

  template< typename OutputIterator >
  void getValue(int first,
      int last,
      OutputIterator dst) const
  {
    int n = last - first;
    m_queue.enqueueReadBuffer( m_data,
        CL_TRUE,
        sizeof(Data_T) * first,
        sizeof(Data_T) * n,
        &( *dst ) );
    m_queue.flush();
  }

protected:
  cl::Context& m_context;
  cl::CommandQueue& m_queue;

  cl::Buffer m_data;
  bool m_empty;
  size_t m_size;
};

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
