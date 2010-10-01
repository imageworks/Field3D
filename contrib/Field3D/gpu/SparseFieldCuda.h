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

#ifndef _INCLUDED_Field3D_gpu_SparseFieldCuda_H_
#define _INCLUDED_Field3D_gpu_SparseFieldCuda_H_

#ifdef NVCC
#error "This file is intended for GCC and isn't compatible with NVCC compiler due to Field3D includes"
#endif

#include "Field3D/SparseField.h"
#include "Field3D/Types.h"


#include "Field3D/gpu/ns.h"
#include "Field3D/gpu/FieldInterpCuda.h"

#include <thrust/device_vector.h>

FIELD3D_GPU_NAMESPACE_OPEN

// forward declarations
template< typename A, typename B > struct SparseFieldSampler;
template< typename A > struct LinearFieldInterp;

//----------------------------------------------------------------------------//
/*! A functor to determine whether a block is required
 * \note A user defined BlockFunctor can allow partial loading onto GPU
 *       eg via a bounding box or frustum check
 */
struct BlockFunctor
{
	//! determine whether block <<bi,bj,bk>> is required
	template< typename FIELD >
	bool operator()( FIELD& f, int bi, int bj, int bk ) const;
};

//----------------------------------------------------------------------------//
//! A concrete BlockFunctor that indicates all blocks are required
struct EveryBlockFunctor : BlockFunctor
{
	template< typename FIELD >
	bool operator()( FIELD& f, int bi, int bj, int bk ) const
	{
		return true;
	}
};

//----------------------------------------------------------------------------//
//! Cuda layer for SparseFields
template< typename Data_T >
struct SparseFieldCuda : public SparseField< Data_T >
{
	typedef typename GpuFieldTraits< Data_T >::value_type value_type;
	typedef typename GpuFieldTraits< Data_T >::cuda_value_type cuda_value_type;
	typedef typename GpuFieldTraits< Data_T >::interpolation_type interpolation_type;
	typedef SparseFieldSampler< value_type, interpolation_type > sampler_type;
	typedef LinearFieldInterp< sampler_type > linear_interp_type;
	typedef SparseField< Data_T > base;

	/*! host to device transfer for blocks
	 *  \note There is an index table and data buffer, with the data buffer
	 *        containing both empty values and block data
	 */
	template< typename IntBuffer, typename Buffer, typename BlockFunctor >
	void hostToDevice( IntBuffer& blockTable, Buffer& buffer, BlockFunctor& bf ) const
	{
		int required_block_count = requiredBlockCount( bf );
		int gpu_cache_element_count = deviceElementsRequired( required_block_count );
		/*
		std::cout << "mem required: " << gpu_cache_element_count * sizeof(Data_T) / 1000000 << "Mb for " << gpu_cache_element_count << " elements "
				<< std::endl;
*/
		const V3i br = base::blockRes();
		const int block_count = br.x * br.y * br.z; // total block count

		buffer.resize( gpu_cache_element_count );

		std::vector< Data_T > host_empty_values( block_count );
		std::vector< int > host_block_table( block_count, -1 ); // initialize to empty value

#if 0
		// for an allocated block, what's the lower left corner voxel index
		std::vector< int > host_reverse_block_table( required_block_count * 3 );
#endif

		int i = 0;
		int allocated_block_index = 0;

		// allow room for empty values
		int write_index = block_count;
		int bs = base::blockSize();
		int block_element_count = bs * bs * bs;

		typename base::block_iterator bi( base::blockBegin() ), be( base::blockEnd() );
		for ( ; bi != be ; ++bi, ++i )
		{
			const typename base::Block& block = base::m_blocks[ base::blockId( bi.x, bi.y, bi.z ) ];
			host_empty_values[ i ] = block.emptyValue;
			if( block.isAllocated && bf( *this, bi.x, bi.y, bi.z ) ){
				host_block_table[ i ] = write_index;
				assert( block.data.size() == block_element_count );
				Field3D::Gpu::copy( block.data.begin(), block.data.end(), buffer.begin() + write_index );
				write_index += block_element_count;

#if 0
				const Box3i& bounds = bi.blockBoundingBox();
				host_reverse_block_table[ allocated_block_index * 3 ] = bounds.min.x;
				host_reverse_block_table[ allocated_block_index * 3 + 1 ] = bounds.min.y;
				host_reverse_block_table[ allocated_block_index * 3 + 2 ] = bounds.min.z;
				++allocated_block_index;
#endif
			}
		}
		assert( i == block_count );

		// host -> device for empty values
		Field3D::Gpu::copy( host_empty_values.begin(), host_empty_values.end(), buffer.begin() );

		// host->device for block table;
		blockTable.resize( host_block_table.size() );
		Field3D::Gpu::copy( host_block_table.begin(), host_block_table.end(), blockTable.begin() );

		m_texMemSize = gpu_cache_element_count * sizeof( Data_T );
		m_blockCount = required_block_count;
	}

	template< typename IntBuffer, typename Buffer >
	void hostToDevice( IntBuffer& blockTable, Buffer& buffer ) const
	{
		EveryBlockFunctor f;
		return hostToDevice( blockTable, buffer, f );
	}

	//----------------------------------------------------------------------------//
	//! manufacture an interpolator for device using a BlockFunctor
	template< typename BlockFunctor >
	boost::shared_ptr< linear_interp_type > getLinearInterpolatorDevice( BlockFunctor& bf ) const
	{
		hostToDevice( m_blockTableCuda, m_bufferCuda, bf );

		return boost::shared_ptr< linear_interp_type >( new linear_interp_type( sampler_type( base::dataResolution(), base::dataWindow(),
				m_blockCount, base::blockOrder(), base::blockRes(), thrust::raw_pointer_cast( &m_blockTableCuda[ 0 ] ),
				(cuda_value_type*) thrust::raw_pointer_cast( &m_bufferCuda[ 0 ] ), m_texMemSize ) ) );
	}

	//----------------------------------------------------------------------------//
	//! manufacture an interpolator for device
	boost::shared_ptr< linear_interp_type > getLinearInterpolatorDevice() const
	{
		EveryBlockFunctor f;
		return getLinearInterpolatorDevice( f );
	}

	//----------------------------------------------------------------------------//
	//! manufacture an interpolator for host
	boost::shared_ptr< linear_interp_type > getLinearInterpolatorHost() const
	{
		std::cerr << "SparseFieldCuda::getLinearInterpolatorHost() not implemented yet\n";
		return boost::shared_ptr< linear_interp_type >();
	}

	//----------------------------------------------------------------------------//
	//! Number of allocated blocks that meet BlockFunctor requirements
	template< typename BlockFunctor >
	int requiredBlockCount( const BlockFunctor& bf ) const
	{
		typename base::block_iterator bi( base::blockBegin() ), be( base::blockEnd() );
		int result = 0;
		for ( ; bi != be ; ++bi )
		{
			if( base::blockIsAllocated( bi.x, bi.y, bi.z )
					&& bf( *this, bi.x, bi.y, bi.z ) ){
				++result;
			}
		}
		return result;
	}

	//----------------------------------------------------------------------------//
	//! Project the required number of Data_T elements for block cache + emptyValues
	int deviceElementsRequired( int required_block_count ) const
	{
		int result = 0;

		// sparse block emptyValues
		V3i br = base::blockRes();
		result += br.x * br.y * br.z;

		// block data
		int bs = base::blockSize();
		result += required_block_count * bs * bs * bs;

		return result;
	}

private:
	mutable BufferCuda< int > m_blockTableCuda;
	mutable BufferCuda< Data_T > m_bufferCuda;
	mutable int m_blockCount;
	mutable int m_texMemSize;
};

FIELD3D_GPU_NAMESPACE_HEADER_CLOSE

#endif // Include guard
