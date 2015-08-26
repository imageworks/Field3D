//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2011 Sony Pictures Imageworks Inc
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

/*! \file FieldCache.h
  \brief Contains the FieldCache class
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_FieldCache_H_
#define _INCLUDED_Field3D_FieldCache_H_

//----------------------------------------------------------------------------//

#include <boost/thread/mutex.hpp>
#include <boost/foreach.hpp>

#include "Field.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// FieldCache
//----------------------------------------------------------------------------//

/* \class FieldCache

   This class is used by Field3DInputFile::readField() to see if the 
   field being loaded already exists in memory. It uses the weak pointer
   system in RefBase to check if a previously loaded field still resides
   in memory. If it is, then readField() returns a pointer rather than 
   reading the data again from disk.

   \note FieldCache does not increment the reference count of cached fields,
   so objects will be deallocated naturally.
 */

//----------------------------------------------------------------------------//

template <typename Data_T>
class FieldCache
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef Field<Data_T> Field_T;
  typedef typename Field_T::Ptr FieldPtr;
  typedef typename Field_T::WeakPtr WeakPtr;
  typedef std::pair<WeakPtr, Field_T*> CacheEntry;
  typedef std::map<std::string, CacheEntry> Cache;

  // Access to singleton -------------------------------------------------------

  //! Returns a reference to the FieldCache singleton
  static FieldCache& singleton();

  // Main methods --------------------------------------------------------------

  //! Checks the cache for a previously loaded field.
  //! \return A pointer to the cached field, if it is still in memory. Null
  //! otherwise.
  FieldPtr getCachedField(const std::string &filename,
                          const std::string &layerPath);
  //! Adds the given field to the cache. 
  void cacheField(FieldPtr field, const std::string &filename,
                  const std::string &layerPath);
  //! Returns the memory use of all currently loaded fields
  long long int memSize() const;

private:

  // Utility functions --------------------------------------------------------

  //! Constructs the cache key for a given file and layer path.
  std::string key(const std::string &filename,
                  const std::string &layerPath);

  // Data members -------------------------------------------------------------

  //! The cache itself. Maps a 'key' to a weak pointer and a raw pointer.
  Cache m_cache;
  //! The singleton instance
  static FieldCache *ms_singleton;
  //! Mutex to prevent multiple allocaation of the singleton
  static boost::mutex ms_creationMutex;
  //! Mutex to prevent reading from and writing to the cache concurrently.
  static boost::mutex ms_accessMutex;
};

//----------------------------------------------------------------------------//
// Implementations
//----------------------------------------------------------------------------//

template <typename Data_T>
FieldCache<Data_T>& FieldCache<Data_T>::singleton()
{
  boost::mutex::scoped_lock lock(ms_creationMutex);
  if (!ms_singleton) {
    ms_singleton = new FieldCache;
  }
  return *ms_singleton;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
typename FieldCache<Data_T>::FieldPtr 
FieldCache<Data_T>::getCachedField(const std::string &filename,
                                   const std::string &layerPath)
{
  boost::mutex::scoped_lock lock(ms_accessMutex);
  // First see if the request has ever been processed
  typename Cache::iterator i = m_cache.find(key(filename, layerPath));
  if (i == m_cache.end()) {
    return FieldPtr();
  }
  // Next, check if all weak_ptrs are valid
  CacheEntry &entry = i->second;
  WeakPtr weakPtr = entry.first;
  if (weakPtr.expired()) {
    return FieldPtr();
  }
  return FieldPtr(entry.second);
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void FieldCache<Data_T>::cacheField(FieldPtr field, const std::string &filename,
                                    const std::string &layerPath)
{
  boost::mutex::scoped_lock lock(ms_accessMutex);
  m_cache[key(filename, layerPath)] = 
    std::make_pair(field->weakPtr(), field.get());
}

//----------------------------------------------------------------------------//

template <typename Data_T>
long long int FieldCache<Data_T>::memSize() const
{
  boost::mutex::scoped_lock lock(ms_accessMutex);

  long long int memSize = 0;

  BOOST_FOREACH (const typename Cache::value_type &i, m_cache) {
    // Check if pointer is valid
    WeakPtr weakPtr = i.second.first;
    if (weakPtr.expired()) {
      continue;
    } else {
      // If valid, accumulate memory
      memSize += i.second.second->memSize();
    }
  }

  return memSize;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
std::string FieldCache<Data_T>::key(const std::string &filename,
                                    const std::string &layerPath)
{ 
  return filename + "/" + layerPath; 
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif
