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

/*! \file PluginLoader.h
  \brief Contains the PluginLoader class.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_PluginLoader_H_
#define _INCLUDED_Field3D_PluginLoader_H_

#include <string>
#include <vector>

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// PluginLoader
//----------------------------------------------------------------------------//

/*! \class PluginLoader
  \brief This class provides methods for loading Field plugins from disk.
  \ingroup file_int
  \todo Look into adding maya-style single-plugin load and unload functions
*/
  
//----------------------------------------------------------------------------//

class PluginLoader
{
    
public:
    
  // Constructors --------------------------------------------------------------

  //! Default constructor
  PluginLoader();
  //! Destructor
  ~PluginLoader();
 
  // Main methods --------------------------------------------------------------

  //! Checks all paths in $FIELD3D_DSO_PATH and loads the plugins it finds
  static void loadPlugins();

#if 0
  //! Doesn't appear to be needed yet, but leave in the library just in case
  bool resolveGlobalsForPlugins(const char *dso);
  bool getDso(char *cptr,const char *dso, std::string &dsoPath);
#endif

private:
  
  // Private data members ------------------------------------------------------

  //! List of plugins loaded
  static std::vector<std::string> ms_pluginsLoaded;

};

//----------------------------------------------------------------------------//
// Utility functions 
//----------------------------------------------------------------------------//

bool getDirSos(std::vector<std::string> &sos, std::string &dir);

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
