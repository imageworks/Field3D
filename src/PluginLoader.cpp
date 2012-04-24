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

/*! \file PluginLoader.cpp
  \brief Contains implementations of plugin loading functions
*/

//----------------------------------------------------------------------------//

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <dirent.h>
#endif

#include <sys/types.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include <boost/tokenizer.hpp>

#include "ClassFactory.h"
#include "PluginLoader.h"

//----------------------------------------------------------------------------//

using namespace std;

//----------------------------------------------------------------------------//
// Local namespace
//----------------------------------------------------------------------------//

namespace {

  void tokenize(const std::string &str, const std::string &delimiters, 
                std::vector<std::string> &retItems)
  {
    typedef boost::tokenizer<boost::char_separator<char> > Tokenizer;
    boost::char_separator<char> sep(delimiters.c_str());
    Tokenizer tok(str, sep);
    for (Tokenizer::iterator i = tok.begin(); i != tok.end(); ++i) {
      retItems.push_back(*i);
    }
  }

}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

typedef int (*RegistrationFunc)(ClassFactory &);


//----------------------------------------------------------------------------//
// Static instances
//----------------------------------------------------------------------------//

std::vector<std::string> PluginLoader::ms_pluginsLoaded;

//----------------------------------------------------------------------------//
// PluginLoader implementations
//----------------------------------------------------------------------------//

static int filter(std::string &name, const char *suffix) 
{
  std::string delimiters = ".";
  std::vector <std::string> items;
  
  tokenize(name, delimiters, items);

  if (items.size() == 0) {
    return 0;
  }

  if (items[items.size() -1] == suffix) {
    return 1;
  }

  return 0;
}

//----------------------------------------------------------------------------//

bool getDirSos(std::vector<std::string> &sos, std::string &dir) 
{
#ifdef WIN32
  const char *ds = dir.c_str();
  HANDLE dirh;
  WIN32_FIND_DATAA  fd;

  dirh = FindFirstFileA(ds, &fd);
  while (dirh != INVALID_HANDLE_VALUE)
  {
    std::string name = fd.cFileName;
    std::string name_lower;

    std::transform(name.begin(), name.end(), name_lower.begin(), ::tolower);

    if (filter(name_lower, "so")) {
      name = dir + "/" + name;
      sos.push_back(name);
    }
    
    if (!FindNextFileA(dirh, &fd))
    {
      ::FindClose(dirh);
      break;
    }
  }
#else
  struct dirent *dirent;

  const char *ds = dir.c_str();
  DIR *dirfd = opendir(ds);
  if (!dirfd) {
    std::string er =
      "Field3D_plugin loader: could not open directory " + dir + "\n"; 
    //perror(er.c_str());
    return false;
  }
  
  dirent = readdir(dirfd);
  while (dirent != NULL) {

    std::string name = dirent->d_name;

    if (filter(name, "so")) {
      name = dir + "/" + name;
      sos.push_back(name);
    }

    dirent = readdir(dirfd);
  }

  closedir(dirfd);
#endif
  return true;
}

static RegistrationFunc findRegistrationFunc(const std::string &sofile)
{
#ifdef WIN32
  HMODULE handle = ::LoadLibraryA(sofile.c_str());
#else
  void *handle = dlopen(sofile.c_str(), RTLD_GLOBAL|RTLD_NOW);
#endif
  // Attempt to load .so file
  if (!handle) {
    std::string   errmsg;
#ifdef WIN32
    char *errstr;
//----------------------------------------------------------------------------//
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
                   NULL, GetLastError(), 
                   MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&errstr,
                   0, NULL);
    if (errstr) {
      errmsg = errstr;
      ::LocalFree(errstr);
    } else {
      errmsg = "Unknown error";
    }
#else
    errmsg = dlerror();
#endif

    std::cout <<
      "Field3D Plugin loader: failed to load plugin: " << errmsg << "\n";
    return 0;
  }

  // Determine plugin type by looking for one of:
  //   registerField3DPlugin()

  RegistrationFunc  fptr;

#ifdef WIN32
  fptr = (RegistrationFunc)GetProcAddress(handle,"registerField3DPlugin");
#else
  fptr = (RegistrationFunc)dlsym(handle,"registerField3DPlugin");
#endif
  if (!fptr) {
    char *debugEnvVar = getenv("FIELD3D_DEBUG");
    if (debugEnvVar) {
      // debug env var exist, so print warning
      Msg::print(Msg::SevWarning,
                 "Field3D plugin loader: failed to load "
                 "the symbol registerField3DPlugin");
    }
  }
  return fptr;
}


//----------------------------------------------------------------------------//

PluginLoader::PluginLoader()
{

}

//----------------------------------------------------------------------------//

PluginLoader::~PluginLoader()
{

}

//----------------------------------------------------------------------------//

void PluginLoader::loadPlugins()
{
  // Get environment variable
  char *cptr = getenv("FIELD3D_DSO_PATH");
  if (!cptr)
    return;

  std::string path = cptr;
  
  // Split paths
  std::vector<std::string> paths;
  const std::string delimiters = ":";
  
  tokenize(path, delimiters, paths);

  // For each path
  for (unsigned int i = 0; i < paths.size(); i++) {

    // List the contents of the directory
    std::vector<std::string> sos;
    if (!getDirSos(sos,paths[i])) {
      continue;
    }
    
    // Open each file
    for (unsigned int j = 0; j < sos.size(); j++) {
      std::string sofile = sos[j];
      
      //First check to see if a plugin of the same name has already been loaded
      const std::string pathDelimiter = "/";
      std::vector<std::string> pluginName;
      tokenize(sofile, pathDelimiter, pluginName);

      bool pluginAlreadyLoaded = false;

      for (unsigned int i = 0; i < ms_pluginsLoaded.size(); i++) {
        if (pluginName.size() > 0) {
          if (ms_pluginsLoaded[i] == pluginName[pluginName.size() - 1]) {
            //This plugin has been loaded so look for another one
            //std::cout << ms_pluginsLoaded[i] << " is already loaded\n";
            pluginAlreadyLoaded = true;
            break;
          } 
        }
      }
    
      if (pluginAlreadyLoaded) {
        continue;
      }
      
      if (pluginName.size() > 0) {
        std::string lastName = pluginName[pluginName.size() -1];
        ms_pluginsLoaded.push_back(lastName);
      }
      
      RegistrationFunc  fptr;

      fptr = findRegistrationFunc(sofile);
      if (fptr) {
        // Call the registration function
        int res = (*fptr)(ClassFactory::singleton());
        if (!res) {
          Msg::print(Msg::SevWarning,
                    "failed to init Field3D plugin " + sofile);
        } else {
          Msg::print("Initialized Field3D Plugin " + sofile);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------//

#if 0

bool PluginLoader::getDso(char *cptr, const char *dso,
                          std::string &dsoPath) 
{

  std::string path = cptr;
  
  // Split paths
  std::vector<std::string> paths;
  const std::string delimiters=":";

  Tokenize(path, paths, delimiters);
  
  // For each path
  for (unsigned int i=0; i < paths.size(); i++) {
    struct dirent *dirent;

    std::string dir = paths[i];
    const char *ds = dir.c_str();
    DIR *dirfd = opendir(ds);
    if (!dirfd) {
      continue;
    }
  
    dirent = readdir(dirfd);
    while (dirent != NULL) {

      std::string name = dirent->d_name;

      if (name  == dso) {
        dsoPath = dir + "/" + name;
        closedir(dirfd);
        return true;
      }

      dirent = readdir(dirfd);
    }
    closedir(dirfd);
  }

  
  return false;
  
}  

//----------------------------------------------------------------------------//

bool PluginLoader::resolveGlobalsForPlugins(const char *dso) {

  // Get environment variable
  char *cptr  = getenv("HOUDINI_DSO_PATH");
  if (!cptr)
    return false;

  std::string sofile;
  if (!getDso(cptr,dso,sofile)) {
    std::string dsostring = dso;
    Msg::print(dsostring + " is not in HOUDINI_DSO_PATH");
    return false;
  }
  
  void *handle = dlopen(sofile.c_str(), RTLD_GLOBAL|RTLD_NOW);

  if (!handle) {
    std::cout << "Field3D Plugin loader: failed to load Houdini plugin: "
              << sofile << " " << dlerror() << "\n";
    return false;
  }

#if 0 
  Msg::print("---------------------------------------------------------");
  Msg::print("Loaded " + sofile);
  Msg::print("---------------------------------------------------------");
#endif
  
  return true;

}

#endif

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
