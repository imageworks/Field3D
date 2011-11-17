#
# Copyright (c) 2009 Sony Pictures Imageworks Inc. 
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the
# distribution. Neither the name of Sony Pictures Imageworks nor the
# names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written
# permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

# ------------------------------------------------------------------------------
# Contains various helper functions for the SCons build system
# ------------------------------------------------------------------------------

import os
import sys

from SCons.Script import *

from os.path import join

# ------------------------------------------------------------------------------
# Strings
# ------------------------------------------------------------------------------

field3DName    = "Field3D"

buildDirPath   = "build"
installDirPath = "install"

release        = "release"
debug          = "debug"
export         = "export"
include        = "include"
src            = "src"

stdMathHeader  = "StdMathLib.h"

siteFile       = "Site.py"

typesHeader    = "Types.h"

arch32         = "m32"
arch64         = "m64"

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------

systemIncludePaths = {
    "darwin" : { arch32 : ["/usr/local/include",
                           "/opt/local/include"],
                 arch64 : ["/usr/local/include",
                           "/opt/local/include"]},
    "linux2" : { arch32 : ["/usr/local/include"],
                 arch64 : ["/usr/local64/include"]}
}

systemLibPaths = {
    "darwin" : { arch32 : ["/usr/local/lib",
                           "/opt/local/lib"],
                 arch64 : ["/usr/local/lib",
                           "/opt/local/lib"]},
    "linux2" : { arch32 : ["/usr/local/lib"],
                 arch64 : ["/usr/local64/lib"]}
}

systemLibs = {
    "darwin" : [],
    "linux2" : ["dl"]
    }

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

def isDebugBuild():
    return ARGUMENTS.get('debug', 0)

# ------------------------------------------------------------------------------

def red(s):
    return "\033[1m" + s + "\033[0m"

# ------------------------------------------------------------------------------

def architectureStr():
    if ARGUMENTS.get('do64', 1):
        return arch64
    else:
        return arch32

# ------------------------------------------------------------------------------

def buildDir():
    basePath = join(buildDirPath, sys.platform, architectureStr())
    if isDebugBuild():
        return join(basePath, debug)
    else:
        return join(basePath, release)

# ------------------------------------------------------------------------------

def installDir():
    basePath = join(installDirPath, sys.platform, architectureStr())
    if isDebugBuild():
        return join(basePath, debug)
    else:
        return join(basePath, release)

# ------------------------------------------------------------------------------

def getMathHeader():
    if os.path.exists(siteFile):
        import Site
        if hasattr(Site, "mathInc"):
            return Site.mathInc
    return stdMathHeader

# ------------------------------------------------------------------------------

def setupLibBuildEnv(env, pathToRoot = "."):
    # Project headers
    env.Append(CPPPATH = [join(pathToRoot, export)])
    # Check if Site.py exists
    siteExists = False
    if os.path.exists(join(pathToRoot, siteFile)):
        sys.path.append(pathToRoot)
        import Site
        siteExists = True
    if siteExists and \
           hasattr(Site, "mathInc") and \
           hasattr(Site, "mathIncPaths") and \
           hasattr(Site, "mathLibs") and \
           hasattr(Site, "mathLibPaths"):
        mathIncStr = '\\"' + Site.mathInc + '\\"'
        env.Append(CPPDEFINES =
                   {"FIELD3D_CUSTOM_MATH_LIB" : None})
        env.Append(CPPDEFINES =
                   {"FIELD3D_MATH_LIB_INCLUDE" : mathIncStr})

# ------------------------------------------------------------------------------

def setupEnv(env, pathToRoot = "."):
    baseIncludePaths = systemIncludePaths[sys.platform][architectureStr()]
    baseLibPaths = systemLibPaths[sys.platform][architectureStr()]
    baseLibs = systemLibs[sys.platform]
    # System include paths
    env.Append(CPPPATH = baseIncludePaths)
    # System lib paths
    env.Append(LIBPATH = baseLibPaths)
    # System libs
    env.Append(LIBS = baseLibs)
    # Check if Site.py exists
    siteExists = False
    if os.path.exists(join(pathToRoot, siteFile)):
        sys.path.append(pathToRoot)
        import Site
        siteExists = True
    # Choose math library
    if siteExists and \
           hasattr(Site, "mathInc") and \
           hasattr(Site, "mathIncPaths") and \
           hasattr(Site, "mathLibs") and \
           hasattr(Site, "mathLibPaths"):
        env.Append(CPPPATH = Site.mathIncPaths)
        env.Append(LIBS = Site.mathLibs)
        env.Append(LIBPATH = Site.mathLibPaths)
        env.Append(RPATH = Site.mathLibPaths)
    else:
        for path in baseIncludePaths:
            env.Append(CPPPATH = join(path, "OpenEXR"))
        env.Append(LIBS = ["Half"])
        env.Append(LIBS = ["Iex"])
        env.Append(LIBS = ["Imath"])
    # Add in site-specific paths
    if siteExists and hasattr(Site, "incPaths"):
        env.AppendUnique(CPPPATH = Site.incPaths)
    if siteExists and hasattr(Site, "libPaths"):
        env.AppendUnique(LIBPATH = Site.libPaths)
        env.AppendUnique(RPATH = Site.libPaths)
    # Custom namespace
    if siteExists and hasattr(Site, "extraNamespace"):
        namespaceDict = {"FIELD3D_EXTRA_NAMESPACE" : Site.extraNamespace}
        env.AppendUnique(CPPDEFINES = namespaceDict)
    # System libs
    env.Append(LIBS = ["z", "pthread"])
    # Hdf5 lib
    env.Append(LIBS = ["hdf5"])
    # Boost threads
    if siteExists and hasattr(Site, "boostThreadLib"):
        env.Append(LIBS = [Site.boostThreadLib])
    else:
        env.Append(LIBS = ["boost_thread-mt"])
    # Compile flags
    if isDebugBuild():
        env.Append(CCFLAGS = ["-g"])
    else:
        env.Append(CCFLAGS = ["-g", "-O3"])
    env.Append(CCFLAGS = ["-Wall"])
    # Set number of jobs to use
    env.SetOption('num_jobs', numCPUs())
    # 64 bit setup
    if architectureStr() == arch64:
        env.Append(CCFLAGS = ["-m64"])
        env.Append(LINKFLAGS = ["-m64"])
    else:
        env.Append(CCFLAGS = ["-m32"])
        env.Append(LINKFLAGS = ["-m32"])
    # Prettify SCons output
    if ARGUMENTS.get("verbose", 0) != "1":
        env["ARCOMSTR"] = "AR $TARGET"
        env["CXXCOMSTR"] = "Compiling " + red("$TARGET")
        env["SHCXXCOMSTR"] = "Compiling " + red("$TARGET")
        env["LDMODULECOMSTR"] = "Compiling " + red("$TARGET")
        env["LINKCOMSTR"] = "Linking " + red("$TARGET")
        env["SHLINKCOMSTR"] = "Linking " + red("$TARGET")
        env["INSTALLSTR"] = "Installing " + red("$TARGET")

# ------------------------------------------------------------------------------

def addField3DInstall(env, pathToRoot):
    env.Prepend(CPPPATH = [join(pathToRoot, installDir(), "include")])
    env.Prepend(LIBS = [field3DName])
    env.Prepend(LIBPATH = [join(pathToRoot, installDir(), "lib")])
    env.Prepend(RPATH = [join(pathToRoot, installDir(), "lib")])

# ------------------------------------------------------------------------------

def numCPUs():
    if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"):
        nCPUs = os.sysconf("SC_NPROCESSORS_ONLN")
        if isinstance(nCPUs, int) and nCPUs > 0:
            return nCPUs
    else: 
        return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    if os.environ.has_key("NUMBER_OF_PROCESSORS"):
         nCPUs = int(os.environ["NUMBER_OF_PROCESSORS"]);
         if nCPUs > 0:
             return nCPUs
    return 1

# ------------------------------------------------------------------------------

def setDylibInternalPath(target, source, env):
    # Copy the library file
    srcName = str(source[0])
    tgtName = str(target[0])
    Execute(Copy(tgtName, srcName))
    # Then run install_name_tool
    cmd = "install_name_tool "
    cmd += "-id " + os.path.abspath(tgtName) + "  "
    cmd += tgtName
    print cmd
    os.system(cmd)

# ------------------------------------------------------------------------------

def bakeMathLibHeader(target, source, env):
    if len(target) != 1 or len(source) != 1:
        print "Wrong number of arguments to bakeTypesIncludeFile"
        return
    out = open(str(target[0]), "w")
    inFile = open(str(source[0]))
    skip = False
    for line in inFile.readlines():
        if not skip and "#ifdef FIELD3D_CUSTOM_MATH_LIB" in line:
            skip = True
            newLine = '#include "' + getMathHeader() + '"\n'
            out.writelines(newLine)
        if not skip:
            out.writelines(line)
        if skip and "#endif" in line:
            skip = False

# ------------------------------------------------------------------------------
