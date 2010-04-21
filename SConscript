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

from BuildSupport import *

# ------------------------------------------------------------------------------

buildPath = buildDir()
installPath = installDir()
sharedLibPath = join(buildPath, field3DName)

# ------------------------------------------------------------------------------

Import("env")

libEnv = env.Clone()

setupEnv(libEnv)
setupLibBuildEnv(libEnv)

libEnv.VariantDir(buildPath, src, duplicate = 0)
files = Glob(join(buildPath, "*.cpp"))

# Declare library
dyLib = libEnv.SharedLibrary(sharedLibPath, files)
stLib = libEnv.Library(sharedLibPath, files)

# Set up install area
headerDir = join(installPath, include, field3DName)
headerFiles = Glob(join(export, "*.h"))
headerFiles.remove(File(join(export, typesHeader)))
headerInstall = libEnv.Install(headerDir, headerFiles)
stLibInstall = libEnv.Install(join(installPath, "lib"), [stLib])

# Set up the dynamic library properly on OSX
dylibInstall = None
if sys.platform == "darwin":
    dylibName = os.path.basename(str(dyLib[0]))
    print str(dyLib), dylibName
    dylibInstallPath = os.path.abspath(join(installPath, "lib", dylibName))
    # Creat the builder
    dylibEnv = env.Clone()
    dylibBuilder = Builder(action = setDylibInternalPath,
                           suffix = ".dylib", src_suffix = ".dylib")
    dylibEnv.Append(BUILDERS = {"SetDylibPath" : dylibBuilder})
    # Call builder
    dyLibInstall = dylibEnv.SetDylibPath(dylibInstallPath, dyLib)
else:
    dyLibInstall = libEnv.Install(join(installPath, "lib"), [dyLib])


# Bake in math library in Types.h
bakeEnv = env.Clone()
bakeBuilder = Builder(action = bakeMathLibHeader,
                      suffix = ".h", src_suffix = ".h")
bakeEnv.Append(BUILDERS = {"BakeMathLibHeader" : bakeBuilder})
bakeTarget = bakeEnv.BakeMathLibHeader(join(headerDir, typesHeader), 
                                       join(export, typesHeader))

# Default targets ---

libEnv.Default(dyLib)
libEnv.Default(stLib)
libEnv.Default(headerInstall)
libEnv.Default(stLibInstall)
libEnv.Default(dyLibInstall)
bakeEnv.Default(bakeTarget)

# ------------------------------------------------------------------------------

