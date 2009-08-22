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

