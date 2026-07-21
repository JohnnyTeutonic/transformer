@echo off
REM Clean CUDA build script for Windows with Visual Studio solution
REM Can be run from "x64 Native Tools Command Prompt for VS 2026"
REM OR from regular PowerShell (will configure for VS solution)

echo ========================================
echo Transformer C++ CUDA Clean Build Script
echo ========================================
echo.

REM Check if nvcc is available
where nvcc.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CUDA compiler not found!
    echo.
    echo Make sure CUDA Toolkit is installed and in PATH
    echo.
    pause
    exit /b 1
)

echo Found CUDA compiler:
where nvcc.exe
echo.

REM Query GPU to verify CUDA 
echo Checking GPU...
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
echo.

REM Clean previous build completely
if exist build_vs_cuda (
    echo Removing old build directory...
    rmdir /s /q build_vs_cuda
)

REM Create build directory
echo Creating fresh build directory...
mkdir build_vs_cuda
cd build_vs_cuda

echo.
echo ========================================
echo Configuring CMake with CUDA (compute_89 for RTX 4060)...
echo Using Visual Studio generator for robust path handling...
echo ========================================
echo.

REM Configure with CUDA - use Visual Studio generator for better Windows compatibility
REM This avoids the path quoting issues with NMake
cmake .. -DCUDA_ENABLED=ON -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Trying fallback to Visual Studio 17 2022...
    cmake .. -DCUDA_ENABLED=ON -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ERROR: CMake configuration failed!
        echo Check the output above for errors.
        echo.
        cd ..
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo Building with MSBuild (Release)...
echo This may take 3-7 minutes for first build...
echo ========================================
echo.

REM Build using cmake --build (works with VS generator)
cmake --build . --config Release --parallel

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    echo Check the output above for errors.
    echo.
    cd ..
    pause
    exit /b 1
)

echo.
echo ========================================
echo ========================================
echo     BUILD SUCCESSFUL! 
echo ========================================
echo ========================================
echo.
echo Your GPU-accelerated transformer is ready!
echo.
echo Executables created:
dir /b *.exe 2>nul
echo.
echo To run training:
echo   train_wikitext.exe
echo.
echo To monitor GPU usage (open another terminal):
echo   nvidia-smi -l 1
echo.
echo Expected speedup: 50-400x faster than CPU!
echo.
pause

