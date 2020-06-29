# ZfpCompression.jl

[![Build Status](https://travis-ci.com/milankl/ZfpCompression.jl.svg?branch=master)](https://travis-ci.com/milankl/ZfpCompression.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/milankl/ZfpCompression.jl?svg=true)](https://ci.appveyor.com/project/milankl/ZfpCompression-jl)

A Julia wrapper for the data compression library [zfp](https://github.com/LLNL/zfp),
written by P Lindstrom ([@lindstro](https://github.com/lindstro)).
From the [zfp documentation](https://zfp.readthedocs.io/en/release0.5.5/):

*zfp is an open source library for compressed numerical arrays that support high
throughput read and write random access. To achieve high compression ratios, zfp
generally uses lossy but optionally error-bounded compression. Bit-for-bit lossless
compression is also possible through one of zfp’s compression modes.*

*zfp works best for 2-4D arrays that exhibit spatial correlation, such as
continuous fields from physics simulations, images, regularly sampled terrain
surfaces, etc. Although zfp also provides a 1D array class that can be used for
1D signals such as audio, or even unstructured floating-point streams, the
compression scheme has not been well optimized for this use case, and rate and
quality may not be competitive with floating-point compressors designed s
pecifically for 1D streams.*

See the documentation, or [zfp's website](https://computing.llnl.gov/projects/floating-point-compression)
for more information.

Requires Julia v1.3 or newer

## Usage
### Lossless compression

1 to 4-D arrays of eltype `Int32,Int64,Float32,Float64` can be compressed calling
the `zfp_compress` function.

```julia
julia> using ZfpCompression

julia> A = rand(Float32,100,50);

julia> Ac = zfp_compress(A)
16952-element Array{UInt8,1}:
 0xfd
 0xe1
 0x80
 0x8d
    ⋮
```
which initializes the zfp compression, preallocates the bitstream used for
the compressed array and performs the compression. This is then returned
as `Array{UInt8,1}`.

Decompression requires knowledge about the type, size and shape of the uncompressed array.
This information is not stored in the compressed array. Therefore, we use `similar` here to
retain that information.

```julia
julia> Ad = similar(A);          # preallocate the decompressed array Ad
julia> zfp_decompress!(Ad,Ac)    # decompress compressed array into the decompressed array
```
In the lossless example from above the compression is reversible
```julia
julia> A == Ad
true
```

### Lossy compression

Lossy compression is achieved by specifying additional keyword arguments
for `zfp_compress`, which are `tol::Real`, `precision::Int`, and `rate::Real`.
If none are specified (as in the example above) the compression is lossless
(i.e. reversible). Lossy compression parameters are

- [`tol` defines the maximum absolute error that is tolerated.](https://zfp.readthedocs.io/en/release0.5.5/modes.html#fixed-accuracy-mode)
- [`precision` controls the precision, bounding a weak relative error](https://zfp.readthedocs.io/en/release0.5.5/modes.html#fixed-precision-mode), see this [FAQ](https://zfp.readthedocs.io/en/develop/faq.html#q-relerr)
- [`rate` fixes the bits used per value.](https://zfp.readthedocs.io/en/release0.5.5/modes.html#fixed-rate-mode)

Only **one** of `tol, precision` or `rate` should be specified. For further details
see the [zfp documentation](https://zfp.readthedocs.io/en/release0.5.5/modes.html#compression-modes).

If we can tolerate a maximum absolute error of 1e-5, we may do
```julia
julia> Ac = zfp_compress(A,tol=1e-3)
9048-element Array{UInt8,1}:
 0xff
 0x2c
 0x01
 0x1a
 0xf3
 0xbc
 0xea
 0xbb
 0xc6
 0xd4
    ⋮
```
which clearly reduces the size of the compressed array. It is **essential**
to provide the same compression parameters also for `zfp_decompress!`. Otherwise
the decompressed array is flawed.
```julia
julia> zfp_decompress!(A2,Ac,tol=1e-3)
julia> maximum(abs.(A2 - A))
0.00030493736f0
```
In this case the maximum absolute error is limited to about 3e-4.

## Installation

Not yet registered, hence

```julia
julia>] add https://github.com/milankl/ZfpCompression.jl
```
The C library is installed and built automatically.
