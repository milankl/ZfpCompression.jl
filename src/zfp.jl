using zfp_jll

# zfp constants
const HEADER_MAGIC = 1
const HEADER_META = 2
const HEADER_MODE = 4
const HEADER_FULL = 7

# zfp types declaration and size
zfp_type(::Type{Int32}) = 1
zfp_type(::Type{Int64}) = 2
zfp_type(::Type{Float32}) = 3
zfp_type(::Type{Float64}) = 4
zfp_type(::Type) = 0

@enum ZfpType begin
    zfp_type_none = 0
    zfp_type_int32 = 1
    zfp_type_int64 = 2
    zfp_type_float = 3
    zfp_type_double = 4
end

#TODO mapping from C to Julia seems to be inconsistent here
@enum ZfpExecPolicy begin
    zfp_exec_serial = 0     # serial execution (default)
    zfp_exec_omp = 1     # OpenMP multi-threaded execution
    zfp_exec_cuda = 2     # CUDA parallel execution
end

function zfp_type(i::Int)
    i == 1 && return Int32
    i == 2 && return Int64
    i == 3 && return Float32
    i == 4 && return Float64
    throw(TypeError())
end

"""Size of zfp types (Int32,Int64,Float32,Float64) in bytes."""
zfp_type_size(i::Int64) = ccall((:zfp_type_size, libzfp), Int64, (Int64,), i)
zfp_type_size(::Type{T}) where {T} = zfp_type_size(zfp_type(T))

# READ IN ARRAYS
struct ZfpField
    type::ZfpType
    nx::Csize_t
    ny::Csize_t
    nz::Csize_t
    nw::Csize_t
    sx::Cptrdiff_t
    sy::Cptrdiff_t
    sz::Cptrdiff_t
    sw::Cptrdiff_t
    data::Ptr{Cvoid}
end

ZfpField(field::Ptr) = unsafe_load(Ptr{ZfpField}(field))

"""Pass a 1-D array into a zfp_field in C, in Julia only as Ptr{Cvoid}."""
function zfp_field(A::AbstractArray{T,1}) where {T}
    n = length(A)
    field = ccall((:zfp_field_1d, libzfp), Ptr{Cvoid},
        (Ptr{Cvoid}, Clong, Cuint), A, zfp_type(T), n)
    sx = strides(A)[1]
    ccall((:zfp_field_set_stride_1d, libzfp), Cvoid, (Ptr{Cvoid}, Clong), field, sx)
    return field
end

"""Pass a 2-D array into a zfp_field in C, in Julia only as Ptr{Cvoid}."""
function zfp_field(A::AbstractArray{T,2}) where {T}
    nx, ny = size(A)
    field = ccall((:zfp_field_2d, libzfp), Ptr{Cvoid},
        (Ptr{Cvoid}, Clong, Cuint, Cuint), A, zfp_type(T), nx, ny)
    sx, sy = strides(A)
    ccall((:zfp_field_set_stride_2d, libzfp), Cvoid, (Ptr{Cvoid}, Clong, Clong), field, sx, sy)
    return field
end

"""Pass a 3-D array into a zfp_field in C, in Julia only as Ptr{Cvoid}."""
function zfp_field(A::AbstractArray{T,3}) where {T}
    nx, ny, nz = size(A)
    field = ccall((:zfp_field_3d, libzfp), Ptr{Cvoid},
        (Ptr{Cvoid}, Clong, Cuint, Cuint, Cuint), A, zfp_type(T), nx, ny, nz)
    sx, sy, sz = strides(A)
    ccall((:zfp_field_set_stride_3d, libzfp), Cvoid, (Ptr{Cvoid}, Clong, Clong, Clong),
        field, sx, sy, sz)
    return field
end

"""Pass a 4-D array into a zfp_field in C, in Julia only as Ptr{Cvoid}."""
function zfp_field(A::AbstractArray{T,4}) where {T}
    nx, ny, nz, nw = size(A)
    field = ccall((:zfp_field_4d, libzfp), Ptr{Cvoid},
        (Ptr{Cvoid}, Clong, Cuint, Cuint, Cuint, Cuint), A, zfp_type(T), nx, ny, nz, nw)
    sx, sy, sz, sw = strides(A)
    ccall((:zfp_field_set_stride_4d, libzfp), Cvoid, (Ptr{Cvoid}, Clong, Clong, Clong, Clong),
        field, sx, sy, sz, sw)
    return field
end

"""Allocate an empty zfp field."""
zfp_field_alloc() = ccall((:zfp_field_alloc, libzfp), Ptr{Cvoid}, (Ptr{Cvoid},), C_NULL)

"""Free the zfp field."""
function zfp_field_free(field::Ptr{Cvoid})
    ccall((:zfp_field_free, libzfp), Cvoid, (Ptr{Cvoid},), field)
end

"""Return type of a zfp field."""
zfp_field_type(field::Ptr{Cvoid}) = zfp_type(
    ccall((:zfp_field_type, libzfp), Int, (Ptr{Cvoid},), field))

"""Return the dimensionality (1,2,3 or 4) of the zfp field."""
zfp_field_dimensionality(field::Ptr{Cvoid}) = ccall((:zfp_field_dimensionality, libzfp),
    Int, (Ptr{Cvoid},), field)

"""Return the dimensionality (1,2,3 or 4) of the zfp field."""
zfp_field_pointer(field::Ptr{Cvoid}) = ccall((:zfp_field_pointer, libzfp),
    Ptr{Cvoid}, (Ptr{Cvoid},), field)

"""Associate a zfp field with a data pointer."""
zfp_field_set_pointer(field::Ptr{Cvoid}, ptr::Ptr) = ccall(
    (:zfp_field_set_pointer, libzfp), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), field, ptr)

# COMPRESSION OPTIONS
"""Open a stream (=object that holds the (de)compression settings) for zfp."""
zfp_stream_open() = ccall((:zfp_stream_open, libzfp), Ptr{Cvoid}, (Ptr{Cvoid},), C_NULL)

"""Open a stream (=object that holds the (de)compression settings) for zfp
from an exisiting bitstream used for storing the compressed array."""
function zfp_stream_open(bitstream::Ptr{Cvoid})
    ccall((:zfp_stream_open, libzfp), Ptr{Cvoid}, (Ptr{Cvoid},), bitstream)
end

"""Set bitrate (=bits per value) to set the compression rate directly.
Should not be larger than the bits per value of the uncompressed array."""
function zfp_stream_set_rate(stream::Ptr{Cvoid}, rate::Real, type::Type, dims::Integer)
    ccall((:zfp_stream_set_rate, libzfp), Cdouble,
        (Ptr{Cvoid}, Cdouble, Cuint, Cuint), stream, Float64(rate), zfp_type(type), dims)
end

"""Set the precision (≈ mantissa bits per value) for compression."""
function zfp_stream_set_precision(stream::Ptr{Cvoid}, precision::Integer)
    ccall((:zfp_stream_set_precision, libzfp), Cuint,
        (Ptr{Cvoid}, Cuint), stream, UInt(precision))
end

"""Set the accuracy (>=max abs error) for compression."""
function zfp_stream_set_accuracy(stream::Ptr{Cvoid}, tol::AbstractFloat)
    ccall((:zfp_stream_set_accuracy, libzfp), Cdouble,
        (Ptr{Cvoid}, Cdouble), stream, Float64(tol))
end

"""Set the zfp compression to lossless = reversible."""
function zfp_stream_set_reversible(stream::Ptr{Cvoid})
    ccall((:zfp_stream_set_reversible, libzfp), Cvoid, (Ptr{Cvoid},), stream)
end

"""Initialize a zfp stream C struct holding the compression settings.
    Only a Ptr{Cvoid} is returned to Julia."""
function zfp_stream(::Type{T},
    ndims::Int;
    tol::Real=0,
    precision::Real=0,
    rate::Int=0) where {T}

    stream = zfp_stream_open()

    # set the compression options
    if tol > 0
        zfp_stream_set_accuracy(stream, tol)
    elseif precision > 0
        zfp_stream_set_precision(stream, precision)
    elseif rate > 0
        maxrate = 8 * zfp_type_size(T)
        rate <= maxrate || @warn "Rate was set to $rate-bit > $maxrate-bit for type $T"
        zfp_stream_set_rate(stream, rate, T, ndims)
    else  # lossless
        zfp_stream_set_reversible(stream)
    end

    return stream
end

# BUFFER
"""Retrieve the max size of the compressed stream in bytes pre-compression."""
function zfp_stream_maximum_size(stream::Ptr{Cvoid}, field::Ptr{Cvoid})
    ccall((:zfp_stream_maximum_size, libzfp), Int,
        (Ptr{Cvoid}, Ptr{Cvoid}), stream, field)
end

"""Retrieve the actual size of the compressed stream in bytes post-compression."""
function zfp_stream_compressed_size(stream::Ptr{Cvoid})
    ccall((:zfp_stream_compressed_size, libzfp), Int, (Ptr{Cvoid},), stream)
end

"""Open a buffer (= pointer for a preallocated data array) into a bitstream
    for zfp to flush the compressed array into."""
function stream_open(buffer::Ptr, bufsize::Int)
    ccall((:stream_open, libzfp), Ptr{Cvoid}, (Ptr{Cvoid}, Int), buffer, bufsize)
end

"""Close the compressed bitstream."""
function stream_close(bitstream::Ptr{Cvoid})
    ccall((:stream_close, libzfp), Cvoid, (Ptr{Cvoid},), bitstream)
end

"""Connect the zfp stream (=object that holds the compression settings) to the
bitstream that will contain the compressed array."""
function zfp_stream_set_bit_stream(stream::Ptr{Cvoid}, bitstream::Ptr{Cvoid})
    ccall((:zfp_stream_set_bit_stream, libzfp), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}),
        stream, bitstream)
end

"""Rewind the data stream."""
function zfp_stream_rewind(stream::Ptr{Cvoid})
    ccall((:zfp_stream_rewind, libzfp), Cvoid, (Ptr{Cvoid},), stream)
end

"""Rewind the data stream."""
function zfp_stream_flush(stream::Ptr{Cvoid})
    ccall((:zfp_stream_flush, libzfp), Cvoid, (Ptr{Cvoid},), stream)
end

"""Close the zfp stream."""
function zfp_stream_close(stream::Ptr{Cvoid})
    ccall((:zfp_stream_close, libzfp), Cvoid, (Ptr{Cvoid},), stream)
end


"""Write the header into the stream, which includes the compression parameters."""
function zfp_write_header(stream::Ptr{Cvoid}, field::Ptr{Cvoid}, HEADER::Int)
    ccall((:zfp_write_header, libzfp), Int, (Ptr{Cvoid}, Ptr{Cvoid}, Cuint),
        stream, field, HEADER)
end

"""Write the header into the stream, which includes the compression parameters."""
function zfp_read_header(stream::Ptr{Cvoid}, field::Ptr{Cvoid}, HEADER::Int)
    ccall((:zfp_read_header, libzfp), Int, (Ptr{Cvoid}, Ptr{Cvoid}, Cuint),
        stream, field, HEADER)
end

# SET OPENMP NUMBER OF THREADS
"""Set the number of OpenMP threads for compression, also switches ZfpExecPolicy to OpenMP."""
function zfp_stream_set_omp_threads(stream::Ptr{Cvoid}, nthreads::Integer)
    success = ccall((:zfp_stream_set_omp_threads, libzfp), Cuint,
        (Ptr{Cvoid}, Cuint), stream, UInt(nthreads))
    success == 0 && throw("Enabling OpenMP failed.")
end

"""Return the current execution policy (serial/OpenMP/CUDA)."""
function zfp_stream_execution(stream::Ptr{Cvoid})
    ccall((:zfp_stream_execution, libzfp), ZfpExecPolicy, (Ptr{Cvoid},), stream)
end

"""Set the current execution policy to serial, OpenMP or CUDA."""
function zfp_stream_set_execution(stream::Ptr{Cvoid}, execution::Symbol)
    if execution == :serial
        exec_policy = ZfpExecPolicy(0)
    elseif execution == :openmp
        exec_policy = ZfpExecPolicy(1)
    elseif execution == :cuda
        # exec_policy = ZfpExecPolicy(2)
        throw("CUDA currently unsupported for ZfpCompression.jl.")
    else
        throw("Execution $execution unsupported.")
    end

    success = ccall((:zfp_stream_set_execution, libzfp), Int,
        (Ptr{Cvoid}, ZfpExecPolicy), stream, exec_policy)

    success == 0 && throw("Enabling $execution failed.")
end

# COMPRESSION AND DECOMPRESSION
"""Low-level C call to run the compression."""
function zfp_compress(stream::Ptr{Cvoid}, field::Ptr{Cvoid})
    ccall((:zfp_compress, libzfp), Int, (Ptr{Cvoid}, Ptr{Cvoid}), stream, field)
end

"""Low-level C call to run the decompression."""
function zfp_decompress(stream::Ptr{Cvoid}, field::Ptr{Cvoid})
    ccall((:zfp_decompress, libzfp), Int, (Ptr{Cvoid}, Ptr{Cvoid}), stream, field)
end

function zfp_compress(src::AbstractArray{T};
    write_header::Bool=true,
    nthreads::Int=1,
    kws...) where {T<:Union{Int32,Int64,Float32,Float64}}

    ndims = length(size(src))
    ndims in [1, 2, 3, 4] || throw(DimensionMismatch("Zfp compression only for 1-4D array."))

    zfpstream = zfp_stream(T, ndims; kws...)  # initialize the compression
    field = zfp_field(src)                  # turn src array into zfp field

    # preallocate the compressed array
    bufsize = zfp_stream_maximum_size(zfpstream, field)
    dest = Vector{UInt8}(undef, bufsize)             # allocate as UInt8
    bitstream = stream_open(pointer(dest), bufsize)  # turn array into zfp pointer
    zfp_stream_set_bit_stream(zfpstream, bitstream)  # connect bitstream pointer to zfp struct
    zfp_stream_rewind(zfpstream)

    # write header
    if write_header && zfp_write_header(zfpstream, field, HEADER_FULL) == 0
        throw(error("Writing header failed."))
    end

    # Enable OpenMP multi-threading
    if nthreads > 1
        zfp_stream_set_omp_threads(zfpstream, nthreads)
    end

    # perform compression
    success = zfp_compress(zfpstream, field)
    success == 0 && throw(error("Zfp compression failed."))
    zfp_stream_flush(zfpstream)
    compressed_size = zfp_stream_compressed_size(zfpstream)

    # free and close
    zfp_field_free(field)
    zfp_stream_close(zfpstream)
    stream_close(bitstream)

    return dest[1:compressed_size]
end

function zfp_decompress!(dest::AbstractArray{T},
    src::AbstractVector{UInt8};
    kws...) where {T<:Union{Int32,Int64,Float32,Float64}}

    ndims = length(size(dest))
    ndims in [1, 2, 3, 4] || throw(DimensionMismatch("Zfp compression only for 1-4D array."))

    zfpstream = zfp_stream(T, ndims; kws...)    # initialize decompression
    field = zfp_field(dest)             # turn destination array into zfp pointer

    # declare src as the bitstream to decompress and connect to zfp struct
    bufsize = zfp_stream_maximum_size(zfpstream, field)
    bitstream = stream_open(pointer(src), bufsize)
    zfp_stream_set_bit_stream(zfpstream, bitstream)
    zfp_stream_rewind(zfpstream)

    # perform decompression
    compressed_size = zfp_decompress(zfpstream, field)

    # free and close
    zfp_field_free(field)
    zfp_stream_close(zfpstream)
    stream_close(bitstream)

    # check for failure
    compressed_size == 0 && throw(error("Zfp decompression failed."))
    return nothing
end

function zfp_decompress(src::AbstractVector{UInt8})

    field = zfp_field_alloc()
    bitstream = stream_open(pointer(src), length(src))
    zfpstream = zfp_stream_open(bitstream)
    if zfp_read_header(zfpstream, field, HEADER_FULL) == 0
        throw(error("Reading header failed."))
    end

    # read header via ccalls
    T = zfp_field_type(field)
    ndims = zfp_field_dimensionality(field)

    # convert field pointer into Julia struct to access nx,ny,nz,nw
    ZF = ZfpField(field)
    n = filter(!=(0), (ZF.nx, ZF.ny, ZF.nz, ZF.nw))

    output = Array{T,ndims}(undef, n...)
    zfp_field_set_pointer(field, pointer(output))

    compressed_size = zfp_decompress(zfpstream, field)

    # free and close
    zfp_field_free(field)
    zfp_stream_close(zfpstream)
    stream_close(bitstream)

    # check for failure
    compressed_size == 0 && throw(error("Zfp decompression failed."))
    return output
end
