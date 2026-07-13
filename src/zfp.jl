using zfp_jll

macro gc_safe(expr)
    quote
        gc_state = ccall(:jl_gc_safe_enter, Int8, ())
        try
            $(esc(expr))
        finally
            ccall(:jl_gc_safe_leave, Cvoid, (Int8,), gc_state)
        end
    end
end

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
    throw(ArgumentError("Unsupported zfp type id $i."))
end

"""Size of zfp types (Int32,Int64,Float32,Float64) in bytes."""
zfp_type_size(i::Int64) = ccall((:zfp_type_size, libzfp), Csize_t, (Cint,), i)
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
        (Ptr{Cvoid}, Cint, Csize_t), A, zfp_type(T), n)
    sx = strides(A)[1]
    ccall((:zfp_field_set_stride_1d, libzfp), Cvoid, (Ptr{Cvoid}, Cptrdiff_t), field, sx)
    return field
end

"""Pass a 2-D array into a zfp_field in C, in Julia only as Ptr{Cvoid}."""
function zfp_field(A::AbstractArray{T,2}) where {T}
    nx, ny = size(A)
    field = ccall((:zfp_field_2d, libzfp), Ptr{Cvoid},
        (Ptr{Cvoid}, Cint, Csize_t, Csize_t), A, zfp_type(T), nx, ny)
    sx, sy = strides(A)
    ccall((:zfp_field_set_stride_2d, libzfp), Cvoid, (Ptr{Cvoid}, Cptrdiff_t, Cptrdiff_t),
        field, sx, sy)
    return field
end

"""Pass a 3-D array into a zfp_field in C, in Julia only as Ptr{Cvoid}."""
function zfp_field(A::AbstractArray{T,3}) where {T}
    nx, ny, nz = size(A)
    field = ccall((:zfp_field_3d, libzfp), Ptr{Cvoid},
        (Ptr{Cvoid}, Cint, Csize_t, Csize_t, Csize_t), A, zfp_type(T), nx, ny, nz)
    sx, sy, sz = strides(A)
    ccall((:zfp_field_set_stride_3d, libzfp), Cvoid,
        (Ptr{Cvoid}, Cptrdiff_t, Cptrdiff_t, Cptrdiff_t),
        field, sx, sy, sz)
    return field
end

"""Pass a 4-D array into a zfp_field in C, in Julia only as Ptr{Cvoid}."""
function zfp_field(A::AbstractArray{T,4}) where {T}
    nx, ny, nz, nw = size(A)
    field = ccall((:zfp_field_4d, libzfp), Ptr{Cvoid},
        (Ptr{Cvoid}, Cint, Csize_t, Csize_t, Csize_t, Csize_t), A, zfp_type(T), nx, ny, nz, nw)
    sx, sy, sz, sw = strides(A)
    ccall((:zfp_field_set_stride_4d, libzfp), Cvoid,
        (Ptr{Cvoid}, Cptrdiff_t, Cptrdiff_t, Cptrdiff_t, Cptrdiff_t),
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
Should not be larger than the bits per value of the uncompressed array.
`align` word-aligns blocks, e.g. for write random access."""
function zfp_stream_set_rate(stream::Ptr{Cvoid}, rate::Real, type::Type, dims::Integer,
                             align::Bool=false)
    ccall((:zfp_stream_set_rate, libzfp), Cdouble,
        (Ptr{Cvoid}, Cdouble, Cuint, Cuint, Cint),
        stream, Float64(rate), zfp_type(type), dims, align)
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

"""Apply the compression mode (tol > precision > rate > lossless) to an
already-open zfp stream."""
function zfp_stream_set_mode!(stream::Ptr{Cvoid}, ::Type{T}, ndims::Int;
    tol::Real=0, precision::Real=0, rate::Real=0) where {T}

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

"""Initialize a zfp stream C struct holding the compression settings.
    Only a Ptr{Cvoid} is returned to Julia."""
function zfp_stream(::Type{T}, ndims::Int; kws...) where {T}
    stream = zfp_stream_open()
    zfp_stream_set_mode!(stream, T, ndims; kws...)
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

"""Open a byte buffer into a bitstream, checking if the buffer is not
strided. zfp reads the buffer through a raw pointer so it must be contiguous."""
function stream_open(buffer::AbstractVector{UInt8})
    if !(buffer isa StridedVector) || stride(buffer, 1) != 1
        throw(ArgumentError("The compressed buffer must be contiguous."))
    end

    return stream_open(pointer(buffer), length(buffer))
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

"""Flush the data stream, i.e. write any buffered bits out to it."""
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
    success == 0 && throw(ErrorException("Enabling OpenMP failed."))
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
        throw(ArgumentError("CUDA currently unsupported for ZfpCompression.jl."))
    else
        throw(ArgumentError("Execution $execution unsupported."))
    end

    success = ccall((:zfp_stream_set_execution, libzfp), Int,
        (Ptr{Cvoid}, ZfpExecPolicy), stream, exec_policy)

    success == 0 && throw(ErrorException("Enabling $execution failed."))
end

# UTILITY FUNCTIONS: promote low-bit ints to Int32, demote Int32 to low-bit ints.
# The underlying C functions process exactly 4^dims values per call; we tile the
# array with the largest dims that fits, falling through to dims=0 (scalar).

const _LowBitInt = Union{Int8, UInt8, Int16, UInt16}

for (T, promote_name, demote_name) in (
        (Int8,   :zfp_promote_int8_to_int32,   :zfp_demote_int32_to_int8),
        (UInt8,  :zfp_promote_uint8_to_int32,  :zfp_demote_int32_to_uint8),
        (Int16,  :zfp_promote_int16_to_int32,  :zfp_demote_int32_to_int16),
        (UInt16, :zfp_promote_uint16_to_int32, :zfp_demote_int32_to_uint16))

    @eval function zfp_promote!(dest::DenseArray{Int32}, src::DenseArray{$T})
        size(dest) == size(src) || throw(DimensionMismatch(
            "size(dest) = $(size(dest)) does not match size(src) = $(size(src))"))
        n = length(src)
        i = 0
        GC.@preserve dest src begin
            for d in 4:-1:0
                block = 1 << (2*d)
                while i + block <= n
                    ccall(($(QuoteNode(promote_name)), libzfp), Cvoid,
                        (Ptr{Int32}, Ptr{$T}, Cuint),
                        pointer(dest, i+1), pointer(src, i+1), d)
                    i += block
                end
            end
        end
        return dest
    end

    @eval function zfp_demote!(dest::DenseArray{$T}, src::DenseArray{Int32})
        size(dest) == size(src) || throw(DimensionMismatch(
            "size(dest) = $(size(dest)) does not match size(src) = $(size(src))"))
        n = length(src)
        i = 0
        GC.@preserve dest src begin
            for d in 4:-1:0
                block = 1 << (2*d)
                while i + block <= n
                    ccall(($(QuoteNode(demote_name)), libzfp), Cvoid,
                        (Ptr{$T}, Ptr{Int32}, Cuint),
                        pointer(dest, i+1), pointer(src, i+1), d)
                    i += block
                end
            end
        end
        return dest
    end
end

function zfp_promote(src::AbstractArray{<:_LowBitInt})
    zfp_promote!(similar(src, Int32), src)
end

function zfp_demote(::Type{T}, src::AbstractArray{Int32}) where {T<:_LowBitInt}
    zfp_demote!(similar(src, T), src)
end

# UTILITY: clamp 32/64-bit integer arrays into the range zfp's integer
# compressor can faithfully represent. zfp requires |x| < 2^30 for the 32-bit
# path and |x| < 2^62 for the 64-bit path; values outside the range get
# truncated by the compressor, so callers should clamp first if they want
# defined behavior. See:
# https://zfp.readthedocs.io/en/release1.0.1/faq.html#q-int32
const _ZfpClampInt = Union{Int32, Int64, UInt32, UInt64}

zfp_max_magnitude(::Type{Int32})  =  Int32(2^30 - 1)
zfp_max_magnitude(::Type{Int64})  =  Int64(2^62 - 1)
zfp_max_magnitude(::Type{UInt32}) = UInt32(2^30 - 1)
zfp_max_magnitude(::Type{UInt64}) = UInt64(2^62 - 1)

"""
    zfp_clamp_int!(arr::AbstractArray{<:Union{Int32,Int64,UInt32,UInt64}}) -> Bool

Clamp every element of `arr` into the zfp-safe range in place: signed types
clamp to `[-2^30+1, 2^30-1]` (resp. `±2^62-1` for 64-bit), unsigned types to
`[0, 2^30-1]` / `[0, 2^62-1]`. Returns `true` if any element was clamped.
"""
function zfp_clamp_int!(arr::AbstractArray{T}) where {T <: _ZfpClampInt}
    hi = zfp_max_magnitude(T)
    lo = T <: Signed ? -hi : zero(T)
    clamped = false

    @inbounds for i in eachindex(arr)
        x = arr[i]
        if x > hi
            arr[i] = hi
            clamped = true
        elseif x < lo
            arr[i] = lo
            clamped = true
        end
    end

    return clamped
end

"""
    zfp_clamp_int(arr) -> (clamped_copy, was_clamped)

Allocating variant of [`zfp_clamp_int!`](@ref): returns a fresh array clamped
into the zfp-safe range alongside a flag indicating whether anything was
actually clamped.
"""
function zfp_clamp_int(arr::AbstractArray{T}) where {T <: _ZfpClampInt}
    out = copy(arr)
    return out, zfp_clamp_int!(out)
end

# COMPRESSION AND DECOMPRESSION
"""Low-level C call to run the compression."""
function zfp_compress(stream::Ptr{Cvoid}, field::Ptr{Cvoid})
    @gc_safe ccall((:zfp_compress, libzfp), Int, (Ptr{Cvoid}, Ptr{Cvoid}), stream, field)
end

"""Low-level C call to run the decompression."""
function zfp_decompress(stream::Ptr{Cvoid}, field::Ptr{Cvoid})
    @gc_safe ccall((:zfp_decompress, libzfp), Int, (Ptr{Cvoid}, Ptr{Cvoid}), stream, field)
end

function zfp_compress(src::AbstractArray{T};
    write_header::Bool=true,
    nthreads::Int=1,
    kws...) where {T<:Union{Int32,Int64,Float32,Float64}}

    dest = UInt8[]
    return zfp_compress!(dest, src; write_header, nthreads, kws...)
end

"""
    zfp_compress!(dest::Vector{UInt8}, src::AbstractArray; kws...)

Compress `src` into `dest`. `dest` may be resized as needed to fit the compressed
data. Same keyword arguments as `zfp_compress`.
"""
function zfp_compress!(dest::Vector{UInt8}, src::AbstractArray{T};
                       write_header::Bool=true,
                       nthreads::Int=1,
                       kws...) where {T<:Union{Int32,Int64,Float32,Float64}}

    ndims = length(size(src))
    ndims in [1, 2, 3, 4] || throw(DimensionMismatch("Zfp compression only for 1-4D array."))

    zfpstream = Ptr{Cvoid}(C_NULL)
    field = Ptr{Cvoid}(C_NULL)
    bitstream = Ptr{Cvoid}(C_NULL)

    # src and dest are only reachable through raw pointers held by the C structs
    # below, so root them for as long as zfp may dereference them.
    compressed_size = GC.@preserve src dest try
        zfpstream = zfp_stream(T, ndims; kws...)  # initialize the compression
        field = zfp_field(src)                  # turn src array into zfp field

        # ensure the destination buffer is large enough for the worst case
        bufsize = zfp_stream_maximum_size(zfpstream, field)
        if length(dest) < bufsize
            resize!(dest, bufsize)
        end

        bitstream = stream_open(dest)  # turn array into zfp pointer
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
        if success == 0
            throw(error("Zfp compression failed."))
        end

        zfp_stream_flush(zfpstream)
        zfp_stream_compressed_size(zfpstream)
    finally
        # free and close, also on the error paths above
        if field != C_NULL
            zfp_field_free(field)
        end
        if zfpstream != C_NULL
            zfp_stream_close(zfpstream)
        end
        if bitstream != C_NULL
            stream_close(bitstream)
        end
    end

    return resize!(dest, compressed_size)
end

function zfp_decompress!(dest::AbstractArray{T},
    src::AbstractVector{UInt8};
    kws...) where {T<:Union{Int32,Int64,Float32,Float64}}

    ndims = length(size(dest))
    ndims in [1, 2, 3, 4] || throw(DimensionMismatch("Zfp compression only for 1-4D array."))

    # src and dest are only reachable through raw pointers held by the C structs
    # below, so root them for as long as zfp may dereference them.
    compressed_size = GC.@preserve src dest begin
        bitstream = stream_open(src)
        zfpstream = zfp_stream_open(bitstream)

        # Opportunistically read a header. If src wasn't written with one the magic
        # bits won't match, so rewind and configure from kwargs instead. Build the
        # field after the probe — a failed header read may have clobbered it.
        probe = zfp_field_alloc()
        if zfp_read_header(zfpstream, probe, HEADER_FULL) != 0
            # zfp writes header_type × header_dims values through dest's pointer,
            # so validate both against dest before decompressing.
            htype = zfp_field_type(probe)
            ZF = ZfpField(probe)
            hsize = map(Int, filter(!=(0), (ZF.nx, ZF.ny, ZF.nz, ZF.nw)))
            zfp_field_free(probe)
            if htype != T || hsize != size(dest)
                zfp_stream_close(zfpstream)
                stream_close(bitstream)
                throw(DimensionMismatch("zfp header describes a size-$hsize array" *
                    " of $htype, but dest is a size-$(size(dest)) array of $T"))
            end
        else
            zfp_field_free(probe)
            zfp_stream_rewind(zfpstream)
            zfp_stream_set_mode!(zfpstream, T, ndims; kws...)    # initialize decompression
        end
        field = zfp_field(dest)  # sets dest's pointer and strides on the field

        # perform decompression
        nbytes = zfp_decompress(zfpstream, field)

        # free and close
        zfp_field_free(field)
        zfp_stream_close(zfpstream)
        stream_close(bitstream)

        nbytes
    end

    # check for failure
    compressed_size == 0 && throw(error("Zfp decompression failed."))
    return nothing
end

"""
    zfp_decompress_allocate(src::AbstractVector{UInt8}) -> Array

Read the zfp header in `src` and allocate an uninitialized `Array{T, N}` of the
right element type and shape to hold the decompressed data. Use with
`zfp_decompress!(dest, src)` to fill the buffer.
"""
function zfp_decompress_allocate(src::AbstractVector{UInt8})
    # The bitstream holds a raw pointer into src, so root src across the reads
    T, ndims, n = GC.@preserve src begin
        bitstream = stream_open(src)
        zfpstream = zfp_stream_open(bitstream)
        field = zfp_field_alloc()

        if zfp_read_header(zfpstream, field, HEADER_FULL) == 0
            zfp_field_free(field)
            zfp_stream_close(zfpstream)
            stream_close(bitstream)
            throw(error("Reading header failed."))
        end

        htype = zfp_field_type(field)
        hdims = zfp_field_dimensionality(field)
        ZF = ZfpField(field)
        hn = filter(!=(0), (ZF.nx, ZF.ny, ZF.nz, ZF.nw))

        zfp_field_free(field)
        zfp_stream_close(zfpstream)
        stream_close(bitstream)

        (htype, hdims, hn)
    end

    return Array{T,ndims}(undef, n...)
end

function zfp_decompress(src::AbstractVector{UInt8})
    output = zfp_decompress_allocate(src)
    zfp_decompress!(output, src)
    return output
end
