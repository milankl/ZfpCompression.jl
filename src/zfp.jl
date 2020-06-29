using zfp_jll

# zfp types declaration and size
zfp_type(::Type{Int32}) = 1
zfp_type(::Type{Int64}) = 2
zfp_type(::Type{Float32}) = 3
zfp_type(::Type{Float64}) = 4
zfp_type(::Type) = 0

"""Size of zfp types (Int32,Int64,Float32,Float64) in bytes."""
zfp_type_size(i::Int64) = ccall((:zfp_type_size,libzfp),Int64,(Int64,),i)
zfp_type_size(::Type{T}) where T = zfp_type_size(zfp_type(T))

# READ IN ARRAYS
"""Pass a 1-D array into a zfp_field in C, in Julia only as Ptr{Cvoid}."""
function zfp_field(A::Array{T,1}) where T
    n = length(A)
    ccall((:zfp_field_1d,libzfp),Ptr{Cvoid},
        (Ptr{Cvoid},Cint,Cuint),A,zfp_type(T),n)
end

"""Pass a 2-D array into a zfp_field in C, in Julia only as Ptr{Cvoid}."""
function zfp_field(A::Array{T,2}) where T
    nx,ny = size(A)
    field = ccall((:zfp_field_2d,libzfp),Ptr{Cvoid},
        (Ptr{Cvoid},Cint,Cuint,Cuint),A,zfp_type(T),nx,ny)
    sx,sy = strides(A)
    ccall((:zfp_field_set_stride_2d, libzfp), Cvoid, (Ptr{Cvoid}, Cint, Cint), field, sx, sy)
    return field
end

"""Pass a 3-D array into a zfp_field in C, in Julia only as Ptr{Cvoid}."""
function zfp_field(A::Array{T,3}) where T
    nx,ny,nz = size(A)
    field = ccall((:zfp_field_3d,libzfp),Ptr{Cvoid},
        (Ptr{Cvoid},Cint,Cuint,Cuint,Cuint),A,zfp_type(T),nx,ny,nz)
    sx,sy,sz = strides(A)
    ccall((:zfp_field_set_stride_3d, libzfp), Cvoid, (Ptr{Cvoid}, Cint, Cint, Cint),
        field, sx, sy, sz)
    return field
end

"""Pass a 4-D array into a zfp_field in C, in Julia only as Ptr{Cvoid}."""
function zfp_field(A::Array{T,4}) where T
    nx,ny,nz,nw = size(A)
    ccall((:zfp_field_4d,libzfp),Ptr{Cvoid},
        (Ptr{Cvoid},Cint,Cuint,Cuint,Cuint,Cuint),A,zfp_type(T),nx,ny,nz,nw)
    sx,sy,sz,sw = strides(A)
    ccall((:zfp_field_set_stride_4d, libzfp), Cvoid, (Ptr{Cvoid}, Cint, Cint, Cint, Cint),
        field, sx, sy, sz, sw)
    return field
end

# COMPRESSION OPTIONS
"""Open a stream (=object that holds the (de)compression settings) for zfp."""
zfp_stream_open() = ccall((:zfp_stream_open,libzfp),Ptr{Cvoid},(Ptr,),C_NULL)

"""Set bitrate (=bits per value) to set the compression rate directly.
Should not be larger than the bits per value of the uncompressed array."""
function zfp_stream_set_rate(stream::Ptr{Cvoid},rate::Real,type::Type,dims::Integer)
    ccall((:zfp_stream_set_rate,libzfp),Cdouble,
        (Ptr{Cvoid},Cdouble,Cuint,Cuint),stream,Float64(rate),zfp_type(type),dims)
end

"""Set the precision (≈ mantissa bits per value) for compression."""
function zfp_stream_set_precision(stream::Ptr{Cvoid},precision::Integer)
    ccall((:zfp_stream_set_precision,libzfp),Cuint,
        (Ptr{Cvoid},Cuint),stream,UInt(precision))
    # p == precision || @warn "Precision set to $p"
end

"""Set the accuracy (=max abs error) for compression."""
function zfp_stream_set_accuracy(stream::Ptr{Cvoid},tol::AbstractFloat)
    ccall((:zfp_stream_set_accuracy,libzfp),Cdouble,
        (Ptr{Cvoid},Cdouble),stream,Float64(tol))
    # t == tol || @warn "Tolerance set to $t"
end

"""Set the zfp compression to lossless = reversible."""
function zfp_stream_set_reversible(stream::Ptr{Cvoid})
    ccall((:zfp_stream_set_reversible,libzfp),Cvoid,(Ptr{Cvoid},),stream)
end

"""Initialize a zfp stream C struct holding the compression settings.
    Only a Ptr{Cvoid} is returned to Julia."""
function zfp_stream(::Type{T},
                    ndims::Int;
                    tol::Real=0,
                    precision::Real=0,
                    rate::Int=0) where T

    stream = zfp_stream_open()

    # set the compression options
    if tol > 0
        zfp_stream_set_accuracy(stream,tol)
    elseif precision > 0
        zfp_stream_set_precision(stream,precision)
    elseif rate > 0
        maxrate = 8*zfp_type_size(T)
        rate <= maxrate || @warn "Rate was set to $rate-bit > $maxrate-bit for type $T"
        zfp_stream_set_rate(stream,rate,T,ndims)
    else  # lossless
        zfp_stream_set_reversible(stream)
    end

    return stream
end

# BUFFER
"""Retrieve the max size of the compressed stream in bytes pre-compression."""
function zfp_stream_maximum_size(stream::Ptr{Cvoid},field::Ptr{Cvoid})
    ccall((:zfp_stream_maximum_size,libzfp),Int,
        (Ptr{Cvoid},Ptr{Cvoid}),stream,field)
end

"""Retrieve the actual size of the compressed stream in bytes post-compression."""
function zfp_stream_compressed_size(stream::Ptr{Cvoid})
    ccall((:zfp_stream_compressed_size,libzfp),Int,(Ptr{Cvoid},),stream)
end

"""Open a buffer (= pointer for a preallocated data array) into a bitstream
    for zfp to flush the compressed array into."""
function stream_open(buffer::Ptr,bufsize::Int)
    ccall((:stream_open,libzfp),Ptr{Cvoid},(Ptr{Cvoid},Cuint),buffer,bufsize)
end

"""Connect the zfp stream (=object that holds the compression settings) to the
bitstream that will contain the compressed array."""
function zfp_stream_set_bit_stream(stream::Ptr{Cvoid},bitstream::Ptr{Cvoid})
    ccall((:zfp_stream_set_bit_stream,libzfp),Cvoid,(Ptr{Cvoid},Ptr{Cvoid}),
        stream,bitstream)
end

"""Rewind the data stream."""
function zfp_stream_rewind(stream::Ptr{Cvoid})
    ccall((:zfp_stream_rewind,libzfp),Cvoid,(Ptr{Cvoid},),stream)
end

# COMPRESSION AND DECOMPRESSION
"""Low-level C call to run the compression."""
function zfp_compress(stream::Ptr{Cvoid},field::Ptr{Cvoid})
    ccall((:zfp_compress,libzfp),Int,(Ptr{Cvoid},Ptr{Cvoid}),stream,field)
end

"""Low-level C call to run the decompression."""
function zfp_decompress(stream::Ptr{Cvoid},field::Ptr{Cvoid})
    ccall((:zfp_decompress,libzfp),Int,(Ptr{Cvoid},Ptr{Cvoid}),stream,field)
end

function zfp_compress(  src::AbstractArray{T};
                        kws...) where {T<:Union{Int32,Int64,Float32,Float64}}

    ndims = length(size(src))
    ndims in [1,2,3,4] || throw(DimensionMismatch("Zfp compression only for 1-4D array."))

    zfp = zfp_stream(T,ndims;kws...)    # initialize the compression
    field = zfp_field(src)              # turn src array into zfp field

    # preallocate the compressed array
    bufsize = zfp_stream_maximum_size(zfp,field)
    dest = Vector{UInt8}(undef,bufsize)             # allocate as UInt8
    bitstream = stream_open(pointer(dest),bufsize)  # turn array into zfp pointer
    zfp_stream_set_bit_stream(zfp,bitstream)        # connect bitstream pointer to zfp struct
    zfp_stream_rewind(zfp)

    # perform compression
    compressed_size = zfp_compress(zfp,field)

    # check for failure
    compressed_size == 0 && throw(error("Zfp compression failed."))

    return dest[1:compressed_size]
end

function zfp_decompress!(   dest::AbstractArray{T},
                            src::Vector{UInt8};
                            kws...) where {T<:Union{Int32,Int64,Float32,Float64}}

    ndims = length(size(dest))
    ndims in [1,2,3,4] || throw(DimensionMismatch("Zfp compression only for 1-4D array."))

    zfp = zfp_stream(T,ndims;kws...)    # initialize decompression
    field = zfp_field(dest)             # turn destination array into zfp pointer

    # declare src as the bitstream to decompress and connect to zfp struct
    bufsize = zfp_stream_maximum_size(zfp,field)
    stream = stream_open(pointer(src),bufsize)
    zfp_stream_set_bit_stream(zfp,stream)
    zfp_stream_rewind(zfp)

    # perform decompression
    compressed_size = zfp_decompress(zfp,field)

    # check for failure
    compressed_size == 0 && throw(error("Zfp decompression failed."))
    return nothing
end
