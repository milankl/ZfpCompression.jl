using zfp_jll

zfp_type(::Type{Int32}) = 1
zfp_type(::Type{Int64}) = 2
zfp_type(::Type{Float32}) = 3
zfp_type(::Type{Float64}) = 4
zfp_type(::Type) = 0

zfp_type_size(i::Int64) = ccall((:zfp_type_size,libzfp),Int64,(Int64,),i)
zfp_type_size(::Type{T}) where T = zfp_type_size(zfp_type(T))

zfp_stream_open() = ccall((:zfp_stream_open,libzfp),Ptr{Cvoid},(Ptr,),C_NULL)

## READ IN ARRAYS
function zfp_field(A::Array{T,1}) where T
    n = length(A)
    ccall((:zfp_field_1d,libzfp),Ptr{Cvoid},
        (Ptr{Cvoid},Cint,Cuint),A,zfp_type(T),n)
end

function zfp_field(A::Array{T,2}) where T
    nx,ny = size(A)
    ccall((:zfp_field_2d,libzfp),Ptr{Cvoid},
        (Ptr{Cvoid},Cint,Cuint,Cuint),A,zfp_type(T),nx,ny)
    #ccall((:zfp_field_set_stride_2d, libzfp), Cvoid, (Ptr{Cvoid}, Cint, Cint), field, 1, 100)
end

function zfp_field(A::Array{T,3}) where T
    nx,ny,nz = size(A)
    ccall((:zfp_field_3d,libzfp),Ptr{Cvoid},
        (Ptr{Cvoid},Cint,Cuint,Cuint,Cuint),A,zfp_type(T),nx,ny,nz)
end

function zfp_field(A::Array{T,4}) where T
    nx,ny,nz,nw = size(A)
    ccall((:zfp_field_4d,libzfp),Ptr{Cvoid},
        (Ptr{Cvoid},Cint,Cuint,Cuint,Cuint,Cuint),A,zfp_type(T),nx,ny,nz,nw)
end

## COMPRESSION OPTIONS
function zfp_stream_set_rate(stream::Ptr{Cvoid},rate::Real,type::Type,dims::Integer)
    ccall((:zfp_stream_set_rate,libzfp),Cdouble,
        (Ptr{Cvoid},Cdouble,Cuint,Cuint),stream,Float64(rate),zfp_type(type),dims)
end

function zfp_stream_set_precision(stream::Ptr{Cvoid},precision::Integer)
    p = ccall((:zfp_stream_set_precision,libzfp),Cuint,
        (Ptr{Cvoid},Cuint),stream,UInt(precision))
    p == precision || @warn "Precision set to $p"
end

function zfp_stream_set_accuracy(stream::Ptr{Cvoid},tol::AbstractFloat)
    t = ccall((:zfp_stream_set_accuracy,libzfp),Cdouble,
        (Ptr{Cvoid},Cdouble),stream,Float64(tol))
    # t == tol || @warn "Tolerance set to $t"
end

function zfp_stream_set_reversible(stream::Ptr{Cvoid})
    ccall((:zfp_stream_set_reversible,libzfp),Cvoid,(Ptr{Cvoid},),stream)
end

## BUFFER
function zfp_stream_maximum_size(stream::Ptr{Cvoid},field::Ptr{Cvoid})
    ccall((:zfp_stream_maximum_size,libzfp),Int,
        (Ptr{Cvoid},Ptr{Cvoid}),stream,field)
end

function zfp_stream_compressed_size(stream::Ptr{Cvoid})
    ccall((:zfp_stream_compressed_size,libzfp),Int,(Ptr{Cvoid},),stream)
end

function stream_open(buffer::Ptr,bufsize::Int)
    ccall((:stream_open,libzfp),Ptr{Cvoid},(Ptr{Cvoid},Cuint),buffer,bufsize)
end

function zfp_stream_set_bit_stream(stream::Ptr{Cvoid},bitstream::Ptr{Cvoid})
    ccall((:zfp_stream_set_bit_stream,libzfp),Cvoid,(Ptr{Cvoid},Ptr{Cvoid}),
        stream,bitstream)
end

function zfp_stream_rewind(stream::Ptr{Cvoid})
    ccall((:zfp_stream_rewind,libzfp),Cvoid,(Ptr{Cvoid},),stream)
end

## COMPRESSION AND DECOMPRESSION
function zfp_compress(stream::Ptr{Cvoid},field::Ptr{Cvoid})
    ccall((:zfp_compress,libzfp),Int,(Ptr{Cvoid},Ptr{Cvoid}),stream,field)
end

function zfp_decompress(stream::Ptr{Cvoid},field::Ptr{Cvoid})
    ccall((:zfp_decompress,libzfp),Int,(Ptr{Cvoid},Ptr{Cvoid}),stream,field)
end

function zfp_compress(  src::AbstractArray{T};
                        tol::Real=0,
                        precision::Int=0,
                        rate::Int=0) where {T<:Union{Int32,Int64,Float32,Float64}}

    ndims = length(size(src))
    ndims in [1,2,3,4] || throw(DimensionMismatch("Zfp compression only for 1-4D array."))

    zfp = zfp_stream_open()
    field = zfp_field(src)

    # propagate compression options
    if tol > 0
        zfp_stream_set_accuracy(zfp,tol)
    elseif precision > 0
        zfp_stream_set_precision(zfp,precision)
    elseif rate > 0
        maxrate = 8*zfp_type_size(T)
        rate <= maxrate || @warn "Rate was set to $rate-bit, beyond $maxrate-bit for type $T"
        zfp_stream_set_rate(zfp,rate,eltype(src),ndims)
    else
        zfp_stream_set_reversible(zfp)
    end

    bufsize = zfp_stream_maximum_size(zfp,field)
    dest = Vector{UInt8}(undef,bufsize)
    stream = stream_open(pointer(dest),bufsize)
    zfp_stream_set_bit_stream(zfp,stream)
    zfp_stream_rewind(zfp)
    compressed_size = zfp_compress(zfp,field)

    compressed_size == 0 && throw(error("Zfp compression failed."))

    return dest[1:compressed_size]
end

function zfp_decompress!(   dest::AbstractArray{T},
                            src::Vector{UInt8};
                            tol::Real=0,
                            precision::Int=0,
                            rate::Int=0) where {T<:Union{Int32,Int64,Float32,Float64}}

    ndims = length(size(dest))
    ndims in [1,2,3,4] || throw(DimensionMismatch("Zfp compression only for 1-4D array."))

    zfp = zfp_stream_open()
    field = zfp_field(dest)

    # propagate compression options
    if tol > 0
        zfp_stream_set_accuracy(zfp,tol)
    elseif precision > 0
        zfp_stream_set_precision(zfp,precision)
    elseif rate > 0
        maxrate = 8*zfp_type_size(T)
        rate > maxrate && @warn "Rate was set to $rate-bit, beyond $maxrate-bit for type $T"
        zfp_stream_set_rate(zfp,rate,eltype(dest),ndims)
    else
        zfp_stream_set_reversible(zfp)
    end

    bufsize = zfp_stream_maximum_size(zfp,field)
    stream = stream_open(pointer(src),bufsize)
    zfp_stream_set_bit_stream(zfp,stream)
    zfp_stream_rewind(zfp)
    compressed_size = zfp_decompress(zfp,field)
    compressed_size == 0 && throw(error("Zfp decompression failed."))
    return nothing
end
