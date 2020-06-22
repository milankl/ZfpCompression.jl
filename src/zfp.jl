using zfp_jll
using Base.GC: @preserve

zfp_type(::Type{Int32}) = 1
zfp_type(::Type{Int64}) = 2
zfp_type(::Type{Float32}) = 3
zfp_type(::Type{Float64}) = 4
zfp_type(::Type) = 0

zfp_type_size(i::Int64) = ccall((:zfp_type_size,libzfp),Int64,(Int64,),i)
zfp_type_size(::Type{T}) where T = zfp_type_size(zfp_type(T))

zfp_stream_open() = ccall((:zfp_stream_open,libzfp),Ptr{Cvoid},(Ptr,),C_NULL)

function zfp_field(A::Array{T,1}) where T
    n = length(A)
    ccall((:zfp_field_1d,libzfp),Ptr{Cvoid},
        (Ptr{Cvoid},Cint,Cuint),A,zfp_type(T),n)
end

function zfp_field(A::Array{T,2}) where T
    nx,ny = size(A)
    ccall((:zfp_field_2d,libzfp),Ptr{Cvoid},
        (Ptr{Cvoid},Cint,Cuint,Cuint),A,zfp_type(T),nx,ny)
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

function zfp_field(A::DenseArray{T}) where T
    zfp_field(zfp_type(T),)

function zfp_stream_maximum_size(stream::Ptr{Cvoid},field::Ptr{Cvoid})
    ccall((:zfp_stream_maximum_size,libzfp),Int,
        (Ptr{Cvoid},Ptr{Cvoid}),stream,field)
end

function zfp_stream_set_precision(stream::Ptr{Cvoid},precision::Integer)
    p = ccall((:zfp_stream_set_precision,libzfp),Cuint,
        (Ptr{Cvoid},Cuint),stream,UInt(precision))
    p == precision || @warn "Precision set to $p"
end

function zfp_stream_set_accuracy(stream::Ptr{Cvoid},tol::AbstractFloat)
    t = ccall((:zfp_stream_set_accuracy,libzfp),Cdouble,
        (Ptr{Cvoid},Cdouble),stream,Float64(tol))
    t == tol || @warn "Tolerance set to $t"
end

function zfp_stream_set_reversible(stream::Ptr{Cvoid})
    ccall((:zfp_stream_set_reversible,libzfp),Cvoid,(Ptr{Cvoid},),stream)
end

# check whether the strides of A correspond to contiguous data
# from Blosc.jl (S Johnson)
iscontiguous(::Array) = true
iscontiguous(::Vector) = true
iscontiguous(A::DenseVector) = stride(A,1) == 1
function iscontiguous(A::DenseArray)
    p = sortperm([strides(A)...])
    s = 1
    for k = 1:ndims(A)
        if stride(A,p[k]) != s
            return false
        end
        s *= size(A,p[k])
    end
    return true
end

# Returns the size of compressed data inside dest
function compress!(dest::DenseVector{UInt8},
                   src::Ptr{T},
                   src_size::Integer;
                   level::Integer=5,
                   shuffle::Bool=true,
                   itemsize::Integer=sizeof(T)) where {T}
    iscontiguous(dest) || throw(ArgumentError("dest must be contiguous array"))
    if !isbitstype(T)
        throw(ArgumentError("buffer eltype must be `isbits` type"))
    end
    if itemsize <= 0
        throw(ArgumentError("itemsize must be positive"))
    end
    if level < 0 || level > 9
        throw(ArgumentError("invalid compression level $level not in [0,9]"))
    end
    if src_size > MAX_BUFFERSIZE
        throw(ArgumentError("data > $MAX_BUFFERSIZE bytes is not supported by Blosc"))
    end
    sz = blosc_compress(level, shuffle, itemsize, src_size, src, dest, sizeof(dest))
    sz < 0 && error("Blosc internal error when compressing data (errorcode: $sz)")
    return convert(Int, sz)
end

compress!(dest::DenseVector{UInt8}, src::AbstractString; kws...) =
    @preserve src compress!(dest, pointer(src), sizeof(src); kws...)

function compress!(dest::DenseVector{UInt8}, src::DenseArray; kws...)
    iscontiguous(src) || throw(ArgumentError("src must be a contiguous array"))
    return @preserve src compress!(dest, pointer(src), sizeof(src); kws...)
end

function compress(src::Ptr{T}, src_size::Integer; kws...) where {T}
    dest = Vector{UInt8}(undef, src_size + MAX_OVERHEAD)
    sz = compress!(dest,src,src_size; kws...)
    @assert(sz > 0 || src_size == 0)
    return resize!(dest, sz)
end
function zfp_compress(src::DenseArray; kws...)
    iscontiguous(src) || throw(ArgumentError("src must be a contiguous array"))
    @preserve src zfp_compress(pointer(src), sizeof(src); kws...)
end
