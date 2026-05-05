using ZfpCompression
using Test

using ZfpCompression: zfp_promote, zfp_promote!, zfp_demote, zfp_demote!, zfp_decompress_allocate,
    zfp_clamp_int

@testset "Lossless 1-4D for all types" begin

    for T in (Float32,Float64)
        # test 1-4D
        for sizes in [(100,),(100,50),(20,30,40),(20,20,20,20)]

            # for arrays
            A = rand(T,sizes...)
            Ac = zfp_compress(A)
            Ad = zfp_decompress(Ac)

            # for views but with unit stride
            A_view = reshape(view(A, :), size(A))
            Ac_view = zfp_compress(A_view)
            Ad_view = zfp_decompress(view(Ac,:))

            @test Ac == Ac_view
            @test Ad == Ad_view == A
        end

        # non-unit strides
        for (size,stride) in zip(((100,),(100,100),(100,100,100),(20,30,40,50)),
            ((1:2:100,), (1:2:100,1:3:100), (1:2:100,1:3:100,1:4:100), (1:2:20, 1:3:30, 1:4:40,1:5:50)))

            A = rand(T,size...)
            A_view = view(A,stride...)
            Ac = zfp_compress(A_view)
            Ad = zfp_decompress(Ac)
            @test A_view == Ad

            # same array but collect the view, to check that exactly the same happens with/without view
            A2 = collect(A_view)
            Ac2 = zfp_compress(A2)
            Ad2 = zfp_decompress(Ac2)
            @test Ac2 == Ac
            @test A2 == A_view == Ad2
        end
    end

    for T in (Int32,Int64)
        # test 2-4D
        for sizes in [(100,50),(20,30,40),(20,20,20,20)]
            A = rand(T,sizes...)
            Ac = zfp_compress(A)
            Ad = zfp_decompress(Ac)
            Adviews = zfp_decompress(view(Ac,:))
            @test Ad == Adviews == A
        end
    end
end

@testset "Max abs error is bound in 1-4D for floats" begin

    for tol in [1e-1,1e-3,1e-5,1e-7]
        for T in (Float32,Float64)
            # test 1-4D
            for sizes in [(100,),(100,50),(20,30,40),(20,20,20,20)]
                A = rand(T,sizes...)
                Ac = zfp_compress(A;tol)
                Ad = zfp_decompress(Ac)
                @test maximum(abs.(Ad-A)) <= tol
            end
        end
    end
end

@testset "zfp_decompress_allocate" begin
    for T in (Float32, Float64, Int32, Int64)
        for sizes in [(100,), (50, 30), (10, 10, 10), (5, 5, 5, 5)]
            A = rand(T, sizes...)
            Ac = zfp_compress(A)

            # right type and shape from the header alone
            dest = zfp_decompress_allocate(Ac)
            @test dest isa Array{T, length(sizes)}
            @test size(dest) == sizes

            # zfp_decompress! auto-detects the embedded header and fills the buffer
            zfp_decompress!(dest, Ac)
            @test dest == A

            # buffer can be reused for a different payload of the same shape/type
            B = rand(T, sizes...)
            Bc = zfp_compress(B)
            zfp_decompress!(dest, Bc)
            @test dest == B
        end
    end

    # bogus input has no valid header
    bogus = rand(UInt8, 64)
    @test_throws ErrorException zfp_decompress_allocate(bogus)
end

@testset "promote/demote round-trip" begin
    # exercise sizes that span the dims=4..0 fall-through (block sizes 256, 64, 16, 4, 1)
    for n in (1, 3, 4, 5, 17, 100, 256, 257, 1000)
        for T in (Int8, UInt8, Int16, UInt16)
            A = rand(T, n)
            P = zfp_promote(A)
            @test P isa Vector{Int32}
            @test length(P) == n
            @test zfp_demote(T, P) == A
        end
    end

    # multi-dimensional, shape preserved
    for T in (Int8, UInt8, Int16, UInt16)
        A = rand(T, 7, 9)
        P = zfp_promote(A)
        @test P isa Matrix{Int32}
        @test size(P) == size(A)
        @test zfp_demote(T, P) == A
    end

    # size mismatch throws
    @test_throws DimensionMismatch zfp_promote!(zeros(Int32, 5), Int8[1,2,3,4])
    @test_throws DimensionMismatch zfp_demote!(zeros(Int8, 5), zeros(Int32, 4))
end

@testset "zfp_clamp_int" begin
    # In-range values pass through untouched and was_clamped is false.
    for T in (Int32, Int64, UInt32, UInt64)
        out, was_clamped = zfp_clamp_int(T[0, 1, 100])
        @test out == T[0, 1, 100]
        @test !was_clamped
    end

    # Signed: clamps both sides to ±(2^30-1) / ±(2^62-1).
    s32, c32 = zfp_clamp_int(Int32[typemin(Int32), -10, 0, 10, typemax(Int32)])
    @test s32 == Int32[-(Int32(2)^30 - 1), -10, 0, 10, Int32(2)^30 - 1]
    @test c32

    s64, c64 = zfp_clamp_int(Int64[typemin(Int64), 0, typemax(Int64)])
    @test s64 == Int64[-(Int64(2)^62 - 1), 0, Int64(2)^62 - 1]
    @test c64

    # Unsigned: lower bound is 0, no negative clamp path.
    u32, cu32 = zfp_clamp_int(UInt32[0, 5, typemax(UInt32)])
    @test u32 == UInt32[0, 5, UInt32(2)^30 - 1]
    @test cu32
end
