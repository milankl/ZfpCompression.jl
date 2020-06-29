using ZfpCompression
using Test

@testset "Lossless 1-4D for all types" begin

    for T in (Float32,Float64,Int32,Int64)
        # test 1-4D
        for sizes in [(100,),(100,50),(20,30,40),(20,20,20,20)]
            A = rand(Float32,sizes...)
            Ac = zfp_compress(A)
            Ad = similar(A)
            zfp_decompress!(Ad,Ac)
            @test Ad == A
        end
    end
end

@testset "Max abs error is bound in 1-4D for all types" begin

    tol = 1e-3

    for T in (Float32,Float64,Int32,Int64)
        # test 1-4D
        for sizes in [(100,),(100,50),(20,30,40),(20,20,20,20)]
            A = rand(Float32,sizes...)
            Ac = zfp_compress(A,tol=tol)
            Ad = similar(A)
            zfp_decompress!(Ad,Ac,tol=tol)
            @test maximum(abs.(Ad-A)) <= tol
        end
    end
end
