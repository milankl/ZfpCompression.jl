using ZfpCompression
using Test

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
            ((1:2:100,) (1:2:100,1:3:100), (1:2:100,1:3:100,1:4:100), (1:2:20, 1:3:30, 1:4:40,1:5:50)))
        
            A = rand(T,size...)
            A_view = view(A,stride...)
            Ac = zfp_compress(A_view)
            Ad = zfp_decompress(Ac)
            @test A_view == Ad
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
