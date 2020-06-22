using ZfpCompression
using Test

@testset "Zfp.jl" begin
    @test 4 == zfp_type_size(1)
end
