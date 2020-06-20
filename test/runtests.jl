using Zfp
using Test

@testset "Zfp.jl" begin
    @test 4 == Zfp.zfp_type_size(1)
end
