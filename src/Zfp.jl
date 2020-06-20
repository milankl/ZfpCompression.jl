module Zfp

    using zfp_jll

    zfp_type_size(i::Int64) = ccall((:zfp_type_size,libzfp),Int64,(Int64,),i)
    
end
