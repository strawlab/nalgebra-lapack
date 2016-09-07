error_chain!{
    errors {
        LapackIllegalArgument(argno: i32) {
            description("illegal argument to LAPACK function")
            display("illegal argument (argument number {}) to LAPACK function", argno)
        }
        LapackFailure(info: i32) {
            description("failure in LAPACK function")
            display("failure in LAPACK function (info={})", info)
        }
        MatrixNotSquare {
            description("a square matrix is required")
            display("a square matrix is required")
        }
    }
}
