#include <cassert>
#include <cstdlib>
#include <chrono>

#include <Kokkos_Core.hpp>
#include <fmt/core.h>

constexpr const int BLOCK_SIZE = 10;

using MatrixR = Kokkos::View<double**, Kokkos::LayoutRight>;
using MatrixL = Kokkos::View<double**, Kokkos::LayoutLeft>;
using std::chrono::high_resolution_clock;
using std::chrono::time_point;

template <class MatrixType>
double matrix_checksum(MatrixType& M) {
  static_assert(2 == MatrixType::rank(), "View must be of rank 2");

  double cs = 0.0;

  for (size_t i = 0; i < M.extent(0); i++) {
    for (size_t j = 0; j < M.extent(1); j++) {
      cs += M(i, j);
    }
  }

  return cs;
}


template <class MatrixType>
auto matrix_init(MatrixType& M) -> void {
  static_assert(2 == MatrixType::rank(), "View must be of rank 2");

  Kokkos::parallel_for(
    "init",
    M.extent(0),
    KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < int(M.extent(1)); ++j) {
        M(i, j) = std::max(i, j) % 6 -2;
      }
    }
  );
}

template <class AMatrixType, class BMatrixType, class CMatrixType>
auto matrix_product(double alpha, AMatrixType const& A, BMatrixType const& B, double beta, CMatrixType& C) -> void {
  static_assert(
    AMatrixType::rank() == 2 && BMatrixType::rank() == 2 && CMatrixType::rank() == 2, "Views must be of rank 2"
  );
  assert(A.extent(0) == C.extent(0));
  assert(B.extent(1) == C.extent(1));
  assert(A.extent(1) == B.extent(0));

  int blocks_i = A.extent(0) / BLOCK_SIZE + (A.extent(0) % BLOCK_SIZE != 0);
  int blocks_j = B.extent(1) / BLOCK_SIZE + (B.extent(1) % BLOCK_SIZE != 0);

  Kokkos::parallel_for(
    "dgemm_kernel",
    blocks_i,
    KOKKOS_LAMBDA(int bi) {
      for (int bj = 0; bj < blocks_j; ++bj) {
        int i_lim = std::min((int) A.extent(0), bi * BLOCK_SIZE + BLOCK_SIZE);

        for (int i = bi * BLOCK_SIZE; i < i_lim; ++i) {
          int j_lim = std::min((int) B.extent(1), bj * BLOCK_SIZE + BLOCK_SIZE);
          
          for (int j = bj * BLOCK_SIZE; j < j_lim; ++j) {
              
            double acc = 0.0;
            for (int k = 0; k < int(A.extent(1)); ++k) {
              acc += A(i, k) * B(k, j);
            }
            C(i, j) *= beta + alpha * acc;
              
          }
        }
      }
    }
  );
}

auto main(int argc, char* argv[]) -> int {
  if (argc < 4) {
    fmt::print("Usage: {} <M> <N> <K>\n", argv[0]);
    return -1;
  }
  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);

  // Known seed for deterministic RNG
  srand48(42);
  Kokkos::initialize(argc, argv);
  {
    auto A = MatrixR("A", m, k);
    auto B = MatrixL("B", k, n);
    auto C = MatrixR("C", m, n);

    time_point<high_resolution_clock> t1, t2;

    double alpha = drand48();
    matrix_init(A);
    matrix_init(B);
    double beta = drand48();
    matrix_init(C);

    Kokkos::fence();
    t1 = high_resolution_clock::now();
    matrix_product(alpha, A, B, beta, C);
    t2 = high_resolution_clock::now();
    auto time = (t2 - t1).count();
    Kokkos::fence();
    // M N K cpus time time(s) checksum
    fmt::print("{}\t{}\t{}\t{}\t{}\t{:.4}s\t{}\n", m, n, k, Kokkos::OpenMP::impl_get_current_max_threads(), time, (double) time / 1e9, matrix_checksum(C));
  }
  Kokkos::finalize();
  return 0;
}
