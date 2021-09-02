//
// Created by hgq on 2021/8/24.
//

#include "src/xnnpack/AlignedAllocator.h"
#include "src/xnnpack/params-init.h"
#include "src/xnnpack/params.h"
#include "src/xnnpack/spmm.h"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>

static void logg(const char *msg, ...) {
  char format[20 + strlen(msg)];
  format[0] = '*';
  format[1] = '*';
  format[2] = '*';
  format[3] = '*';
  format[4] = '*';
  format[5] = 'h';
  format[6] = 'g';
  format[7] = 'q';
  format[8] = '-';
  format[9] = 'd';
  format[10] = 'e';
  format[11] = 'b';
  format[12] = 'u';
  format[13] = 'g';
  format[14] = ':';
  format[15] = ' ';
  format[16] = '\0';
  strcat(format, msg);
  strcat(format, "\r\n");
  va_list ap;
  va_start(ap, msg);
  vprintf(format, ap);
  va_end(ap);
}

static void SpMMBenchmark(xnn_f32_spmm_minmax_ukernel_function spmm, float sparsity) {
  const size_t mc = 8;     // row of output matrix
  const size_t nc = 4;     // col of output matrix
  const size_t kc = 5;     // deep of mvm
  const uint32_t mr = 32;  // block-row of BSR
  const int32_t nr = 1;    // block-col of BSR

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  // if using blocks, generate the reduced matrix first and then extrude along
  // the block dimension (n), to get the full matrix
  // number of blocked-cols. ncols == nc when nr == 1
  size_t ncols = nc / nr + nc % nr;
  std::vector<float> b(ncols * kc);  // right matrix data
  std::vector<float> bias(nc);       // bias of matmul. bias is along with col of output matrix
  std::vector<float> w;
  std::vector<uint32_t> nmap;
  std::vector<int32_t> dmap;
  const size_t sparse_end = std::min(size_t(float(b.size()) * sparsity), b.size());  // number of zeroes
  const size_t num_nonzeroes = nr * (b.size() - sparse_end);                         // number of non-zeroes

  // Sparse representation of weights consists of four components:
  // 1. An array of float values storing non-zero kernel elements, and all (group_output_channels) bias elements. All
  // elements within non-zero block are assumed to be non-zero.
  // 2. An array of int32_t values storing increment for input pointer after each processed tile. This array is derived
  // from scaled difference in array 2 using parameters to setup function.
  // 3. An array of uint32_t values storing the number of non-zero kernel elements per each output channel.
  // 4. An array of int32_t values storing scaled [by sizeof(input element)] difference between input channels
  // corresponding to successive non-zero blocks.

  const size_t w_elements = num_nonzeroes + nc;  // non-zeroes weight and bias
  const size_t c_elements = mc * nc;             // output matrix size
  const size_t dmap_elements = num_nonzeroes / nr;
  const size_t nmap_elements = nc;  // number of non-zeroes per output channel(col of output matrix)

  // Micro-kernel can access one element beyond w and dmap for software
  // pipelining.
  w.reserve(w_elements + 1);
  dmap.reserve(dmap_elements + 1);
  nmap.resize(nmap_elements);

  std::vector<size_t> a_offsets(1);

  // Re-generate weights. Note: each re-generation produces the number of
  // non-zeroes.
  std::fill(b.begin(), b.begin() + sparse_end, 0.0f);                // fill zero weights for b
  std::generate(b.begin() + sparse_end, b.end(), std::ref(f32rng));  // non-zero weights
  std::shuffle(b.begin(), b.end(), rng);                             // shuffle data in b
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));

  uint32_t first_j = 0, last_j = 0;
  bool is_first_nonzero = true;
  for (uint32_t i = 0; i < nc / nr; i++) {  // outer loop align block. if block == 1，outer loop align col
    for (uint32_t n = 0; n < nr; n++)       // loop in block, add bias into w
      w.push_back(bias[nr * i + n]);
    for (uint32_t j = 0; j < kc; j++) {  // loop align row of right-matrix
      if (b[i * kc + j] != 0.0f) {       // add non-zero element from b into w
        for (size_t l = 0; l < nr; l++)  // fill each block by one element of b
          w.push_back(b[i * kc + j] + static_cast<float>(i));
        if (is_first_nonzero) {
          first_j = j;
        } else {
          const ptrdiff_t increment = int32_t(j - last_j) * int32_t(mc) * int32_t(sizeof(float));
          dmap.push_back(increment);  // 计算dmap，计算w中同一列的非零数据对应在input中的间隔
        }
        last_j = j;
        is_first_nonzero = false;
        nmap[i] += 1;  // increment ith col's number of non-zero element
      }
    }
  }
  for (uint32_t i = nc / nr; i < ncols; i++) {
    w.push_back(bias[i]);
    for (uint32_t j = 0; j < kc; j++) {
      if (b[i * kc + j] != 0.0f) {
        w.push_back(b[i * kc + j]);
        if (is_first_nonzero) {
          first_j = j;
        } else {
          const ptrdiff_t increment = int32_t(j - last_j) * int32_t(mc) * int32_t(sizeof(float));
          dmap.push_back(increment);
        }
        last_j = j;
        is_first_nonzero = false;
        nmap[i] += 1;
      }
    }
  }
  {
    const ptrdiff_t increment = int32_t(first_j - last_j) * int32_t(mc) * int32_t(sizeof(float));
    dmap.push_back(increment);
  }

  a_offsets[0] = first_j * mc;

  // Micro-kernel can access one element beyond w and dmap for software
  // pipelining.
  w.resize(w.size() + 1);
  dmap.resize(dmap.size() + 1);

  std::vector<float, AlignedAllocator<float, 64>> a(kc * mc);
  std::vector<float, AlignedAllocator<float, 64>> c(c_elements);

  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::fill(c.begin(), c.end(), nanf(""));

  xnn_f32_minmax_params params{};
  xnn_init_f32_minmax_params(&params, -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());
  spmm(mc * sizeof(float), nc, a.data() + a_offsets[0], w.data(), dmap.data(), nmap.data(), c.data(),
       mc * sizeof(float), &params);
}

int main() {
  logg("---------------------------------------------------");
#if XNN_ARCH_ARM64
  SpMMBenchmark(xnn_f32_spmm_minmax_ukernel_32x1__neonfma_pipelined, 0.8f);
#endif  // XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  SpMMBenchmark(xnn_f32_spmm_minmax_ukernel_32x1__sse, 0.8f);
#endif
}
