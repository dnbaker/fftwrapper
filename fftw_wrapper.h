#ifndef _FFT_WRAPPER_H_
#define _FFT_WRAPPER_H_

#include <string>
#include <cassert>
#include <numeric>
#include <complex>
#include <vector>
#include <cstdlib>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include "blaze/Math.h"
#include "fftw3.h"


namespace fft {

template<typename T>
auto get_data(T &a) {
    if constexpr(blaze::IsMatrix<T>::value) return &a(0, 0);
    else return &a[0];
}
template<typename T>
const auto get_data(const T &a) {
    if constexpr(blaze::IsMatrix<T>::value) return &a(0, 0);
    else return &a[0];
}

using std::size_t;

static inline void *almalloc(size_t aln, size_t sz) {
#if _GLIBCXX_HAVE_ALIGNED_ALLOC
    return std::aligned_alloc(aln, sz);
#else
    return fftw_malloc(sz);
#endif
}

static inline void alfree(void *ptr) {
#if _GLIBCXX_HAVE_ALIGNED_ALLOC
    std::free(ptr);
#else
    return fftw_free(ptr);
#endif
}

template<typename FloatType>
struct FFTTypes;

template<>
struct FFTTypes<float> {
    using PlanType = fftwf_plan;
    using ComplexType = fftwf_complex;
    using FloatType = float;
    static constexpr decltype(&fftwf_destroy_plan) destroy_fn = &fftwf_destroy_plan;
    static constexpr decltype(&fftwf_execute_r2r)     r2rexec = &fftwf_execute_r2r;
    static constexpr decltype(&fftwf_execute_dft_r2c) r2cexec = &fftwf_execute_dft_r2c;
    static constexpr decltype(&fftwf_execute_dft_c2r) c2rexec = &fftwf_execute_dft_c2r;
    static constexpr decltype(&fftwf_execute_dft)     c2cexec = &fftwf_execute_dft;
    static constexpr decltype(&fftwf_sprint_plan) sprintfn = &fftwf_sprint_plan;
    static constexpr decltype(&fftwf_fprint_plan) fprintfn = &fftwf_fprint_plan;
    static constexpr decltype(&fftwf_cleanup)     cleanupfn= &fftwf_cleanup;
    static constexpr decltype(&fftwf_flops)       flopsfn = &fftwf_flops;
    static constexpr decltype(&fftwf_cost)        costfn = &fftwf_cost;
    static constexpr decltype(&fftwf_plan_dft_r2c) r2cplan = &fftwf_plan_dft_r2c;
    static constexpr decltype(&fftwf_plan_dft_c2r) c2rplan = &fftwf_plan_dft_c2r;
    static constexpr decltype(&fftwf_plan_dft) c2cplan = &fftwf_plan_dft;
    static constexpr decltype(&fftwf_plan_r2r) r2rplan = &fftwf_plan_r2r;
};
template<>
struct FFTTypes<double> {
    using PlanType = fftw_plan;
    using ComplexType = fftw_complex;
    using FloatType = double;
    static constexpr decltype(&fftw_destroy_plan) destroy_fn = &fftw_destroy_plan;
    static constexpr decltype(&fftw_execute_r2r)     r2rexec = &fftw_execute_r2r;
    static constexpr decltype(&fftw_execute_dft_r2c) r2cexec = &fftw_execute_dft_r2c;
    static constexpr decltype(&fftw_execute_dft_c2r) c2rexec = &fftw_execute_dft_c2r;
    static constexpr decltype(&fftw_execute_dft)     c2cexec = &fftw_execute_dft;
    static constexpr decltype(&fftw_sprint_plan) sprintfn = &fftw_sprint_plan;
    static constexpr decltype(&fftw_fprint_plan) fprintfn = &fftw_fprint_plan;
    static constexpr decltype(&fftw_cleanup)     cleanupfn= &fftw_cleanup;
    static constexpr decltype(&fftw_flops)       flopsfn = &fftw_flops;
    static constexpr decltype(&fftw_cost)        costfn = &fftw_cost;
    static constexpr decltype(&fftw_plan_dft_r2c) r2cplan = &fftw_plan_dft_r2c;
    static constexpr decltype(&fftw_plan_dft_c2r) c2rplan = &fftw_plan_dft_c2r;
    static constexpr decltype(&fftw_plan_dft) c2cplan = &fftw_plan_dft;
    static constexpr decltype(&fftw_plan_r2r) r2rplan = &fftw_plan_r2r;
};
template<>
struct FFTTypes<long double> {
    using PlanType = fftwl_plan;
    using ComplexType = fftwl_complex;
    using FloatType = double;
    static constexpr decltype(&fftwl_destroy_plan) destroy_fn = &fftwl_destroy_plan;
    static constexpr decltype(&fftwl_execute_r2r)     r2rexec = &fftwl_execute_r2r;
    static constexpr decltype(&fftwl_execute_dft_r2c) r2cexec = &fftwl_execute_dft_r2c;
    static constexpr decltype(&fftwl_execute_dft_c2r) c2rexec = &fftwl_execute_dft_c2r;
    static constexpr decltype(&fftwl_execute_dft)     c2cexec = &fftwl_execute_dft;
    static constexpr decltype(&fftwl_sprint_plan) sprintfn = &fftwl_sprint_plan;
    static constexpr decltype(&fftwl_fprint_plan) fprintfn = &fftwl_fprint_plan;
    static constexpr decltype(&fftwl_cleanup)     cleanupfn= &fftwl_cleanup;
    static constexpr decltype(&fftwl_flops)       flopsfn = &fftwl_flops;
    static constexpr decltype(&fftwl_cost)        costfn = &fftwl_cost;
    static constexpr decltype(&fftwl_plan_dft_r2c) r2cplan = &fftwl_plan_dft_r2c;
    static constexpr decltype(&fftwl_plan_dft_c2r) c2cplan = &fftwl_plan_dft_c2r;
    static constexpr decltype(&fftwl_plan_dft) c2rplan = &fftwl_plan_dft;
    static constexpr decltype(&fftwl_plan_r2r) r2rplan = &fftwl_plan_r2r;
};

#define USE_MANUAL 0
#define USE_SSE2   1
#define USE_AVX2   2
#define USE_AVX512 3

#if HAS_AVX_512
#  define ALIGNMENT_SIZE 512
#  define VXFMT   USE_AXV512
#elif __AVX2__
#  define ALIGNMENT_SIZE 256
#  define VXFMT     USE_AXV2
#elif __SSE2__
#  define ALIGNMENT_SIZE 128
#  define VXFMT     USE_SSE2
#else
#  define ALIGNMENT_SIZE  64
#  define VXFMT   USE_MANUAL
#endif

#if !NDEBUG
#define assert_aligned(ptr, aln) do {if(static_cast<decltype(aln)>(ptr) & aln) {std::fprintf(stderr, "ptr " #ptr " at %p is aligned to %zu\n", ptr, aln); exit(EXIT_FAILURE);}} while(0)
#else
#define assert_aligned(ptr, aln)
#endif

enum class Alignment : size_t
{
    Normal = sizeof(void*),
    SSE    = 16,
    AVX    = 32,
    KB     = 64,
    KL     = 64,
    AVX512 = 64
};


enum tx {
    R2HC       = FFTW_R2HC, // “halfcomplex” format, i.e. real and imaginary parts for a transform of size n stored as: r0, r1, r2, ..., rn/2, i(n+1)/2-1, ..., i2, i1
    HC2R       = FFTW_HC2R,
    DHT        = FFTW_DHT,
    REDFT00    = FFTW_REDFT00,
    DCT1       = FFTW_REDFT00,
    REDFT10    = FFTW_REDFT10,
    DCT2       = FFTW_REDFT10,
    DCT        = FFTW_REDFT10,
    R2R        = FFTW_REDFT10, // Default is DCT (R2R/DCT/DCT2/REDFT10)
    REDFT01    = FFTW_REDFT01,
    DCT3       = FFTW_REDFT01,
    REDFT11    = FFTW_REDFT11,
    DCT4       = FFTW_REDFT11,
    RODFT00    = FFTW_RODFT00,
    DST1       = FFTW_RODFT00,
    RODFT10    = FFTW_RODFT10,
    DST2       = FFTW_RODFT10,
    RODFT01    = FFTW_RODFT01,
    DST3       = FFTW_RODFT01,
    RODFT11    = FFTW_RODFT11,
    DST4       = FFTW_RODFT11,
    C2R        = -3,
    R2C        = -2,
    C2C        = -1,
    UNKNOWN    = std::numeric_limits<int>::min()
};

template<typename FloatType>
class FFTWDispatcher {

    using typeinfo = FFTTypes<FloatType>;
    
    std::vector<int>       dims;
    std::vector<tx>       kinds;
    int                  stride;
    bool                forward;
    typename typeinfo::PlanType    plan_;
    double     add_, mul_, fma_;
    tx                      tx_;
    using ComplexType = typename typeinfo::ComplexType;
    static const size_t FloatSize = sizeof(FloatType);
    static const size_t ComplexSize = sizeof(typeinfo::ComplexType);
    

public:
    FFTWDispatcher(std::vector<int> &&dims, bool from_c, bool to_c, tx txarg=DCT, bool forward=true, int stride=1):
        dims(std::move(dims)), stride(stride), forward(forward),
        plan_(nullptr), tx_(txarg)
    {
        if(!from_c && !to_c) {
            assert(tx_ == REDFT00 || tx_ == REDFT10 || tx_ == REDFT01 || tx_ == REDFT11 ||
                   tx_ == RODFT00 || tx_ == RODFT10 || tx_ == RODFT01 || tx_ == RODFT11 ||
                   tx_ == DHT);
            kinds.resize(dims.size(), tx_);
        }
    }

    FFTWDispatcher(int n, bool from_c=false, bool to_c=false, tx txarg=REDFT10, bool forward=true, int stride=1): FFTWDispatcher(std::vector<int>{n}, from_c, to_c, txarg, forward, stride) {}

    template<typename T1, typename T2>
    void make_plan(T1 &in, T2 *out=nullptr, tx tx_=UNKNOWN) {
        make_plan(get_data(in), get_data(out? *out: in), tx_);
    }
    void make_plan(void *in, void *out, tx txarg) {
        tx_ = txarg;
        std::fill(kinds.begin(), kinds.end(), tx_);
        make_plan(in, out);
    }
    void make_plan(void *in, void *out) {
        if(tx_ == UNKNOWN) throw std::runtime_error("Please choose a valid transformation.");
        if(plan_) typeinfo::destroy_fn(plan_);
        switch(static_cast<int>(tx_)) {
            case C2C:
                plan_ = typeinfo::c2cplan(dims.size(), dims.data(), static_cast<ComplexType *>(in), static_cast<ComplexType *>(out), forward ? FFTW_FORWARD: FFTW_BACKWARD, FFTW_MEASURE);
                break;
            case HC2R: case DHT: case REDFT00: case REDFT10: case REDFT01: case REDFT11: case RODFT00: case RODFT10: case RODFT01: case RODFT11: case R2HC:
                assert(kinds.size() == dims.size());
                plan_ = typeinfo::r2rplan(dims.size(), dims.data(), static_cast<FloatType *>(in), static_cast<FloatType *>(out), reinterpret_cast<fftw_r2r_kind*>(kinds.data()), FFTW_MEASURE);
                break;
            case R2C:
                plan_ = typeinfo::r2cplan(dims.size(), dims.data(),
                                        static_cast<FloatType *>(in), static_cast<ComplexType *>(out), FFTW_MEASURE);
                break;
            case C2R:
                plan_ = typeinfo::c2rplan(dims.size(), dims.data(),
                                        static_cast<ComplexType *>(in), static_cast<FloatType *>(out), FFTW_MEASURE);
                break;
            default:
                std::fprintf(stderr, "Unexpected code %i. Abort!\n", static_cast<int>(tx_));
                exit(EXIT_FAILURE);
                break;
        }
    }
    std::string sprintf() {
        char *str(typeinfo::sprintfn(plan_));
        std::string ret(str);
        std::free(str);
        return ret;
    }
    void clear() {
        typeinfo::cleanupfn(plan_);
    }
    double cost() {
        return typeinfo::costfn(plan_);
    }
    double flops() {
        typeinfo::flopsfn(plan_, &add_, &mul_, &fma_);
        return add_ + mul_ + fma_;
    }
    auto printf(FILE *fp) {
        return typeinfo::fprintfn(plan_, fp);
    }
    template<typename DType1, typename DType2>
    void run(DType1 *a, DType2 *b=nullptr) {
        if(!plan_) throw std::runtime_error("Can not execute null plan.");
        void *inp(a), *outp(b ? b: a);
#if 0
    template<typename ValType1, typename ValType2, typename=std::enable_if_t<std::is_floating_point<ValType1>::value>, typename=std::enable_if_t<std::is_floating_point<ValType2>::value>>
    void run(const ValType1 *inp, ValType2 *outp)
#endif
        switch(static_cast<int>(tx_)) {
            case C2C:
                typeinfo::c2cexec(plan_, reinterpret_cast<ComplexType *>(inp), reinterpret_cast<ComplexType *>(outp)); break;
            case R2C:
                typeinfo::r2cexec(plan_, reinterpret_cast<FloatType *>(inp), reinterpret_cast<ComplexType *>(outp)); break;
            case C2R:
                typeinfo::c2rexec(plan_, reinterpret_cast<ComplexType *>(inp), reinterpret_cast<FloatType *>(outp)); break;
            case HC2R:    case R2HC: case DHT:
            case REDFT00: case REDFT10:
            case REDFT01: case REDFT11:
            case RODFT00: case RODFT10:
            case RODFT01: case RODFT11:
                typeinfo::r2rexec(plan_, reinterpret_cast<FloatType *>(inp), reinterpret_cast<FloatType*>(outp)); break;
            default: {
                std::fprintf(stderr, "Unexpected code %i. Abort!\n", static_cast<int>(tx_));
                exit(EXIT_FAILURE);
                break;
            }
        }
    }
    ~FFTWDispatcher() {
        if(plan_) typeinfo::destroy_fn(plan_);
    }
};

} // namespace fft

#endif // #ifdef _FFT_WRAPPER_H_
