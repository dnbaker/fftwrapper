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
#include "fftw3.h"


namespace fft {

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

// Overloads for executing an fft
void execute_fft(fftw_plan plan, fftw_complex *in, fftw_complex *out) {
    fftw_execute_dft(plan, in, out);
}
void execute_fft(fftw_plan plan, double *in, fftw_complex *out) {
    fftw_execute_dft_r2c(plan, in, out);
}
void execute_fft(fftw_plan plan, fftw_complex *in, double *out) {
    fftw_execute_dft_c2r(plan, in, out);
}
void execute_fft(fftw_plan plan, double *in, double *out) {
    fftw_execute_r2r(plan, in, out);
}

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
#define assert_aligned(ptr, aln) do {if(ptr & aln) {std::fprintf(stderr, "ptr " #ptr " at %p is aligned to %zu\n", ptr, aln); exit(EXIT_FAILURE);}} while(0)
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


enum transformation {
    R2HC       = FFTW_R2HC, // “halfcomplex” format, i.e. real and imaginary parts for a transform of size n stored as: r0, r1, r2, ..., rn/2, i(n+1)/2-1, ..., i2, i1
    HC2R       = FFTW_HC2R,
    DHT        = FFTW_DHT,
    REDFT00    = FFTW_REDFT00,
    REDFT10    = FFTW_REDFT10,
    REDFT01    = FFTW_REDFT01,
    REDFT11    = FFTW_REDFT11,
    RODFT00    = FFTW_RODFT00,
    RODFT10    = FFTW_RODFT10,
    RODFT01    = FFTW_RODFT01,
    RODFT11    = FFTW_RODFT11,
    R2R        = REDFT10,
    C2R        = -3,
    R2C        = -2,
    C2C        = -1
};

class FFTWDispatcher {
    std::vector<int>       dims;
    std::vector<transformation>  kinds;
    size_t               stride;
    //bool           use_floats;
    bool                forward;
    bool      owns_in, owns_out;
    fftw_plan             plan_;
    double     add_, mul_, fma_;
    transformation           tx;
    void              *in, *out;

public:
    FFTWDispatcher(std::vector<int> &&dims, bool from_c, bool to_c, transformation tx=REDFT10, bool forward=true, void *din=nullptr, void *dout=nullptr):
        dims(dims), stride(std::accumulate(dims.begin(), dims.end(), UINT64_C(0))),
        owns_in(!din), owns_out(!dout), plan_(nullptr),
        //use_floats(use_f), 
        forward(forward), tx(tx)
    {
        if(from_c) {
            if(owns_in) in = almalloc(ALIGNMENT_SIZE, sizeof(fftw_complex) * stride);
            else        in = din;
            if(to_c) {
                tx = C2C;
                if(owns_out) out = almalloc(ALIGNMENT_SIZE, sizeof(fftw_complex) * stride);
                else out = dout;
            } else {
                tx = C2R;
                if(owns_out) out = almalloc(ALIGNMENT_SIZE, sizeof(double) * stride);
                else out = dout;
            }
        } else {
            if(owns_in) in = almalloc(ALIGNMENT_SIZE, sizeof(fftw_complex) * stride);
            else in = din;
            if(to_c) {
                tx = R2C;
                if(owns_out) out = almalloc(ALIGNMENT_SIZE, sizeof(fftw_complex) * stride);
                else out = dout;
            } else {
                if(owns_out) out = almalloc(ALIGNMENT_SIZE, sizeof(double) * stride);
                else out = dout;
                assert(tx == REDFT00 || tx == REDFT10 || tx == REDFT01 || tx == REDFT11 ||
                       tx == RODFT00 || tx == RODFT10 || tx == RODFT01 || tx == RODFT11 ||
                       tx == DHT);
                while(kinds.size() < dims.size()) kinds.push_back(tx);
            }
        }
        make_plan();
    }
    FFTWDispatcher(size_t n, bool from_c=false, bool to_c=false, transformation tx=REDFT10, bool forward=true): FFTWDispatcher(std::vector<int>{n}, from_c, to_c, tx, forward, nullptr, nullptr) {}

    void make_plan() {
        if(plan_) fftw_destroy_plan(plan_);
        static_assert(sizeof(fftw_r2r_kind) == sizeof(transformation), "Failed transformation assert");
        static_assert(alignof(fftw_r2r_kind) == alignof(transformation), "Failed alignment assert");
        switch(static_cast<int>(tx)) {
            case C2C:
                plan_ = fftw_plan_dft(dims.size(), dims.data(), static_cast<fftw_complex *>(in), static_cast<fftw_complex *>(out), forward ? FFTW_FORWARD: FFTW_BACKWARD, FFTW_MEASURE);
                break;
            case HC2R: case DHT: case REDFT00: case REDFT10: case REDFT01: case REDFT11: case RODFT00: case RODFT10: case RODFT01: case RODFT11: case R2HC:
                assert(kinds.size() == dims.size());
                plan_ = fftw_plan_r2r(dims.size(), dims.data(), static_cast<double *>(in), static_cast<double *>(out), reinterpret_cast<fftw_r2r_kind*>(kinds.data()), FFTW_MEASURE);
                break;
            case R2C:
                plan_ = fftw_plan_dft_r2c(dims.size(), dims.data(),
                                          static_cast<double *>(in), static_cast<fftw_complex *>(out), FFTW_MEASURE);
                break;
            case C2R:
                plan_ = fftw_plan_dft_c2r(dims.size(), dims.data(),
                                          static_cast<fftw_complex *>(in), static_cast<double *>(out), FFTW_MEASURE);
                break;
            default:
                std::fprintf(stderr, "Unexpected code %i. Abort!\n", static_cast<int>(tx));
                exit(EXIT_FAILURE);
                break;
        }
    }
    std::string sprintf() {
        char *str(fftw_sprint_plan(plan_));
        std::string ret(str);
        std::free(str);
        return ret;
    }
    void clear() {
        fftw_cleanup();
    }
    double cost() {
        return fftw_cost(plan_);
    }
    double flops() {
        fftw_flops(plan_, &add_, &mul_, &fma_);
        return add_ + mul_ + fma_;
    }
    void printf(FILE *fp) {
        return fftw_fprint_plan(plan_, fp);
    }
    void assign(void *new_in, void *new_out) {
        assert_aligned(in, ALIGNMENT_SIZE);
        assert_aligned(out, ALIGNMENT_SIZE);
        if(owns_in)  alfree(in);
        if(owns_out) alfree(out);
        in = new_in, out = new_out;
        owns_in = owns_out = true;
    }
    void run() {
        run(in, out);
    }
    void run(void *inp, void *outp) {
        if(!plan_) throw std::runtime_error("Can not execute null plan.");
        run(in, out);
        switch(static_cast<int>(tx)) {
            case C2C: {
                execute_fft(plan_, static_cast<fftw_complex *>(in), static_cast<fftw_complex *>(out)); break;
            }
            case R2C: {
                execute_fft(plan_, static_cast<double *>(in), static_cast<fftw_complex *>(out)); break;
            }
            case HC2R:    case R2HC: case DHT:
            case REDFT00: case REDFT10:
            case REDFT01: case REDFT11:
            case RODFT00: case RODFT10:
            case RODFT01: case RODFT11: {
                execute_fft(plan_, static_cast<double *>(in), static_cast<double*>(out)); break;
            }
            default: {
                std::fprintf(stderr, "Unexpected code %i. Abort!\n", static_cast<int>(tx));
                exit(EXIT_FAILURE);
                break;
            }
        }
    }
    ~FFTWDispatcher() {
        if(plan_) fftw_destroy_plan(plan_);
    }
};

} // namespace fft

#endif // #ifdef _FFT_WRAPPER_H_
