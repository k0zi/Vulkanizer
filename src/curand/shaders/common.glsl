// Shared by the generator kernels: the push constants (curand.cpp's Params), the output kinds and
// their stores, and the math the distributions need: accurate f32 (and, with -DDOUBLE, f64) log,
// exp, sin(pi x)/cos(pi x) and the inverse normal CDF, since GLSL's built-ins are not accurate enough
// (and do not exist in f64).
//
// The conversions from raw bits follow cuRAND's host API: uniform floats are x * 2^-32 + 2^-33,
// normals come from the Box-Muller transform (pseudo-random generators) or the inverse normal CDF
// (Sobol), log-normals are exp() of the normals.

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#ifdef DOUBLE
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#endif

layout(local_size_x_id = 0) in;

// The device's address bits 63..47 (vk::Device::hostAddressBits), specialization constant 5:
// addresses arrive without them.
layout(constant_id = 5) const uint kAddressHigh = 0xFFFF8000u;
uint64_t canon(uint64_t a) {
    uvec2 w = unpackUint2x32(a);
    w.y |= kAddressHigh;
    return packUint2x32(w);
}

layout(push_constant, std430) uniform Params {
    uint64_t out_;     // the caller's output
    uint64_t state;    // XORWOW, MRG32k3a: the thread states; Sobol: the direction vectors
    uint64_t aux;      // XORWOW, MRG32k3a init: the jump matrices; Sobol: the scramble constants (or 0)
    uint64_t table;    // Poisson: the CDF thresholds
    uint64_t offset;   // Philox: outputs before this call; stateful: groups before it; Sobol: points
    uint64_t count;    // outputs (Sobol: points per dimension)
    uint64_t seed;
    uint64_t mean, stddev;  // normal: the parameters (f32 in the low word, or f64 bits); Poisson: lambda
    uint kind;
    uint flags;
    uint threads;      // stateful generators: the thread count
    uint dims;         // Sobol: dimensions
    uint tableSize;    // Poisson: CDF entries
    uint tableBase;    // Poisson: the value of the first entry
} p;

const uint kU32 = 0u, kU64 = 1u, kUniform = 2u, kUniformDouble = 3u, kNormal = 4u, kNormalDouble = 5u,
           kLogNormal = 6u, kLogNormalDouble = 7u, kPoisson = 8u;
const uint kFlagInit = 1u;    // stateful: compute the thread states instead of generating
const uint kFlagSobol64 = 2u;
const uint kFlagMtgp = 4u;    // Philox kernel standing in for MTGP32/MT19937 (another key)

layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32 { uint v; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer U64 { uint64_t v; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v; };

uint64_t invocation() {
    return uint64_t(gl_GlobalInvocationID.x) + uint64_t(gl_GlobalInvocationID.y) * gl_NumWorkGroups.x * gl_WorkGroupSize.x;
}

void storeU32(uint64_t i, uint v) { U32(canon(p.out_ + i * 4ul)).v = v; }
void storeU64(uint64_t i, uint64_t v) { U64(canon(p.out_ + i * 8ul)).v = v; }
void storeF32(uint64_t i, float v) { F32(canon(p.out_ + i * 4ul)).v = v; }

bool doubleKind(uint kind) { return kind == kUniformDouble || kind == kNormalDouble || kind == kLogNormalDouble; }
bool normalKind(uint kind) { return kind >= kNormal && kind <= kLogNormalDouble; }

// ---- f32 math -------------------------------------------------------------------------------------

// log(x) for x > 0: x = m 2^e with m in [sqrt(1/2), sqrt(2)), log(m) = 2 atanh((m - 1) / (m + 1)).
float logf_(float x) {
    int e;
    float m = frexp(x, e);
    if (m < 0.70710678) {
        m *= 2.0;
        e -= 1;
    }
    precise float f = m - 1.0;
    precise float s = f / (2.0 + f);
    precise float s2 = s * s;
    precise float r = s2 * (0.66666667 + s2 * (0.4 + s2 * (0.28571429 + s2 * 0.22222222)));
    return float(e) * 0.693145751953125 + (float(e) * 1.4286068e-06 + (2.0 * s + s * r));
}

// exp(x): x = k log(2) + r, |r| <= log(2) / 2.
float expf_(float x) {
    float k = round(x * 1.4426950409);
    precise float r = (x - k * 0.693145751953125) - k * 1.4286068e-06;
    precise float e = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * (0.041666667 + r * (0.0083333333 +
                      r * (0.0013888889 + r * 0.00019841270))))));
    return ldexp(e, int(k));
}

// sin(pi x), cos(pi x): x = n / 2 + r with |r| <= 1/4, Taylor series in pi r.
void sincospif_(float x, out float s, out float c) {
    float n = round(2.0 * x);
    precise float r = (x - 0.5 * n) * 3.14159265;
    precise float r2 = r * r;
    precise float sr = r * (1.0 - r2 * (0.16666667 - r2 * (0.0083333333 - r2 * (0.00019841270 - r2 * 2.7557319e-06))));
    precise float cr = 1.0 - r2 * (0.5 - r2 * (0.041666667 - r2 * (0.0013888889 - r2 * (2.4801587e-05 - r2 * 2.7557319e-07))));
    uint q = uint(int(n)) & 3u;
    s = q == 0u ? sr : q == 1u ? cr : q == 2u ? -sr : -cr;
    c = q == 0u ? cr : q == 1u ? -sr : q == 2u ? -cr : sr;
}

// The inverse normal CDF: in the tails P. J. Acklam's rational approximation (relative error below
// 1.2e-9), in the middle M. Giles' single precision erfinv ("Approximating the erfinv function",
// GPU Computing Gems Jade, 2011), which unlike Acklam's central part keeps its accuracy in f32.
float acklamTail(float q) {
    return (((((-7.784894002430293e-03 * q - 3.223964580411365e-01) * q - 2.400758277161838e+00) * q -
              2.549732539343734e+00) * q + 4.374664141464968e+00) * q + 2.938163982698783e+00) /
           ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q + 2.445134137142996e+00) * q +
             3.754408661907416e+00) * q + 1.0);
}
float normcdfinvf_(float u) {
    if (u < 0.02425) return acklamTail(sqrt(-2.0 * logf_(u)));
    if (u > 0.97575) return -acklamTail(sqrt(-2.0 * logf_(1.0 - u)));
    // sqrt(2) erfinv(2u - 1), with log((1 - x)(1 + x)) = log(4u(1 - u))
    precise float w = -logf_(4.0 * u * (1.0 - u)) - 2.5;
    precise float r = 2.81022636e-08;
    r = 3.43273939e-07 + r * w;
    r = -3.5233877e-06 + r * w;
    r = -4.39150654e-06 + r * w;
    r = 0.00021858087 + r * w;
    r = -0.00125372503 + r * w;
    r = -0.00417768164 + r * w;
    r = 0.246640727 + r * w;
    r = 1.50140941 + r * w;
    precise float x = 1.41421356 * (r * (2.0 * u - 1.0));
    return x;
}

// x * 2^-32 + 2^-33, rounded once per operation as cuRAND does.
float uniformf(uint x) {
    precise float v = float(x) * 2.3283064e-10 + 1.1641532e-10;
    return v;
}

// The Box-Muller pair of two words.
vec2 boxMullerf(uint x, uint y) {
    float u = uniformf(x);
    precise float v = float(y) * 4.6566129e-10 + 2.3283064e-10;
    float r = sqrt(-2.0 * logf_(u));
    float s, c;
    sincospif_(v, s, c);
    precise vec2 n = vec2(s * r, c * r);
    return n;
}

float scalef(float n) {
    precise float v = n * uintBitsToFloat(uint(p.stddev)) + uintBitsToFloat(uint(p.mean));
    return p.kind == kLogNormal ? expf_(v) : v;
}

// ---- f64 math -------------------------------------------------------------------------------------
#ifdef DOUBLE
layout(buffer_reference, std430, buffer_reference_align = 8) buffer F64 { double v; };
void storeF64(uint64_t i, double v) { F64(canon(p.out_ + i * 8ul)).v = v; }

const double kLn2Hi = 6.93147180369123816490e-01LF, kLn2Lo = 1.90821492927058770002e-10LF;
const double kPi = 3.14159265358979323846LF;

double logd_(double x) {
    int e;
    double m = frexp(x, e);
    if (m < 0.70710678118654752LF) {
        m *= 2.0LF;
        e -= 1;
    }
    precise double f = m - 1.0LF;
    precise double s = f / (2.0LF + f);
    precise double s2 = s * s;
    precise double r = 0.0LF;
    for (int k = 23; k >= 3; k -= 2) r = s2 * (2.0LF / double(k) + r);
    return double(e) * kLn2Hi + (double(e) * kLn2Lo + (2.0LF * s + s * r));
}

double expd_(double x) {
    double k = round(x * 1.44269504088896340736LF);
    precise double r = (x - k * kLn2Hi) - k * kLn2Lo;
    precise double e = 1.0LF;
    for (int i = 13; i >= 1; --i) e = 1.0LF + r * e / double(i);
    return ldexp(e, int(k));
}

void sincospid_(double x, out double s, out double c) {
    double n = round(2.0LF * x);
    precise double r = (x - 0.5LF * n) * kPi;
    precise double r2 = r * r;
    precise double sr = 1.0LF, cr = 1.0LF;
    for (int k = 22; k >= 2; k -= 2) {
        sr = 1.0LF - r2 * sr / double((k + 1) * k);
        cr = 1.0LF - r2 * cr / double(k * (k - 1));
    }
    sr *= r;
    uint q = uint(int(n)) & 3u;
    s = q == 0u ? sr : q == 1u ? cr : q == 2u ? -sr : -cr;
    c = q == 0u ? cr : q == 1u ? -sr : q == 2u ? -cr : sr;
}

// erfc(z) for z >= 0: 1 - erf(z) from erf's series of positive terms below 1.5, the continued
// fraction above (relative error about 1e-14).
double erfcd_(double z) {
    if (z < 1.5LF) {
        precise double t = z, s = z;
        for (int n = 1; n < 45; ++n) {
            t *= 2.0LF * z * z / double(2 * n + 1);
            s += t;
        }
        return 1.0LF - 1.12837916709551257390LF * expd_(-z * z) * s;
    }
    precise double f = z;
    for (int k = 80; k >= 1; --k) f = z + 0.5LF * double(k) / f;
    return 0.56418958354775628695LF * expd_(-z * z) / f;
}

// The inverse normal CDF: Acklam's approximation and a Halley step on 0.5 erfc(-x / sqrt(2)).
double normcdfinvd_(double u) {
    bool upper = u > 0.5LF;
    double q = upper ? 1.0LF - u : u;
    double x;
    if (q < 0.02425LF) {
        double t = sqrt(-2.0LF * logd_(q));
        x = (((((-7.784894002430293e-03LF * t - 3.223964580411365e-01LF) * t - 2.400758277161838e+00LF) * t -
               2.549732539343734e+00LF) * t + 4.374664141464968e+00LF) * t + 2.938163982698783e+00LF) /
            ((((7.784695709041462e-03LF * t + 3.224671290700398e-01LF) * t + 2.445134137142996e+00LF) * t +
              3.754408661907416e+00LF) * t + 1.0LF);
    } else {
        double h = q - 0.5LF, r = h * h;
        x = (((((-3.969683028665376e+01LF * r + 2.209460984245205e+02LF) * r - 2.759285104469687e+02LF) * r +
               1.383577518672690e+02LF) * r - 3.066479806614716e+01LF) * r + 2.506628277459239e+00LF) * h /
            (((((-5.447609879822406e+01LF * r + 1.615858368580409e+02LF) * r - 1.556989798598866e+02LF) * r +
               6.680131188771972e+01LF) * r - 1.328068155288572e+01LF) * r + 1.0LF);
    }
    if (q > 0.0LF) {
        precise double e = 0.5LF * erfcd_(-x * 0.70710678118654752440LF) - q;
        precise double w = e * 2.50662827463100050242LF * expd_(0.5LF * x * x);
        x = x - w / (1.0LF + 0.5LF * x * w);
    }
    return upper ? -x : x;
}

// (x ^ y << 21) * 2^-53 + 2^-54: 53 random bits from two words.
double uniformd(uint x, uint y) {
    uint64_t z = uint64_t(x) ^ (uint64_t(y) << 21);
    precise double v = double(z) * 1.1102230246251565e-16LF + 5.5511151231257827e-17LF;
    return v;
}

dvec2 boxMullerd(uint x0, uint x1, uint x2, uint x3) {
    double u = uniformd(x0, x1);
    uint64_t z = uint64_t(x2) ^ (uint64_t(x3) << 21);
    precise double v = double(z) * 2.2204460492503131e-16LF + 1.1102230246251565e-16LF;
    double r = sqrt(-2.0LF * logd_(u));
    double s, c;
    sincospid_(v, s, c);
    precise dvec2 n = dvec2(s * r, c * r);
    return n;
}

double scaled(double n) {
    precise double v = n * uint64BitsToDouble(p.stddev) + uint64BitsToDouble(p.mean);
    return p.kind == kLogNormalDouble ? expd_(v) : v;
}
#endif

// ---- Poisson --------------------------------------------------------------------------------------

// Inversion: the table holds floor(CDF(tableBase + i) * 2^33); the value of the word x is the first
// entry above 2x + 1 (the midpoint of x's interval of width 2^-32). Past the table's end, lambda is
// large and the normal approximation takes over.
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Cdf { uint64_t v; };
uint poisson(uint x) {
    uint64_t v = 2ul * uint64_t(x) + 1ul;
    if (p.tableSize == 0u) {
        float lambda = uintBitsToFloat(uint(p.mean));
        float n = normcdfinvf_(uniformf(x));
        return uint(max(0.0, round(lambda + sqrt(lambda) * n)));
    }
    uint lo = 0u, hi = p.tableSize;  // first index with v < cdf[index]
    while (lo < hi) {
        uint mid = (lo + hi) / 2u;
        if (v < Cdf(canon(p.table + uint64_t(mid) * 8ul)).v) hi = mid;
        else lo = mid + 1u;
    }
    return p.tableBase + lo;
}
