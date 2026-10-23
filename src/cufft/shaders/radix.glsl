// The butterflies of a Stockham pass (fft_pass.comp, fft_local.comp), one function per radix so
// every loop has a constant trip count and the arrays stay in registers. Before including, define
//   LOAD(e)      element e of the current line (the pass's input),
//   STORE(e, x)  element e of the line in the pass's output,
// and the roots of unity come from root(m) = exp(-+2 pi i m / len) (the table; conjugated for the
// inverse). With Ns = span and R the radix, butterfly j < len / R takes x_r = LOAD(j + r len / R),
// twiddles x_r by w^(r (j mod Ns)) (w = exp(-+2 pi i / (Ns R))), transforms them with an R-point
// DFT and stores y_q at (j / Ns) Ns R + j mod Ns + q Ns.

cplx root(int m) {
    const cplx w = at(p.table, m).v;
    return inverse() ? conjugate(w) : w;
}

// -i x (forward) or +i x (inverse)
cplx rotate(cplx x) { return inverse() ? cplx(-x.y, x.x) : cplx(x.y, -x.x); }

// The twiddled inputs of butterfly j (v), and where its outputs go.
#define BUTTERFLY_INPUTS(R)                                                                        \
    const int quarter = len / R, k = j % Ns, twiddleStep = len / (Ns * R);                         \
    cplx v[R];                                                                                     \
    for (int r = 0; r < R; ++r) {                                                                  \
        cplx x = LOAD(j + r * quarter);                                                            \
        if (r > 0 && k > 0) x = cmul(x, root(r * k * twiddleStep));                                \
        v[r] = x;                                                                                  \
    }                                                                                              \
    const int first = (j / Ns) * Ns * R + k;

void butterfly2(int j, int Ns, int len) {
    BUTTERFLY_INPUTS(2)
    STORE(first, v[0] + v[1]);
    STORE(first + Ns, v[0] - v[1]);
}

void butterfly4(int j, int Ns, int len) {
    BUTTERFLY_INPUTS(4)
    const cplx t0 = v[0] + v[2], t1 = v[0] - v[2], t2 = v[1] + v[3], t3 = rotate(v[1] - v[3]);
    STORE(first, t0 + t2);
    STORE(first + Ns, t1 + t3);
    STORE(first + 2 * Ns, t0 - t2);
    STORE(first + 3 * Ns, t1 - t3);
}

// Odd prime radices: the DFT by its definition, with the R roots loaded once.
#define DEFINE_BUTTERFLY(R)                                                                        \
    void butterfly##R(int j, int Ns, int len) {                                                    \
        BUTTERFLY_INPUTS(R)                                                                        \
        cplx w[R];                                                                                 \
        for (int m = 0; m < R; ++m) w[m] = root(m * quarter);                                      \
        for (int q = 0; q < R; ++q) {                                                              \
            cplx s = v[0];                                                                         \
            for (int r = 1; r < R; ++r) s += cmul(v[r], w[(r * q) % R]);                           \
            STORE(first + q * Ns, s);                                                              \
        }                                                                                          \
    }

DEFINE_BUTTERFLY(3)
DEFINE_BUTTERFLY(5)
DEFINE_BUTTERFLY(7)
DEFINE_BUTTERFLY(11)
DEFINE_BUTTERFLY(13)

void butterfly(int R, int j, int Ns, int len) {
    switch (R) {
    case 2: butterfly2(j, Ns, len); break;
    case 3: butterfly3(j, Ns, len); break;
    case 4: butterfly4(j, Ns, len); break;
    case 5: butterfly5(j, Ns, len); break;
    case 7: butterfly7(j, Ns, len); break;
    case 11: butterfly11(j, Ns, len); break;
    default: butterfly13(j, Ns, len); break;
    }
}
