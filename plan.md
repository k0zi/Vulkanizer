# Vulkanizer – CUDA Driver API Vulkanon

> Cél: egy `libcuda.so` drop-in csere, ami az alkalmazás felé NVIDIA CUDA drivernek látszik,
> de minden memóriaműveletet és kernelt Vulkan compute-on futtat. Így CUDA-alapú szoftverek
> (végső soron AI toolok, pl. PyTorch) olyan GPU-kon is futhatnak, amik nem támogatják a CUDA-t / ROCm-et.

## 1. Architektúra

```
CUDA alkalmazás (nvcc -cudart shared)
   ▼
libcudart.so.12 ◄── Vulkanizer (src/runtime)       ← Runtime API: cudaMalloc, kernel<<<>>>, regisztráció
   ▼
libcuda.so.1  ◄── Vulkanizer (src/driver)          ← Driver API: cuInit, cuMemAlloc, cuLaunchKernel, ...
   ├─ src/vk        Vulkan réteg: instance, device, bufferek (BDA), submit, compute pipeline-ok
   ├─ src/ptx       PTX lexer + parser (modul → entry-k, paraméterek, később utasítások)
   ├─ src/compiler  PTX → SPIR-V (Vulkan "Shader" execution model) fordító
   └─ (később) libcublas.so / libcudnn.so / ... shim-ek saját Vulkan kernelekkel
```

Betöltés: `LD_LIBRARY_PATH=build/lib` → az app a mi `libcudart.so.12`-nket és `libcuda.so.1`-ünket kapja.
Az exportált szimbólumokat version script szabályozza (`src/driver/libcuda.map`, csak `cu*`).

### Kulcsdöntések

| Téma | Döntés |
|---|---|
| Nyelv | C++20, CMake, nincs külső függőség a Vulkan SDK-n kívül |
| Header | Saját, clean-room `include/cuda.h` (csak a nyilvános dokumentáció alapján; az enum értékek ABI-kompatibilisek) |
| Device pointer | `CUdeviceptr` = Vulkan buffer device address (`VK_KHR_buffer_device_address`), így a pointer-aritmetika a kernelben természetes |
| Allokáció | A ≤ 4 MiB foglalások 64 MiB-os blokkokból (sub-allokátor), a nagyobbak saját VkBuffer + VkDeviceMemory; címtartomány-térkép a pointer → buffer kereséshez. Pinned/managed memória: host-memória, a device ugyanazon a címen éri el (5.2) |
| Kernel ABI | A CUDA paraméter-buffer bájtra pontosan a push constant blokkba kerül (≤128/256 B); fölötte később UBO / BDA-s param buffer |
| Block méret | Specialization constant (ID 0,1,2 = blockDim.x/y/z) → pipeline cache blockDim-enként |
| Dinamikus shared mem | Specialization constant-tal méretezett workgroup tömb (ID 3, 32 bites szavakban) |
| Block packing | Barrier és shared memory nélküli kernelnél (natív warpokkal) több kis CUDA block osztozik egy workgroupon: spec constant 4 = K, a workgroup `(K·blockDim.x, y, z)`, a grid `ceil(gridDim.x / K)`; a rács utáni blockok azonnal kilépnek |
| Végrehajtás | Egy compute queue; a munka batch-ekben (command buffer ring) megy ki, a streamek ezen osztoznak (3.7), a host-hozzáférés kiüríti a sort |
| Warp | A CUDA warp mindig 32. Subgroup 32 → direkt map; 64 → subgroup = 2 warp, maszkolt ballot/shuffle; <32 → emuláció shared memorián keresztül (részleges) |
| Compute capability | Alapból 7.5-öt jelentünk (`VULKANIZER_CC=86` stb. felülírja) – minél alacsonyabb, annál kevesebb PTX feature kell |

### Teszt hardver (fejlesztő gép)

- AMD Radeon (Renoir, Vega/GCN5) iGPU, Mesa RADV – subgroup **64** fix, UMA memória (DEVICE_LOCAL|HOST_VISIBLE)
- llvmpipe (CPU Vulkan) – subgroup **8** – második, nagyon eltérő célpont (`VULKANIZER_DEVICE=llvmpipe`)
- Nincs `nvcc` → a teszt PTX-ek a repóban vannak (`tests/data/*.ptx`). Később: `clang -x cuda --cuda-device-only -nocudainc -nocudalib -S` PTX generálásra CUDA SDK nélkül.

## 2. Fázisok és lépések

Minden lépésnél a **Kész, ha** sor a lezárás feltétele. A sorrend egyben a javasolt munkasorrend.

### Állapot (2026-10-23)

Kész: a 0–4. fázis a 3.6 és a 4.2 kivételével, az 5. fázisból az 5.2 (managed/pinned memória), az 5.3 (CUDA Graphs),
az 5.4 (teljesítmény), az 5.5 csomagolási része (`.deb`, `.tar.gz`, `vulkanizer-run`), az 5.6 (cooperative
groups, több GPU, IPC, NCCL, UVA), valamint az 5.7 (llama tg ~31 → ~38 t/s). Ami fut: nvcc-vel fordított programok (CUDA 12 és 13), llama.cpp, PyTorch (CUDA 12 és 13,
Triton, `torch.compile`), ComfyUI (Z-Image Turbo).

Hátra, javasolt sorrendben:
1. **5.1** cuDNN shim – halasztva (indoklás az 5.1-nél).
2. **4.2** További könyvtárak: a `libcusolver`, `libcufft` és `libcusparse` kész; a `libcurand` csak ha kell.
3. **3.6** Textúrák / surface-ek – ha egy célalkalmazás kéri.
4. **5.5** Jogi átnézés publikálás előtt (a csomagolás kész).
5. Nyitott részfeladatok: rögzített Vulkan command bufferek, child graph, memória-csomópontok (5.3); külön copy
   queue (5.4).

### Fázis 0 – Váz ✅
- [x] CMake projekt, build/test/run/clean scriptek
- [x] Clean-room `cuda.h` (subset)
- [x] Vulkan réteg: device választás, BDA bufferek, copy/fill, compute pipeline + dispatch
- [x] Driver API: init, device query, context (primary + saját), memória, stream/event (szinkron), cuGetProcAddress
- [x] PTX lexer + modul-szintű parser (.version/.target/.entry + paraméter layout, globálisok)
- [x] `cuLaunchKernel` végig bekötve – csak a fordító hiányzik (stub)
- [x] Tesztek: device query, memória, PTX modul, Vulkan BDA dispatch (GLSL), vectorAdd (disabled amíg nincs fordító)

### Fázis 0.5 – Infrastruktúra ✅
- [x] **0.5.1** `git init`, első commit.
  Kész, ha: a repo tiszta, a `build/` ignorálva.
- [x] **0.5.2** `ci.sh`: `./test.sh --all-devices --asan`, majd `--release` build + teszt. GitHub Actions később (llvmpipe-on).
  Kész, ha: `./ci.sh` egy paranccsal végigfut és 0-val tér vissza.
- [x] **0.5.3** Opcionális eszközök a README-ben: `clang` (PTX generálás CUDA SDK nélkül), `vulkan-validationlayers`, `ninja`.
  Kész, ha: a README felsorolja őket telepítési paranccsal.
- [x] **0.5.4** `tests/data/` konvenció: minden `.ptx` mellé `.cu` forrás és a generáló parancs (fejléc-kommentben).
  Kész, ha: a `vector_add.ptx` mellett ott a `vector_add.cu`.
- [x] **0.5.5** `tools/vulkanizer-ptxc`: PTX → SPIR-V CLI, opcionálisan `spirv-dis`/`spirv-val` futtatással. Ez a fordító fő fejlesztői eszköze.
  Kész, ha: `vulkanizer-ptxc x.ptx -o x.spv` lefut (eleinte a stub hibájával).

### Fázis 1 – Első valódi kernel (`vectorAdd`) ✅
- [x] **1.1 PTX utasítás-parser**
  - `.reg` deklarációk (`%r<10>` tartományok is), címkék.
  - Utasítások: opcode + módosítók (`.s32`, `.rn`, `.global`, `.wide`…), predikátum (`@%p1`, `@!%p1`).
  - Operandusok: reg, imm, `[addr+off]`, vektor `{a,b}`.
  Kész, ha: a `tests/ptx_parse.cpp` a `vector_add.ptx` minden utasítását helyesen bontja.
- [x] **1.2 IR**: alapblokkok utasítás-listával, regiszter = Function storage változó (nincs SSA), CFG (succ/pred).
  Kész, ha: az IR dumpolható (`vulkanizer-ptxc --dump-ir`), és a CFG helyes a tesztkernelekre.
- [x] **1.3 SPIR-V emitter alap** (`src/compiler/spirv_builder.{h,cpp}`, SPIRV-Headers `spirv.hpp`)
  - típus/konstans dedup, `PhysicalStorageBuffer64`;
  - push constant struct a paraméter-layoutból (`Entry::params`);
  - spec constant `LocalSize`, builtinek (`%tid`, `%ntid`, `%ctaid`, `%nctaid`).
  Kész, ha: egy üres kernel `spirv-val`-on átmegy, és a pipeline létrejön.
- [x] **1.4 Alap utasítások**: `ld.param`, `ld/st.global`, `mov`, `add/sub/mul(.lo/.hi/.wide)/mad`, `setp`, `selp`, `cvt` (int↔int, int↔float), `cvta.to.global`, `shl/shr/and/or/xor/not`, `bra`, `ret`.
  Kész, ha: mindegyikre van legalább egy PTX teszt.
- [x] **1.5 Vezérlési folyam strukturálása**
  - Első kör: if/else és egyszerű loop minták (dominátor-fa + `OpSelectionMerge`/`OpLoopMerge`).
  - Fallback: "loop + switch" állapotgép tetszőleges CFG-re.
  Kész, ha: a predikált kilépés és a grid-stride loop helyes eredményt ad.
- [x] **1.6 Validálás**: debug buildben minden fordítás után `spirv-val --target-env vulkan1.2` (CLI-n át, `VULKANIZER_SPIRV_VAL=1`).
  Kész, ha: érvénytelen SPIR-V esetén a hiba a logban és a `cuModuleLoadDataEx` error logban is megjelenik.
- [x] **1.7 Mérföldkő:** a `vector_add` teszt engedélyezve van.
  Kész, ha: zöld RADV-n és llvmpipe-on.
- [x] **1.8 Tesztek bővítése**: saxpy, 2D index (`ctaid.y`), predikált kilépés, grid-stride loop, int64 aritmetika.
  Kész, ha: mind zöld mindkét device-on, ASAN-nal is.
- [x] **1.9 Fatbin**
  - a `__cudaFatBinaryWrapper` és a fatbin header parse-olása;
  - a PTX bejegyzés kiválasztása (a legmagasabb `sm_XX`, ami ≤ a jelentett CC);
  - LZ4-tömörített változat (saját kis dekóder).
  Kész, ha: egy kézzel összerakott fatbin (`tests/data`) betölthető és futtatható.

### Fázis 2 – CUDA Runtime kompatibilitás (előrehozva)
Enélkül egyetlen nvcc-vel fordított, normál bináris sem indul el, ezért a valódi kernel-feature-ök elé került.

**Döntés (2026-10-20):** az NVIDIA libcudart a drivertől két nem dokumentált belső táblát kér
(`cuGetExportTable`, UUID `6bd5fb6c…` és `f8cff951…`), ezek nélkül nem inicializál. Ezek helyett
**saját libcudart** készül (`src/runtime`, `libcudart.so.12`), csak a nyilvános Runtime API-ra és az
nvcc által generált regisztrációs ABI-ra építve. Ára: a runtime-ot statikusan linkelő binárisok
(nvcc alapértelmezés) nem futnak; `-cudart shared`-del fordítottak (és pl. a PyTorch) igen.

- [x] **2.1** `cuGetExportTable` probe: logolja a kért interfészeket és hibát ad.
- [x] **2.2** Valódi nvcc toolchain helyben (`tools/setup_cuda.sh`, GPU és driver nélkül), az nvcc által generált host stubok
  (`__cudaRegisterFatBinary`, `__cudaRegisterFunction`, `__cudaPush/PopCallConfiguration`, `cudaLaunchKernel`) feltérképezve.
- [x] **2.3** Saját libcudart: regisztráció, lusta modul-betöltés eszközönként, launch, eszközök (`cudaDeviceProp`),
  memória, streamek, eventek, hibakezelés. A típusok clean-room-ok; az ABI-egyezést a `cudart_abi` teszt ellenőrzi a hivatalos fejlécek ellen.
- [x] **2.4 Mérföldkő:** nvcc-vel fordított programok (`tests/cuda/*.cu`) változtatás nélkül futnak RADV-n és llvmpipe-on.
- [x] **2.5** További Runtime API a célalkalmazások igénye szerint:
  - [x] `__device__`/`__constant__` változók: `__cudaRegisterVar`, `cudaMemcpyTo/FromSymbol`, `cudaGetSymbolAddress/Size`;
  - [x] `cudaMemcpy2D`, `cudaPointerGetAttributes`, `cudaOccupancy*`, `cudaLaunchHostFunc`; a `cudaMallocManaged`
    `cudaErrorNotSupported`-et ad, a managed memória az 5.2 része.
  Kész, ha: a kiválasztott célalkalmazás (4.3) indulásakor nincs hiányzó runtime szimbólum.

### Fázis 3 – Valódi kernelek
- [x] **3.1** Shared memory: statikus és dinamikus (`extern`, spec constant 3), sub-word store atomikusan; `bar.sync`, `membar`/`fence`.
- [x] **3.2** Atomics: `atom`/`red` global és shared tárra (add/min/max/inc/dec/exch/cas/and/or/xor, 32/64 bit, f32 CAS-ciklussal).
  Hiányzik: 64 bites atomics shared memórián.
- [x] **3.3** Local memory (per-szál word tömb), `.func` hívások PTX-szintű inline-olása, `.global`/`.const` modulváltozók inicializálóval, `cuModuleGetGlobal`, generikus címek (shared/local ablak tag-gel).
  Hiányzik: rekurzió, indirekt hívás.
- [x] **3.4 Warp műveletek**: `shfl.sync`, `vote.sync`, `activemask`, `match`, `redux.sync` (egész), `%lanemask_*`.
  - subgroup 32 és 64 (RADV): a warp a subgroup 32-es szelete;
  - subgroup < 32 (llvmpipe): shared memórián át, workgroup barrierrel. Csak akkor helyes, ha az egész blokk egységesen hajtja végre.
  A `redux.sync` lane-enként, a maszk szerint gyűjt (mint a `match`), így részleges maszkkal is helyes (`tests/warp_matrix.cpp`).
  Hiányzik: `redux.sync` f32 min/max (sm_100a).
- [x] **3.5 Math**
  - `sqrt/rsqrt/rcp/ex2/lg2/sin/cos/tanh`, `copysign`, `testp`, `popc/clz/brev/bfind/bfe/bfi/prmt/lop3/shf`, carry-láncok, `sad/cnot/slct/set`;
  - f16/bf16 konverzió és aritmetika (f32-ben, pontos RNE).
  - Minden float művelet `NoContraction` (a Mesa különben átrendezi, ami elrontotta az `expf`-et).
  - Az f32 `fma` f64-ben számol (sem a RADV, sem az llvmpipe nem fused, ami elrontotta a `sinf`-et).
- [ ] **3.6** Textúrák / surface-ek. Alacsony prioritás, csak ha egy célalkalmazás igényli.
- [x] **3.7 Aszinkron végrehajtás**: a kernelek batch-ekben futnak (command buffer ring, köztük barrier). A host-oldali műveletek előtte kiürítik a sort. Az eventek GPU timestamp-et használnak, a streamek egy queue-n osztoznak.
  Eredmény: 66 → 7 µs/launch a Renoir iGPU-n.
  Hiányzik: valódi párhuzamos streamek, host callbackek (`cuLaunchHostFunc`), aszinkron memcpy.
- [x] **3.8** Ha a grid nagyobb a `maxComputeWorkGroupCount`-nál: `vkCmdDispatchBase`-szel darabolva, a `%nctaid` push constantból jön.
- [x] **3.9** Ha a paraméter-buffer nem fér el a push constantban: batch-enkénti feltöltési aréna, a push constantban csak a cím megy át. Dinamikus `ld.param` is működik.
- [x] **3.10** Kernel cache lemezen (`src/driver/kernel_cache.cpp`): a lefordított SPIR-V és metaadatai a
  `~/.cache/vulkanizer/kernels` alatt, kulcs: a libcuda build-ID-je + a PTX modul hash-e + kernelnév + target-jellemzők
  (`VULKANIZER_CACHE_DIR=<dir>|off`). A pipeline-okat a Mesa saját shader cache-e tárolja. Méretkorlát LRU-val
  (`VULKANIZER_CACHE_SIZE`, alap 256 MiB): a korábbi buildek bejegyzései így kikerülnek.
- [x] **3.11 Teszt korpusz** (`tests/cuda/rt_kernels.cu`, CPU referenciával): reduction, tiled matmul, histogram, atomics, warp-műveletek, változók és generikus pointerek, math, bitműveletek, half, device függvényhívások, launch limitek. Plusz `rt_printf`.
- [x] **3.12** Device oldali `printf` (launch-onkénti buffer, host oldali formázás).
- [x] **3.13 Korán kilépő szálak és `__syncthreads`**
  - CUDA-ban a kilépett szálak "megérkezettnek" számítanak a későbbi barriereknél. Vulkanban a barrier csak egységes vezérlésben definiált.
  - Egy wave-en (AMD: 64 lane = 2 CUDA warp) belül egy szelekció két ágát a driver tetszőleges sorrendben futtathatja.
    A ggml `flash_attn_tile` kernelében a páratlan warpok shared memóriába írnak, a párosak `__syncthreads` után olvassák;
    az ACO a barrieres ágat futtatta előbb.
  - Megoldás: barrieres kernelben ha egy `if`-nek csak az egyik ága szinkronizál (és egyik ág sem ugrik ki magából),
    a másik ág fut előbb, `if (!c) E; if (c) T` alakban; ha az egyik ág mindig visszatér, az kerül külön szelekcióba előre.
    Plusz release-fence a kilépéskor.
  - Eredmény: FLASH_ATTN_EXT 4001/4001 RADV-n; `tools/build_llama.sh --fa`. A tile-kernel pp-ben lassabb (pp512 347 → 209 t/s),
    tg-ben kicsit gyorsabb (31 → 32), ezért az FA build opcionális.

### Fázis 4 – Könyvtárak és első AI workload
- [x] **4.1** `libcublas.so.12` shim: saját `cublas_v2.h`, handle/stream/math mode, `Sgemm`, `GemmEx`, batched és strided batched változatok, `StrsmBatched`.
  A kernelek GLSL-ben vannak (`src/cublas/shaders`), és a `vulkanizerCreateSpirvFunction` driver-bővítésen át futnak.
  Hiányzik: gyors GEMM (cooperative matrix, jobb tiling), `cublasLt`.
- [ ] **4.2** `libcublasLt` (kész: matmul, batch, bias/ReLU/GELU epilógus), `libcurand`, `libcufft`, `libcusparse`, `libcusolver`.
  Majd ha a PyTorch vagy egy célalkalmazás kéri. A cuBLAS is csak a GEMM/TRSM családot tudja (4.1).
  - [x] `libcusolver.so.11` / `.so.12` (CUDA 12 / 13) – a PyTorch `torch.linalg` valós (f32, f64) része, saját GLSL
    kernelekkel (`src/linalg`, közös a cuBLAS batched faktorizációival). Egy munkacsoport oszloponként/párként, a
    batch a grid z-je; a futás a handle streamjén megy, host-szinkron nélkül.
    - LU (`getrf/getrs`, batched cuBLAS-ból is), Cholesky (`potrf/potrs`, batched, 64 bites API), QR (`geqrf`,
      `orgqr`, `ormqr`, `Xgeqrf`, cuBLAS `geqrfBatched`), `trsm` (cuBLAS `S/Dtrsm`, `DtrsmBatched`).
    - Szimmetrikus sajátérték (`syevd`, `syevj`, `syevjBatched`, `Xsyevd`, `XsyevBatched`): ciklikus Jacobi,
      round-robin párosítással (egy kör n/2 független forgatás), rögzített 12 (f32) / 16 (f64) sweep.
    - SVD (`gesvdj`, `gesvdjBatched`, `gesvd`): egyoldali (Hestenes) Jacobi a magas alakon (széles mátrixnál A^T-n);
      a hosszú oldal vektorait (full módban és nulla szinguláris értékeknél is) QR teszi teljes ortonormált bázissá.
    - cuBLAS f64 GEMM (`Dgemm`, `DgemmBatched`, `DgemmStridedBatched`, `GemmEx` f64-gyel) – a linalg tesztekhez
      és a `pinv`-hez kellett.
    - PyTorch: `TORCH_LINALG_PREFER_CUSOLVER=1` (a MAGMA nem fut); a jiterator NVRTC-je cubint kérne, ezt egy
      `LD_AUDIT` könyvtár (`libvulkanizer_nvrtc.so`) PTX-re fordítja át (`-arch sm_XX` → `compute_XX`,
      `nvrtcGetCUBIN*` → `nvrtcGetPTX*`).
    - Tesztek: `tests/cusolver.cpp` (ctest, RADV és llvmpipe), `torch_linalg` (inv, solve, det, slogdet, lu,
      cholesky, qr, householder, ormqr, lstsq, solve_triangular, eigh, eigvalsh, svd, svdvals, pinv, matrix_rank,
      matrix_norm; CUDA 12 és 13 – ALL OK).
    - Nincs: komplex típusok, `gesvdaStridedBatched`, LDL^T (`sytrf`), nem szimmetrikus sajátérték (`Xgeev`);
      ezek `CUSOLVER_STATUS_NOT_SUPPORTED`-et adnak. Nagy mátrixokra (n ≳ 1000) a Jacobi lassú (n²·sweep dispatch).
  - [x] `libcufft.so.11` / `.so.12` (CUDA 12 / 13) – `torch.fft` (fft/ifft, rfft/irfft, fft2, fftn, hfft, `stft`).
    - API: plan 1–3 dimenzióra, batch, advanced layout (embed, stride, dist), C2C/R2C/C2R, f32 / f64 / f16
      (`cufftXtMakePlanMany`; az f16 f32-ben számol), workspace (`SetAutoAllocation`/`SetWorkArea`).
    - Menet: gather a felhasználói tömbből a munkaterület összefüggő komplex tömbjébe (R2C: képzetes rész 0; C2R:
      Hermite-kiegészítés), tengelyenként FFT, scatter. A valós transzformációk teljes hosszú komplexként futnak.
    - Stockham autosort, radix 4, 2, 3, 5, 7, 11, 13 (radixonként külön, kigörgetett butterfly függvény); a gyökök
      táblázata a hoston, f64-ben készül. 2048 (f64: 1024) hosszig az egész FFT egy dispatch shared memóriában, egy
      munkacsoport több vonallal (strided tengelyen szomszédos vonalak, így coalesced az olvasás); hosszabbra radixonként
      egy globális pass. 13-nál nagyobb prímtényezőre Bluestein (chirp-z, kettőhatvány-FFT-kkel; a szűrő FFT-je a hoston).
    - Tanulság: a 64 bites egész osztás (index → koordináták) GPU-n emulált és nagyon lassú – 32 bites indexeléssel
      5–7× gyorsabb lett (a terv < 2^31 elemet enged).
    - Mérés (Renoir iGPU vs. 8 szálas CPU-s pocketfft): fft 64×65536 c64 17,5 vs 35 ms, rfft2 16×512×512 11 vs 16 ms,
      fft 256×1000 0,6 vs 0,75 ms; prím (1009, Bluestein) ~2 ms.
    - Tesztek: `tests/cufft.cpp` (ctest, 30 eset, RADV és llvmpipe), `torch_fft` (CUDA 12 és 13 – ALL OK).
    - Hátra (ha kell): négylépéses FFT a 2048-nál hosszabb vonalakra (két shared memóriás dispatch a sok globális pass
      helyett), R2C/C2R fél hosszú komplex trükkel, callbackek.
  - [x] `libcusparse.so.12` (CUDA 12 és 13 is `.so.12`; a `lib/cuda13`-ban a 13-as verziószámmal) – a `torch.sparse`
    (a `cusparseCreate` korábban inicializálási hibával állt meg).
    - Generikus API: CSR/COO és sűrű leírók (32/64 bites indexek, 0/1 bázis, sor-/oszlopfolytonos, strided batch),
      SpMV, SpMM (op(A) = A^T: a transzponált CSR-jét a munkaterületen építi, atomikus számlálással), SDDMM,
      SpSV/SpSM (egy munkacsoport jobb oldalanként, soronként; mindkét háromszög, transzponált, egységátló),
      SpGEMM; a régi API-ból `csrgeam2` (+`Xcsrgeam2Nnz`), `Xcoo2csr`, `CreateIdentityPermutation`, `Xcoosort`,
      `Xcsrsort`, mátrixleírók.
    - Hoston fut (a streamet szinkronizálja): SpGEMM és a két rendezés – ritkán hívottak, a GPU-s változat (hash
      vagy rendezés) sokkal több munka.
    - COO bemenetnél a sorok rendezettek (a PyTorch coalesced tenzorai ilyenek): a sor-offseteket bináris
      kereséssel számolja.
    - Nincs: BSR (`bsrmm`, `bsrmv`, `bsrsv2`, `bsrsm2`) és komplex típusok – `NOT_SUPPORTED` stubok.
    - Közben javítva: a PTX `griddepcontrol` (sm_90 programmatic dependent launch, a CUB merge sortjában) no-op
      (egy stream kernelei amúgy is egymás után, egymás írásait látva futnak). llvmpipe: a csökkenő sorrendű
      ciklus következő iterációjának betöltését a fordítója előre hozza, a -1. sor címe a bufferen kívül esik –
      az indexet 0-ra kell vágni.
    - Tesztek: `tests/cusparse.cpp` (ctest, 43 eset, RADV és llvmpipe), `torch_sparse` (sparse.mm COO/CSR, addmm,
      mv, backward, dense@CSR, sampled_addmm, CSR+CSR, CSR@CSR, triangular_solve, bmm COO, COO→CSR; CUDA 12 és
      13 – ALL OK). A batched CSR-t a PyTorch CUDA buildje maga tiltja.
  - [ ] `libcurand` – a PyTorch nem használja (saját Philox generátora van); csak ha egy alkalmazás kéri.
- [x] **4.3** Célalkalmazás: a llama.cpp CUDA backendje (`tools/build_llama.sh`).
- [x] **4.4 Mérföldkő:** a SmolLM2-135M (Q8_0 és Q4_0) szöveget generál a Renoir iGPU-n.
  - `llama-bench`: pp64 ≈ 227 t/s, tg32 ≈ 25 t/s.
  - A ggml `test-backend-ops` 12570+ műveletet hasonlít a CPU-hoz, és mind helyes.
  - Közben előjött hibák: zstd-s fatbin, `dp4a`, `bar.red`, a CUB architektúra-lekérdezése, és egy inicializálatlan kilépési változó a strukturált vezérlésben.
- [x] **4.5** PyTorch: a `torch.cuda.is_available()` igazra vált, majd egy egyszerű MLP lefut.
  - A hivatalos wheelekben nincs PTX (csak SASS sm_75…sm_120), ezért a conda-forge CUDA buildje kell
    (`tools/setup_torch.sh`, PTX `compute_120`); futtatás: `./run_torch.sh`.
  - Kellett hozzá: a libcudart teljes PyTorch-felülete (stream prioritás, memory pool, library API,
    driver entry pointok, ...), cuBLAS stubok + `Sgemv`/`SgemmEx`, saját `libcublasLt` (matmul + bias/ReLU/GELU
    epilógus), `__assertfail` → trap, `.ptr .align` paraméterlayout, f16/bf16 (x2) atomikus add,
    lusta PTX-törzs tokenizálás (15 MB modul: 300 → 22 ms).
  - Eredmény: MLP tanítás (Adam, 50 lépés) a CPU-val 6e-8 loss / 5e-7 súly eltéréssel; conv2d, softmax,
    LayerNorm, embedding, attention (math), Transformer réteg, f16 matmul, scatter/index_add helyes
    (`tests/torch/`). cuDNN és a fused attention ki van kapcsolva; `cudaErrorAssert` nincs modellezve.
- [x] **4.6** llama.cpp teljesítmény: flash attention, CUDA graphs (vagy egyenértékű batch-elés), gyorsabb GEMM/MMQ, összevetés a ggml Vulkan backendjével.
  - Kész: vektorizált f16 TN GEMM kernel packed-f16 szorzással `COMPUTE_16F`-hez (213 → ~900 GFLOPS);
    a 64 bites div/rem gyorsútja `DontFlatten`-nel (a driver különben mindig lefuttatta a 64 bites ciklust,
    pl. soft_max 12 → 3,2 ms); `.relaxed/.acquire/.cg/.cv` hozzáférések volatile-ként (CUB look-back GPU hang).
  - TinyLlama-1.1B Q4_0 Renoir-on: pp512 125 → ~320 t/s (natív Vulkan 401), tg ~23 t/s (natív 56).
  - tg: f32-ből emulált fma (a RADV GFX9-en a `precise fma` is mul+add), `dp4a` packed 16 bites
    szorzással (ahol nincs hardveres dot4), a tétlen sorba írt kis HtoD közvetlen memcpy, block packing
    a kis, barrier nélküli kerneleknek. TinyLlama tg64 23 → ~30 t/s.
  - cuBLAS GEMV út (`gemv.comp`) az n = 1 / m = 1 GEMM-ekhez, minden transzponálással, típussal és batch-módban
    (ggml f16 k=128 batched: 24 → 6,7 ms; a natív 2,5 ms, mert a broadcastolt A-t egyszer olvassa).
  - Összevetés (2026-10-22, TinyLlama-1.1B Q4_0, Renoir, `llama-bench -p 512 -n 64`): pp512 ~365–400 t/s
    (natív Vulkan ~347), tg64 ~30 t/s (natív ~51). A pp utolérte a natívat, a tg-ben maradt a különbség.
  - A tg idejének ~73%-a a `mul_mat_vec_q` (Q4_0). Egy 4096×14336-os matvec 1608 µs (20,5 GB/s), a natív 1039 µs.
    Mikrobenchmarkkal (nyers Vulkan, GLSL) kizárva: a BDA (`global_load`) és a descriptoros (`buffer_load`) olvasás
    egyformán ~35 GB/s; egy egyszerű PTX olvasó kernel a Vulkanizeren át is 35 GB/s; a memóriatípus (tiszta
    `DEVICE_LOCAL` vs. host-visible) nem számít; a 16 bites betöltés ~10%. Az mmvq hozzáférési mintája
    kézzel írt GLSL-ben is 20,7 GB/s, vagyis a fordítás nem veszít: a lassúság a CUDA-kernel felépítéséből jön.
    A generikus paraméterekkel egy workgroup egy sort számol, így mind a 4096 workgroup újraolvassa a 16 KB-os
    q8_1 aktivációvektort; enélkül ugyanez a minta 29 GB/s. A natív shader workgroup-onként több sort számol.
    A sorok számát a llama.cpp fordítási időben, architektúra szerint választja (a GCN-táblát csak HIP-buildben),
    ezt a Vulkanizer a forrás módosítása nélkül nem tudja befolyásolni.
  - Kipróbálva és elvetve: kétszeres ciklus-unroll a fordítóban (több betöltés repül egyszerre; mmvq-n 2%,
    tg-ben zajon belül). (A `VULKANIZER_FAST_FMA` itt mért „1%”-a tévesnek bizonyult, lásd 5.7.)
  - Nyitva, de nem a Vulkanizer szűk keresztmetszete: a flash attention tile kernel pp-ben lassú (tensor core
    nélkül), ezért alapból ki van kapcsolva; a `cpy_q_f32` (q4_0 → f32) 3,4× lassabb a natívnál (szálanként egy
    blokk, 128 bájtos lépésközű írások), de inferenciában nem fut.
- [x] **4.7** ComfyUI (Z-Image Turbo) teljes képgenerálás.
  - Már kész: a conda-forge környezetben fut (`venv-vulkanizer`, a torchvision feloldás nélkül, URL-ről).
    Indítás: `run_torch.sh` + `--disable-cuda-malloc` (a 4.9 óta DynamicVRAM-mal is). A `torch._native` Triton/CuTe DSL
    kernelei `TORCH_DISABLE_NATIVE_JIT=1`-gyel ki vannak kapcsolva (`run_torch.sh`).
  - GPU-n kész (2026-10-21): a sablon 1024×1024-en, seed 42-vel végigfut a Renoiron, 117,8 s/lépés (8 lépés, összesen ~21 perc).
    Közben előjött hibák:
    - A 64 parancsos batchek 1024×1024-en 2 s-nál tovább futottak, és az amdgpu reseteli a GPU-t (`lockup_timeout`,
      alapból 2000 ms a compute ringen is). Megoldás: minden dispatch timestampet kap, és pipeline-onként a mért legnagyobb
      ns/workgroup alapján egy submit legfeljebb `VULKANIZER_BATCH_MS` (alap: 50 ms) becsült munkát kap. A nagyobb gridek
      darabokra bomlanak (`vkCmdDispatchBase`, a `%nctaid` push constantból jön). Egy új pipeline első indítása egy 256
      workgroupos, külön mért próbadarab.
    - A pufferben átadott kernelparaméterek (`paramsInBuffer`, >256 bájt, pl. a TensorIterator offset calculatorai)
      a batch arénájában voltak. A több batchre bontott dispatch későbbi darabjai újrahasznosított slot arénájára
      mutattak, amit a következő indítás felülírhatott. Most a `Device::dispatch` minden batchbe saját másolatot tölt fel.
      Regressziós teszt: `VULKANIZER_BATCH_MS=0.01 ./run_torch.sh torch_split`. Ez a hibát a régi kóddal sem
      váltotta ki (ritka slot/offset-egybeesés kell hozzá), de a javítás után a ComfyUI-s futás timeout nélkül ment végig.
    - Diagnosztika: device lostnál a legrégebbi befejezetlen batch dispatch-ei listázódnak méretükkel és becsült
      idejükkel, egyszer. A `CUDA_LAUNCH_BLOCKING=1` minden indítás után megvárja a GPU-t.
  - CPU-s összevetés (2026-10-22, ugyanaz a seed, a CPU fp32-ben, a GPU fp16-ban számol; a CPU-s futás ~10 óra volt):
    kép SSIM 0,929 (8×8 blokk, luma), PSNR 26,6 dB, MAE 6,2/255, a pixelek 4,6%-a tér el 32-nél többel;
    latent koszinusz 0,982, relatív L2 0,19. A kompozíció azonos, az eltérés az éleken és a finom részleteken van.
  - Egyetlen modell-forward 256×256-on (azonos conditioning és zaj, egy euler lépés σ=1-ről), relatív L2 / koszinusz:
    GPU fp16 (Vulkanizer) vs CPU fp32: 4,78e-2 / 0,99886; CPU fp16 vs CPU fp32: 4,66e-2 / 0,99892;
    GPU fp16 vs CPU fp16: 3,85e-3 / 0,999993. A Vulkanizer eltérése az fp32-től akkora, mint maga az fp16-é,
    vagyis a képek közti különbség az fp16 kerekítésből jön, a 8 lépés felerősíti; a maradék a GEMM-ek összegzési sorrendje.
    Megismételhető: `python3 tests/torch/zimage_forward.py` (ComfyUI checkout és Z-Image modellek kellenek hozzá).
  - **Kész, ha:** a Z-Image Turbo sablon 1024×1024-es képet generál, és az eredmény megegyezik a CPU-s futáséval.
  - Az alábbi hiányzó funkciók a ComfyUI nélkülük is futó, de lassabb vagy szűkebb útjait nyitnák meg:
- [x] **4.8** Stream-rendezett allokáció a Driver API-ban: `cuMemAllocAsync`, `cuMemFreeAsync` (`_ptsz` is),
  `cuMemAllocFromPoolAsync`, `cuMemPool*`, `cuDevice{Get,Set,GetDefault}MemPool` (`src/driver/api_mempool.cpp`).
  A felszabadítás a stream addigi munkája után történik várakozás nélkül (`Device::freeBufferAfter`, serial alapján).
  A poolok nem cache-elnek, csak az attribútumokat és a statisztikát tartják. A runtime `cudaMallocAsync`/`cudaMemPool*`
  ezekre épül; `cuGetProcAddress` a per-thread flaggel a `_ptsz`/`_ptds` változatot adja.
- [x] **4.9** Virtuális memóriakezelés (VMM) sparse bufferekkel, a ComfyUI DynamicVRAM-jához (comfy-aimdo).
  - Kész (`src/driver/api_vmm.cpp`, `tests/vmm.cpp`): reserve/free, create/release, map/unmap, set/get access,
    granularitás (2 MB), `cuMemRetainAllocationHandle`, `cuMemHostRegister/Unregister` (csak nyilvántartás,
    `DEVICEMAP` nem megy). RADV-n egy 12 GB-os foglalás 4 db ~4 GB-os sparse bufferből áll.
  - A RADV felülről lefelé osztja a címeket: a többi darab az első alá kerül. A VA-tartomány tetején túlcsorduló
    címet nem szabad kérni, mert a libdrm címkezelője ettől elromlik, és utána minden kérés elbukik.
  - Unmapkor a felvett munkát meg kell várni (`finish`), mapkor nem. A host-oldali másolás sparse memóriába
    staging bufferen át megy (lassú modellbetöltés; UMA-n a VkDeviceMemory közvetlen mappelése gyorsítana).
  - ComfyUI + aimdo kész (2026-10-22): a DynamicVRAM magától bekapcsol („comfy-aimdo inited for GPU: ... RADV RENOIR”,
    integrált GPU-ként 2 GB RAM-tartalékkal). A Z-Image Turbo sablon 1024×1024-en, 8 lépéssel 111 s/lépés, összesen 19:50
    (a 256×256-os, 2 lépéses próba 43 s). A modell két VBAR-ban (4,7 + 11,7 GB) rezidens.
  - API: `cuMemAddressReserve/Free`, `cuMemCreate/Release`, `cuMemMap/Unmap`, `cuMemSetAccess`,
    `cuMemGetAllocationGranularity`; mellette `cuMemHostRegister/Unregister`.
  - Megvalósítás: reserve = `SPARSE_BINDING | SPARSE_RESIDENCY` VkBuffer BDA-val; create = VkDeviceMemory;
    map/unmap = `vkQueueBindSparse` a stream munkájához rendezve. A granularitás a sparse alignment (RADV: 64 KB) és 2 MB közül a nagyobb.
  - Fő akadály: RADV-n a `maxBufferSize` 4 GB, az aimdo viszont 16 GB-os (cast buffer) és modellméretű (Z-Image: 11,5 GB)
    tartományokat foglal. Ezért egy foglalás több, ≤4 GB-os sparse bufferből áll, egybefüggő címtartományon
    (`bufferDeviceAddressCaptureReplay`, `VkBufferOpaqueCaptureAddressCreateInfo`). A `findAllocation` és a
    copy/fill/write/read darabhatárokon vágjon.
  - **Kész, ha:** a ComfyUI `--disable-dynamic-vram` nélkül indul, és az aimdo-val generál.
- [x] **4.10** Triton backend: a `torch._native` műveletei, a `torch.compile`, a comfy-kitchen triton backendje.
  - A Triton NVIDIA backendje a `cuda.h`-val lefordít egy C launchert (ehhez a saját `include/cuda.h` kell).
    A PTX-et `ptxas`-szal cubinná alakítaná, helyette a PTX-et kell betölteni (ptxas-wrapper vagy
    `TRITON_PTXAS_PATH`, ami PTX-et ad vissza; a `cuModuleLoadData` PTX-et is elfogad).
  - Kellhet még: `cuLaunchKernelEx`, `cuFuncSetAttribute`, `cuOccupancyMaxActiveClusters` stb.
  - **Kész, ha:** `TORCH_DISABLE_NATIVE_JIT` nélkül is helyes a `tests/torch/ops.py`, és lefut egy `torch.compile`-olt MLP.
  - Tesztelve (2026-10-22): `tests/torch/triton_ops.py` (kézi `@triton.jit`, és a `torch._native` Triton-os bmm outer productja), `tests/torch/compile.py`
    (`torch.compile`-olt MLP előre és vissza, 5 Inductor Triton kernel, rel. hiba 2e-7), és
    `TORCH_DISABLE_NATIVE_JIT=0`-val a korábbi Python `ops.py` (most `torch_ops`, C++).
    Egy hiba jött elő: a Triton a skalár `ld`/`st` adatát egyelemű vektorként írja (`ld.global.b32 {%r1}, [..]`);
    most elfogadjuk (`tests/data/kernels.ptx`, `vec4`).
  - A `TORCH_DISABLE_NATIVE_JIT=1` alapértelmezés marad: ugyanez a kapcsoló tiltja a CuTe DSL kerneleket, amik SASS-ra
    fordulnak (az `nvidia-cutlass-dsl` most nincs telepítve, így azok amúgy sem futnának).
  - Hátra: a comfy-kitchen triton backendje (a ComfyUI-ban „available, disabled”), ezt a ComfyUI maga kapcsolja.
  - Kész:
    - `tools/triton/ptxas`: a PTX-et adja tovább „cubin”-ként, a verziója (12.8) határozza meg a PTX ISA verziót.
    - A `run_torch.sh` beállítja a `TRITON_PTXAS_PATH`-ot, és a Triton `driver.c`-je a saját `include/cuda.h`-val fordul (`C_INCLUDE_PATH`);
      ehhez a header kapott launch config, tensor map, limit és pointer-attribútum típusokat.
    - Driver: `cuLaunchKernelEx` (`_ptsz`), `cuFuncSetAttribute/SetCacheConfig`, `cuCtxSet/GetLimit`, `cuPointerGetAttribute(s)`,
      `cuOccupancy*`. A `cuTensorMapEncode*` `NOT_SUPPORTED` (TMA, sm_90).
    - A fordító most már ismeri a `.shared::cta`, `.L1::evict_*` módosítókat, és emulálja az `mma.sync`-et
      (m16n8k8/k16, f16/bf16 → f32/f16) és az `ldmatrix`-ot (x1/x2/x4, `.trans`) warp shuffle-lel (`tests/warp_matrix.cpp`).
    - `tests/cuda_abi`: a driver típusai és enumjai az NVIDIA headerhez mérve.
- [x] **4.11** CUDA 13 runtime (`libcudart.so.13`, `libcublas.so.13`): a cu130-as PyTorch buildekhez és a ComfyUI
  „cu130 or higher” optimalizált útjaihoz. Az ABI-eltéréseket (`cudaDeviceProp`, eltávolított API-k) verziónként kell kezelni.
  - Kész (2026-10-22): a runtime és a cuBLAS ugyanabból a forrásból kétszer fordul (`VULKANIZER_CUDART_MAJOR`), a 13-as
    változat a `build/lib/cuda13`-ba kerül (külön könyvtár, hogy a `libcudart.so` linkek ne ütközzenek).
    - Runtime 13: a `cudaDeviceProp` 8 mezője kiesett (`clockRate`, `computeMode`, ...), NUMA/PCI mezők jöttek; a
      `cudaFuncAttributes` két új mezőt kapott. A 12-es `_v2`/`_v3` nevek (`cudaGetDeviceProperties_v2`,
      `cudaStreamGetCaptureInfo_v3`, `cudaGraphAddNode_v2`, ...) a 13-ban a sima nevek; a `cudaMemAdvise` `cudaMemLocation`-t kap.
    - cuBLAS 13: az API ugyanaz, csak a szimbólumverzió más. A cu134-es PyTorch a 64 bites (`_64`) formákat hívja:
      `cublasSgemm_v2_64`, `cublasSgemmStridedBatched_64`, `cublasSgemv_v2_64` valódi (int-be férő méretekkel),
      a többi stub, mint eddig.
    - Driver: a `cuDriverGetVersion` 13040-et ad (a 13-as runtime ennél régebbit nem fogad el). A `cuGetProcAddress`
      a hívó CUDA-verziója szerint adja a változatot (táblázat a `cudaTypedefs.h` PFN-jei alapján); új:
      `cuCtxGetDevice_v2`, `cuCtxSynchronize_v2` (CUDA 13: kontextust kapnak), `cuCtxCreate_v3/_v4`, `cuEventElapsedTime_v2`.
      Hiányzó változatra „not found”-ot ad, nem egy régebbi, más szignatúrájút.
    - Tesztek: `tools/setup_cuda.sh --cuda13` (nvcc 13.4 a `.cuda13`-ba) után az `rt_*_13` nvcc-programok a
      `libcudart.so.13`-on, `cudart_abi_13` és `cuda_abi_13` a CUDA 13 headerekhez mérve. PyTorch 2.14.1 cuda134
      (`tools/setup_torch.sh --cuda13`): `torch13_ops`, `torch13_mlp`, `torch13_split`, `triton_ops.py`, `compile.py` helyes.
  - A ComfyUI „cu130 optimized” útja (comfy-kitchen `cuda` backend) itt nem futhat: statikusan linkelt cudart, PTX csak
    `sm_100f`/`sm_120f`-re, és sm ≥ 8.0 kell neki. cu13-as torch-csal a kitchen magától az `eager` backendre esik vissza
    (kipróbálva), tehát nem törik el semmi.
- Nem Vulkanizer-feladat: a `torchaudio` (nincs a torch 2.14-hez conda-forge build), a ComfyUI-Manager
  „PyTorch is not installed” üzenete emiatt jön.

### Fázis 5 – Széles kompatibilitás és teljesítmény
- [ ] **5.1** cuDNN shim: konvolúció, batchnorm/layernorm, softmax, attention.
  - Halasztva (2026-10-22): az NVIDIA cuDNN 9 nem fut (statikus cudart, `cuGetExportTable`). A PyTorch saját CUDA-kernelei
    cuDNN nélkül működnek, egy shim ugyanezt adná (im2col + GEMM), viszont bekapcsolt cuDNN mellett a PyTorch az RNN-t,
    a CTC-t, a `grid_sample`-t is oda küldené: ami a shimből hiányzik, az a mostani működő állapothoz képest elromlana.
    Ha mégis: `TORCH_CUDNN_V8_API_DISABLED=1` mellett a konvolúcióhoz elég a v7 API.
- [x] **5.2** Unified/managed memory, host-mapped memória (`VK_EXT_external_memory_host`), valódi pinned memória.
  - Kész (`src/driver/api_host.cpp`, `tests/host_memory.cpp`, `rt_kernels managed`): a pinned (`cuMemAllocHost`,
    `cuMemHostAlloc`), a regisztrált (`cuMemHostRegister`) és a managed (`cuMemAllocManaged`) memória host-lap, amit a
    device ugyanazon a pointeren ér el: a `h | 0xFFFF800000000000` device-címen; a kernelek a címek felső bitjeit
    OR-ral állítják be (a sign extension helyett), így a h host-pointer is érvényes device-cím.
    - Pinned és managed: a device saját host-cached memóriája (`vk::Device::allocateHostMemory`), amit
      `VK_EXT_map_memory_placed`-del egy általunk választott, 2 MiB-re igazított h címre mappelünk, a memória
      capture-címe pedig h | bitek (sima buffer; ha nem sikerül, sparse buffer).
    - Regisztrált: az alkalmazás lapjai importálva (`VK_EXT_external_memory_host`, `importHostMemory`), sparse
      bufferhez kötve; az importált memória saját capture-címét a RADV figyelmen kívül hagyja.
    - Sparse bufferhez kötött rendszermemória (és minden userptr) minden submitot lassít (a kernel minden CS-nél
      újraellenőrzi a lapjait, ~4 ms / 256 MB): ezért nem import a pinned memória.
  - A managed memória nem migrál: host-memória marad, amit a device helyben ír-olvas (integrált GPU-n ez ugyanaz a
    memória). `cuMemAdvise`, `cuMemPrefetchAsync`, `cuStreamAttachMemAsync` elfogadja és nem csinál semmit.
    `CONCURRENT_MANAGED_ACCESS` = 0: a host a device munkájának végét várja meg.
  - A szinkron memcpy/memset pinned vagy managed célra megvárja a GPU-t (CUDA-szemantika).
  - Korlát: két regisztráció ugyanazon a lapon – a második host-only marad (a lap device-címe foglalt).
- [x] **5.3** CUDA Graphs: stream capture és visszajátszás (`src/driver/api_graph.cpp`, `tests/graph.cpp`).
  - A gráfok a driverben vannak, így a runtime és a Driver API indításai (Triton) egyaránt rögzülnek. Capture alatt
    a kernelindítás, memcpy/memset, host függvény, event record/wait csomópont vagy függőségi él lesz; másik stream
    eseményre várva csatlakozik (fork/join). Szinkronizálás capture alatt érvényteleníti azt. Egy indított gráf a
    csomópontjait sorban futtatja a stream-en (egy queue: a felvételi sorrend topologikus sorrend), a kernelek előre
    becsomagolt paraméterekkel. `cuGraphExecUpdate` azonos alakú gráfra; explicit kernel/memset/host/üres csomópont.
  - llama.cpp `GGML_CUDA_GRAPHS=ON`-nal épül (`tools/build_llama.sh`), a kimenet azonos; teljesítményben nem számít,
    mert a tg a GPU-n múlik (egy gráf ~470 kernele ~1 ms host-idő, egyenként ~4 ms). PyTorch: `torch_graph`
    (`at::cuda::CUDAGraph`, CUDA 12 és 13), `torch.compile(mode="reduce-overhead")` (`tests/torch/compile.py`).
  - Ehhez kellett: a device-címek 48 bitesek lettek (lásd az ismert korlátoknál), mert a PyTorch CUDA graph tree-je
    int64-ként kezeli a pointert.
  - Hátra: rögzített Vulkan command bufferek (csak host-időt spórolna), child graph, memória-csomópontok
    (`cuMemAllocAsync` capture alatt), `cudaGraphAddNode` általános paraméterekkel, feltételes csomópontok.
- [x] **5.4 Teljesítmény**
  - [x] DtoH: a write-combined device-memóriából a CPU ~0,2 GB/s-mal olvas. A `Device::read` a GPU-val másol egy
    host-cached staging bufferbe (2 × 8 MiB, a GPU és a CPU másolása átfedve), 4 KiB alatt közvetlenül olvas.
    A pinned/managed memória és a device közti másolás GPU-n fut, az `*Async` változatok nem várnak.
    Mérés (`vulkanizer-bandwidth`, 256 MiB): DtoH pageable 0,17 → ~3,8 GB/s, DtoH pinned 0,25 → ~12 GB/s.
  - [x] profilozó mód (`VULKANIZER_PROFILE`).
  - [x] sub-allokátor (`Device::allocateBuffer`): a ≤ 4 MiB foglalások közös, 64 MiB-os blokkokból jönnek (first fit,
    256 B igazítás, összevonás felszabadításkor; egy üres blokk megmarad, a többi a rajta futó munka után felszabadul).
    A felszabadítás nem vár a GPU-ra: minden későbbi device-munka a már rögzített után fut, a host-hozzáférés pedig
    megvárja azt. `cuMemAlloc`+`cuMemFree`: 4 KiB 42 → 0,9 µs, 1 MiB 31 → 3,4 µs. Kikapcsolás: `VULKANIZER_NO_SUBALLOC=1`.
    A nagyobb foglalások külön Vulkan-memóriát kapnak (16 MiB ~0,8 ms, a kernel nullázza a lapokat).
  - Hátra (halasztva): külön transfer queue a másolásoknak. A másolások már most sem blokkolnak (a compute queue-n,
    sorban futnak); a külön queue a kernelekkel való átfedést hozná, integrált GPU-n (közös memória-sávszélesség)
    kevés haszonnal.
- [x] **5.7** llama.cpp tokengenerálás (tg) a natív Vulkan backendhez közel – lezárva (2026-10-23) ~31 → ~38 t/s-nál;
  a maradék különbség a llama.cpp mmvq kernelének felépítéséből jön (lásd lent), általános fordítóoptimalizálással
  nem csökkenthető tovább.
  - Kiinduló helyzet (2026-10-23, TinyLlama Q4_0, Renoir, felváltva mérve): Vulkanizer tg64 ~31 t/s, natív ~60–65.
    A gép teljesítményállapota (memóriaórajel) futásról futásra 20%-ot is mozdít: csak felváltott A/B mérés számít.
  - [x] `.ftz` fma natívan: a ggml-cuda `--use_fast_math`-tal fordul, a forró `fma.rn.ftz.f32` utasításokra nem kell
    a ~30 utasításos egyszeri-kerekítés emuláció (az a RADV GFX9 unfused `Fma`-ja miatt kell; a `DenormPreserve`
    sem ad fused fma-t, a driver szétbontja). A nem-ftz fma marad emulált (a matematikai könyvtárak hibamentes
    transzformációi); f16 fma mindig natív (két f16 szorzata f32-ben pontos). `VULKANIZER_EXACT_FMA=1` a régi
    viselkedés. tg64 31 → 37,7 t/s (+21%), a szöveg azonos.
  - Elemzés (`VULKANIZER_PIPELINE_STATS=1`: VGPR, SGPR, spill, wave/SIMD pipeline-onként): a Q4_0 mmvq nem
    sávszélesség-, hanem utasításkiadás-korlátos. K = 2048-nál egy 128 szálas workgroup egy sort számol egyetlen
    iterációban (~500 utasítás/szál); tokenenként ~264 ezer sor ≈ 280 millió wave-utasítás ≈ 20 ms a Vega 8-on.
    Occupancy rendben (6–7 wave/SIMD, nincs spill). A kód nagyobb tételei: 64 bites címszámítás (~55 utasítás,
    `v_add_co/addc` párok, minden betöltéshez külön), dp4a emuláció (~12 utasítás/dp4a, nincs hardveres dot4).
  - [x] Tokenek közti üresjárat: a `VULKANIZER_PROFILE` most a batch-ek közti GPU-tétlenséget is méri (rövid =
    submit-késleltetés, hosszú = a host). Tokenenként ~2,6 ms tétlenség volt: a ggml-cuda a bemeneteket (tokenek,
    pozíciók, maszk) pinned host-bufferből tölti fel, ~20 `cudaMemcpyAsync` + `cudaStreamSynchronize` párral, és a
    pinned forrás GPU-másolást jelentett, így mindegyik egy GPU-oda-vissza út volt (~120 µs). Most a pinned/managed →
    device másolás üres GPU mellett host-memcpy (mint pageable forrásnál): a tétlenség ~0,6 ms, tg +2–3%.
  - Kipróbálva, nem segít (felváltott A/B): `shfl.bfly` → `OpGroupNonUniformShuffleXor` (az ACO már swizzle-t ad);
    nagyobb batch (256/1024 parancs; zajon belül); korai flush üres GPU-nál (8 parancs után; zajon belül); a dp4a
    mezőkivonásos változata (−15%, az ACO nem fűzi SDWA-ba; a csomagolt 16 bites marad).
  - A 64 bites címszámítás szűkítése (vizsgálva, 2026-10-23): a kernelben nincs negyed sebességű vektoros szorzás (a
    konstans szorzásokat a driver eltolásra bontja vagy skalárban végzi); a ~300 VALU-utasításból ~60 a 64 bites
    cím (`v_add_co/addc`, `v_lshlrev_b64`, előjel-kiterjesztés). Ezt maga a PTX írja elő (`mul.wide.s32 sor, 18`,
    `mul.lo.s64 index, 36`, paraméterekből jövő indexekkel), így a gyors címzés (skalár bázis + 32 bites offset)
    csak akkor volna helyes, ha az offset bizonyítottan elfér 32 biten; ez paraméterfüggő értékekre nem bizonyítható.
    A konstans offsetek (`+4`, `-2`) beolvasztását a felső bitek OR-ja akadályozza, amit a forrásra (paraméter-
    betöltésre) áttenni nem biztonságos: a null-ellenőrzés (`setp.eq.s64 p, ptr, 0`) és a memóriába írt pointerek
    értéke megváltozna. Elvetve: biztonságosan kevés nyerhető (felső becslés ~10% a kernelen).
  - Nem készült el (llama-specifikus, ezért elvetve): kernelcsere a llama.cpp mmvq-jére (soronként több sor, újrahasznált aktiváció) – llama-specifikus;
    a KV-cache másolás (`cpy_scalar` f16, 5004 utasítás, ~1 ms/token: a 64 bites div/rem gyors útja mellett a
    lassú út is beinline-olódik).
- [ ] **5.5** Csomagolás: install target, `.deb`, dokumentáció. Jogi átnézés publikálás előtt.
  - [x] Install és csomagok (`cmake/install.cmake`, `tools/package.sh [--check]`): CPack `.deb` (függőségek
    `dpkg-shlibdeps`-ből: libvulkan1, libstdc++; ajánlott: mesa-vulkan-drivers) és `.tar.gz`. A könyvtárak a
    `<prefix>/lib/<arch>/vulkanizer/` (+ `cuda13/`) alá kerülnek, a linker keresési útvonalán kívül (nincs ldconfig),
    hogy egy ugyanott lévő NVIDIA drivert ne takarjanak el; `$ORIGIN` RUNPATH-szal megtalálják egymást. A fejlécek saját
    gyökérben (`<prefix>/include/vulkanizer/`, a CUDA toolkit `cuda.h`-jával ne ütközzenek), a PyTorch/Triton
    segédfájlok a `<prefix>/share/vulkanizer/` alatt.
  - [x] `vulkanizer-run` indító (a `run.sh`/`run_torch.sh` telepített párja): `LD_LIBRARY_PATH` a CUDA 12 és 13
    könyvtárakra; `--torch` a Python környezet NVIDIA-könyvtárai elé tölti a Vulkanizert (a CUDA 12/13-at a környezet
    `libcudart.so.13`-ja alapján ismeri fel) és beállítja a PyTorch/Triton változókat. Kipróbálva a kicsomagolt
    `.deb`-ből: nvcc-programok (CUDA 12 és 13), llama.cpp, PyTorch CUDA 12 és 13, Triton.
  - [x] Dokumentáció: README „Telepítés (csomag)” szakasz mind a 8 nyelven.
  - [ ] Jogi átnézés publikálás előtt (nem fejlesztési feladat).
- [x] **5.6** Cooperative groups (grid-szintű szinkronizálás), multi-GPU, NCCL, IPC.
  - [x] Cooperative groups (`grid.sync()`, `rt_kernels cooperative`, CUDA 12 és 13): `cuLaunchCooperativeKernel`
    (`_ptsz`), `cuLaunchKernelEx`/`cudaLaunchKernelEx` cooperative attribútuma, `cudaLaunchCooperativeKernel`,
    graph-csomópontként is. A toolkit fejléce a grid workspace címét a `%envreg1`/`%envreg2` regiszterekből olvassa: a
    fordító ezeket egy push-constant mezőből adja (`KernelBinary::workspaceOffset`), a driver kooperatív indításnál
    egy eszközönkénti workspace címét írja bele, máskor 0-t (`is_valid()` hamis, mint a CUDA-ban).
    - Minden blokknak egyszerre rezidensnek kell lennie: a grid legfeljebb occupancy × SM-szám, különben
      `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`; a dispatch nem darabolódik (`Device::dispatch(..., whole)`) és nem
      csomagolódik. Az SM-szám AMD-n a valós CU-szám (`VK_AMD_shader_core_properties2`), a runtime occupancy-függvényei
      a driverét hívják (`MAX_BLOCKS_PER_MULTIPROCESSOR` = 32, llvmpipe-on 1: szálanként egy workgroup fut).
    - A PTX memóriarendezése: az `ld/st/atom/red` `.acquire`/`.release`/`.acq_rel` minősítője az utasítás köré tett,
      hatókörének megfelelő fence (eddig elveszett: az `atom.add.release.gpu` relaxed volt). A grid workspace-t használó
      kernelekben a `bar.sync` device-hatókörű, és a globális elérések Volatile-ok: a blokk többi wave-jének store-jai
      így érnek a GPU szintjére, és a más blokkok által írt adatot nem a driver korábbi loadjaiból olvassa vissza.
    - Korlát: a rezidencia becslés; egy sok regisztert használó kernelből kevesebb fér el egy CU-n, a túl nagy grid
      ilyenkor holtpontra jut (GPU-n timeout → device lost, llvmpipe-on végtelen várakozás).
  - [x] Több GPU (`rt_kernels multigpu`, `VULKANIZER_ALLOW_CPU=1`-gyel RADV + llvmpipe): `cuMemcpyPeer(Async)`,
    `cudaMemcpyPeer(Async)` eszközök között host-memórián át; másik eszköz eventjére váró stream (`cuStreamWaitEvent`) a
    hoston vár; `cuDeviceCanAccessPeer` = 0, `cuCtxEnablePeerAccess` → `PEER_ACCESS_UNSUPPORTED` (külön Vulkan
    device-ok nem osztoznak memórián); valós PCI-cím (`VK_EXT_pci_bus_info`), különben az eszközindex.
    - Közben javítva: az llvmpipe-on minden nvcc-teszt segfaultolt az 5.2 óta (a kernelek a 0xFFFF8000 felső biteket
      OR-olták az llvmpipe CPU-pointereire). A felső bitek most eszközönként a tényleges BDA-ból jönnek
      (`Device::hostAddressBits()`), és minden pipeline az 5-ös specializációs konstansként kapja meg. Emulált warpokon
      (subgroup < 32) a predikált `redux`/`shfl` stb. minden szálon fut (a barrierek nem lehetnek divergens ágban), a
      predikátum csak a célregisztert választja.
    - [x] Eszközök közti UVA: a sima másolások (`cuMemcpy*`, `cudaMemcpy(Async)` `Default`/`DeviceToDevice`) és a
      pointer-attribútumok a current context eszköze után a többi eszköz foglalásai közt is keresnek
      (`findPiecesOnAnyDevice`); két eszköz között a hoston át másol, mint a `cuMemcpyPeer`. A PyTorch
      `tensor.to("cuda:1")` és `copy_` így működik (eddig `invalid argument`). Korlát: két külön Vulkan device
      címtartománya átfedhet (két egyforma GPU); ilyenkor a current context eszköze nyer.
    - llama.cpp két eszközön (RADV + llvmpipe) eljut a rétegek szétosztásáig és a peer másolásig, de az llvmpipe-on a
      `rope_norm` kernel (blockDim.x = 1) összeomlik: az llvmpipe a `set_rows_stride != 0` feltétel mögötti
      `row_indices[i]` loadot a feltétel elé emeli (Volatile-lal is), és a null pointerből olvas. A Vulkanizer a loadot a
      feltételen belül adja (Mesa-oldali viselkedés); RADV-n ugyanez nem fordul elő.
  - [x] IPC (`cuIpc*`/`cudaIpc*`, a PyTorch `torch.multiprocessing` CUDA-tenzor megosztása; CUDA 12 és 13):
    - Minden eszközfoglalás exportálható Vulkan-memória (`VK_KHR_external_memory_fd`, opaque fd; a sub-allocation
      blokkok is). A 64 bájtos handle adat: az exportáló pid-je, a memória azonosítója, a pozíció benne, a buffer és a
      memória létrehozási paraméterei és az eszköz UUID-ja. Fd-t nem visz: az exportáló folyamat az első
      `cuIpcGetMemHandle`-nél egy absztrakt Unix socketet indít (`vulkanizer-ipc-<uid>-<pid>`, `SO_PEERCRED`: csak
      azonos usernek), amin SCM_RIGHTS-szal ad fd-t; ez a Yama ptrace-korláttal is működik (a `pidfd_getfd`-vel
      szemben, ami gyerek→szülő irányban nem). Az importáló egyszer importálja az egész memóriát, a további
      megnyitások (ugyanabba a memóriába) refcounttal osztoznak rajta.
    - Interprocess event: a rögzítése megvárja az eszközt (`cuEventRecord`), így a másik folyamatban megnyitott event
      mindig kész – a PyTorch a megosztásnál így használja (record, utána handle).
    - Az exportálható memória nem lassít (llama tg/pp felváltott A/B-vel a zajon belül); `VULKANIZER_IPC=0`
      kikapcsolja. Közben: az export jelzőjét az eszköz létrehozása után rögtön be kell állítani – az init-próbák
      már sub-allocation blokkot hoznak létre, és a későbbi foglalások abból kapnak.
    - Tesztek: `tests/ipc.cpp` (fork, két folyamat: kis és nagy foglalás, belső pointer, kétszeres megnyitás, GPU-írás
      mindkét irányban, event; RADV és llvmpipe); PyTorch `spawn`: tenzor a gyereknek, helyben módosítva, és vissza.
  - [x] NCCL (`libnccl.so.2`, egy könyvtár CUDA 12-höz és 13-hoz: csak a driver API-t használja) – a PyTorch
    `torch.distributed` "nccl" backendje (DDP, FSDP), egy gépen, akár egy GPU-n több ranggal (az NVIDIA NCCL-je ezt
    nem engedi). Az NVIDIA libnccl-je itt nem futhat: statikusan linkeli a libcudart-ot, ami a driver nem
    dokumentált `cuGetExportTable` interfészeit kéri.
    - Egy rang: minden kollektíva egy eszközmásolás a streamen (premultiplikált összeg kivételével).
    - Több rang: minden művelet post + complete. A post megvárja a streamet, és a rang hozzájárulását egy saját
      POSIX shm objektumba teszi (neve: a unique id tokenje, a művelet sorszáma és a rang), nem blokkol; a complete
      megvárja a kellő objektumokat, rang-sorrendben redukál (minden rang ugyanazokat a biteket kapja), és a
      device-ra másolja. Az objektumot az utolsó olvasója törli. A group (`ncclGroupStart/End`) előbb mindent
      postol, aztán completel: a send/recv párok és egy folyamat több kommunikátora (`ncclCommInitAll`) sem
      várnak egymásra. A pont-pont üzenetek rang-páronként sorszámozottak.
    - Kollektívák: AllReduce, Reduce, Broadcast/Bcast, AllGather, ReduceScatter, AlltoAll, Gather, Send/Recv;
      sum/prod/max/min/avg és `ncclRedOpCreatePreMulSum` (host vagy device skalár); f16/bf16/f32/f64 és az egész
      típusok (fp8-ra csak az adatmozgatók). `ncclCommSplit` (a (szín, kulcs) párok allgatherrel), `CommInitAll`,
      `InitRankScalable`, `MemAlloc`, `CommRegister` (no-op). Nincs: szimmetrikus ablakok, device-kommunikátorok,
      one-sided signal, shrink/grow – `ncclInvalidUsage` stubok.
    - Az adat a hoston megy át (lassabb, mint egy GPU-s transzport; a DDP/FSDP egy gépen így is működik).
    - Tesztek: `tests/nccl.cpp` (3 forkolt folyamat minden kollektívával és splittel, egy rang, egy folyamat 3
      kommunikátora; RADV és llvmpipe); PyTorch: egyrangú DDP, `torchrun --nproc_per_node 2` DDP (a replikák
      azonosak maradnak, `all_gather_object`, `barrier`) és FSDP2 (`fully_shard`; az egyfolyamatos referenciától
      7e-9) CUDA 12-n és 13-on.

## 3. Ismert korlátok / kockázatok

- **Statikusan linkelt libcudart** (az nvcc alapértelmezése) nem fut: az NVIDIA runtime a driver nem dokumentált belső tábláit igényli. Megoldás: `-cudart shared` újrafordítás, vagy később a táblák pótlása.
- **Csak-SASS binárisok** (PTX nélkül) nem futtathatók – nincs reális SASS → SPIR-V út.
- **GPU-timeout**: a kernel driver reseteli a GPU-t, ha egy submit túl sokáig fut (amdgpu: 2 s). A batchek mért
  kernelidők alapján legfeljebb `VULKANIZER_BATCH_MS`-nyi munkát kapnak. Ha egyetlen workgroup is tovább fut a
  limitnél, vagy a becslés nagyon alulmér (ugyanaz a pipeline sokkal nagyobb K-val), a reset továbbra is előfordulhat.
- **Vulkan shader modell**: nincs rekurzió, nincs függvénypointer, structured CF kötelező, kernelből `malloc`/`printf` nincs (printf később emulálható bufferrel).
- **Warp size ≠ 32** eszközökön a warp-szintű kód lassú vagy részlegesen támogatott.
- **Device-címek**: a RADV a buffereket a 48 bites címtér felső felébe teszi (0xffff8...), ami int64-ként negatív
  (a PyTorch CUDA graph tree-je, `ctypes` ettől elbukik). A Vulkanizer ezért a bitek 63..48 nélkül adja ki a címeket
  (`vk::kAddressMask`, 0x8000... alakúak, pozitívak), a kernelek pedig minden elérés előtt visszaállítják a felső
  biteket (a PTX-fordító a báziscímen, egyetlen 32 bites OR-ral a device felső bitjeivel – RADV-n 0xFFFF8000,
  llvmpipe-on 0, az 5-ös specializációs konstans; a GLSL shaderek `canon()`-nal), így a
  host-memória h címe is érvényes device-cím (lásd 5.2). Mérhető ára nincs.
- **UVA**: több Vulkan device címtere ütközhet; a pointer-keresés a current context device-án belül történik
  (eszközök között `cudaMemcpyPeer` működik, a `cudaMemcpyDefault` két eszköz pointerével nem).
- **Regisztrált host-memória** (`cuMemHostRegister`): amíg él, minden GPU-submit lassabb (a kernel minden submitnál
  újraellenőrzi az importált lapokat, ~4 ms / 256 MB). A pinned és a managed memóriát ez nem érinti.
- **libcudart privát export table-jei** (`cuGetExportTable`) nem dokumentáltak, CUDA verziónként változhatnak → verzió-specifikus kezelés, sok trace-elés.
- **Jogi**: NVIDIA EULA (CUDA ≥ 11.6) tiltja az SDK kimenetének visszafejtését/átfordítását nem-NVIDIA platformra. Clean-room API implementáció; NVIDIA headert/binárist nem teszünk a repóba. Publikálás előtt jogi átnézés.

## 4. Konfiguráció (környezeti változók)

| Változó | Jelentés |
|---|---|
| `VULKANIZER_LOG` | `off`/`error`/`warn` (default)/`info`/`debug`/`trace` – `trace` minden API hívást logol |
| `VULKANIZER_DEVICE` | Device név részlet (case-insensitive), pl. `radv`, `llvmpipe` |
| `VULKANIZER_ALLOW_CPU` | `1` = CPU Vulkan device-ok (llvmpipe) is látszanak (több GPU kipróbálásához is) |
| `VULKANIZER_SM_COUNT` | Jelentett multiprocessor-szám (alap: AMD-n a valós CU-szám, máshol 8; CPU-n a szálak száma) |
| `VULKANIZER_VALIDATION` | `1` = Vulkan validation layer bekapcsolása |
| `VULKANIZER_CC` | Jelentett compute capability, pl. `75`, `86` |
| `VULKANIZER_SPIRV_VAL` | `1` = minden generált SPIR-V `spirv-val`-on megy át (hibás modul → a kernel nem indítható) |
| `VULKANIZER_DUMP` | Könyvtár, ahova a generált SPIR-V (`<kernel>.spv`) és a modulok PTX-e (`<hash>.ptx`) kiíródik |
| `VULKANIZER_CACHE_DIR` | Kernel cache helye (alap: `$XDG_CACHE_HOME/vulkanizer`, `~/.cache/vulkanizer`); `off` kikapcsolja |
| `VULKANIZER_CACHE_SIZE` | A kernel cache méretkorlátja MiB-ban (alap: 256). Fölötte a legrégebben használt bejegyzések törlődnek a korlát 3/4-éig (találatkor a fájl mtime-ja frissül; processzenként egyszer, az első új kernel tárolásakor) |
| `VULKANIZER_EAGER` | `1` = a nagy modulok kerneleit is betöltéskor fordítja (alapból csak a ≤16 kerneleseket) |
| `VULKANIZER_PROFILE` | `1` = kilépéskor kiírja a kernelenkénti GPU-időt és az API-hívásonkénti host-időt |
| `VULKANIZER_BATCH_MS` | Egy submit becsült GPU-ideje legfeljebb ennyi ms (alap: 50); a hosszabb dispatch több submitra bomlik. Az amdgpu 2 s után reseteli a GPU-t. `0` = nincs korlát |
| `CUDA_LAUNCH_BLOCKING` | `1` = minden kernelindítás megvárja a GPU-t (hiba/lefagyás az okozó indításnál jelentkezik) |
| `VULKANIZER_FAST_FMA` | `1` = az f32 `fma` a driver natív (esetleg nem fused) `Fma`-ja, ott is, ahol a próba szerint nem fused (gyorsabb, kevésbé pontos) |
| `VULKANIZER_EXACT_FMA` | `1` = a `.ftz` (fast math) f32 `fma` is emulált egyszeri kerekítéssel (alapból ott a driver `Fma`-ja fut) |
| `VULKANIZER_PIPELINE_STATS` | `1` = minden pipeline létrehozásakor kiírja a driver statisztikáit (VGPR, SGPR, spill, wave/SIMD, utasítások) |
| `VULKANIZER_FMA_F64` | `1` = a nem fused eszközön az f32 `fma` f64-ben számolódik (alapból f32 TwoProduct/TwoSum emuláció) |
| `VULKANIZER_NO_VMM` | `1` = nincs sparse binding, így nincs CUDA VMM (`cuMemAddressReserve` stb. `NOT_SUPPORTED`) |
| `VULKANIZER_NO_SUBALLOC` | `1` = minden `cuMemAlloc` külön Vulkan-memóriát kap (nincs sub-allokátor) |
| `VULKANIZER_NO_PACK` | `1` = nincs block packing (minden CUDA block külön workgroup) |
| `VULKANIZER_GEMM_F32` | `1` = a cuBLAS `COMPUTE_16F` GEMM-ek is f32 szorzással futnak (alapból packed f16, ha az eszköz tud f16 aritmetikát) |

## 5. Referenciák

- ZLUDA (CUDA → ROCm, Driver API szintű csere) – architektúra minta
- chipStar (CUDA/HIP → OpenCL/Level Zero SPIR-V)
- PTX ISA dokumentáció, CUDA Driver API dokumentáció
- SPIR-V spec + Vulkan environment, SPIRV-Tools (`spirv-val`, `spirv-opt`, `spirv-dis`)
- VkFFT, llama.cpp/ggml Vulkan backend (GEMM kernel ötletek)
