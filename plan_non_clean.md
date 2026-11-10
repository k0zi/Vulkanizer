# Vulkanizer – nem clean-room feladatok (eltéve)

> Ide kerülnek azok a feladatok, amelyek nem oldhatók meg clean-room módon, vagyis csak a nyilvános
> dokumentációból, a nyilvános fejlécekből és a programok kívülről megfigyelhető viselkedéséből. Megoldani csak
> NVIDIA-binárisok visszafejtésével, belőlük kinyert adatokkal, vagy NVIDIA-védelmek megkerülésével lehetne őket
> (döntés: 2026-10-23). Nem dolgozunk rajtuk. Ha valaha mégis előkerülnek, előbb jogi átnézés kell
> (lásd `plan.md`, 3. fejezet, „Jogi”).
> A clean-room munka a [plan.md](plan.md)-ben folytatódik.

## 1. Statikusan linkelt libcudart (`cuGetExportTable`) – a régi `plan.md` 6.5

- **Mit oldana meg:** az nvcc alapértelmezése (`-cudart static`), a CuPy, az NVIDIA NCCL, cuDNN és libcurand,
  valamint a comfy-kitchen cu130-as útja is futna. Ezek a programok a CUDA runtime-ot statikusan linkelik, és a
  runtime a drivertől nem dokumentált belső táblákat kér.
- **Miért nem clean-room:** a táblák többsége kívülről megfigyelhető. Az utolsó viszont egy integritás-ellenőrzés
  (`d4082055…` slot 1: runtime-verzió, Unix-idő, 16 bájtos eredmény), amelyet a runtime ellenőriz, és nélküle nem
  indul. Az algoritmus kívülről nem figyelhető meg, átvétele pedig egy NVIDIA-védelem megkerülése lenne.
- **Ami megmaradt:** `src/driver/export_tables.cpp`. A CMake `VULKANIZER_EXPORT_TABLES` option alapból `OFF`, így
  nem fordul bele a libcuda-ba. Ha be van kapcsolva, futáskor `VULKANIZER_EXPORT_TABLES=1|probe` választja a
  módot.
  - A `probe` mód minden kért interfészre logoló táblát ad: slotonként a regiszter-argumentumokat és a mutatott
    memória első 32 bájtját.
  - `VULKANIZER_EXPORT_PROBE_OK=<uuid>:<slot>`: az adott slot sikert ad vissza.
  - `VULKANIZER_EXPORT_PROBE_DENY=<uuid>`: az adott interfészt elutasítja.
- **Megfigyelt sorrend** (nvcc 12.9, `-cudart static`, `rt_vector_add`):
  1. `f8cff951…`: táblaként elfogadja.
  2. `cuDriverGetVersion`, `cuInit`.
  3. `6bd5fb6c…` slot 2 (ctx out, device): `cuDevicePrimaryCtxRetain`-nel elfogadja.
  4. A teljes Driver API `cuGetProcAddress`-szel.
  5. `a094798c…` slot 2 és 6 (terület- és méret-kimenet): egy-egy nullázott, 4 KiB-os puffert elfogad.
  6. `42d85a81…`, `c693336e…`, `263e8860…` slot 3 (`int*` kimenet).
  7. `d4082055…` slot 1: az integritás-ellenőrzés. Elutasítva a runtime „operation not supported”-del leáll.
- **Clean-room helyettesítő (ez marad a `plan.md`-ben):** `-cudart shared` újrafordítás; a `vulkanizer-check`
  felismeri a statikus runtime-ot, és ezt javasolja.
- **Régi kockázati megjegyzés:** a privát export table-ök nem dokumentáltak, és CUDA-verziónként változhatnak:
  verzió-specifikus kezelés, sok trace-elés kellene hozzájuk.

## 2. Csak SASS-t tartalmazó binárisok futtatása

- **Mit oldana meg:** a hivatalos PyTorch wheel-eket és minden PTX nélkül fordított programot (ma a conda-forge
  PTX-es buildje kell).
- **Miért nem clean-room:** a SASS (NVIDIA natív gépi kód) ISA-ja nem dokumentált. Egy SASS → SPIR-V fordítóhoz
  generációnként vissza kellene fejteni az utasításkódolást, ráadásul NVIDIA-binárisok (cubinok) elemzésével.
- **Clean-room helyettesítő:** a `vulkanizer-check` felismeri a csak SASS-t tartalmazó binárist; a hibaüzenet
  felsorolja az architektúrákat, és virtuális architektúrával (`-arch=compute_XX`) való újrafordítást javasol.

## 3. cuRAND: NVIDIA-val bitre azonos sorozatok, ahol az algoritmus nem dokumentált

- **Mit oldana meg:** a scrambled Sobol és a Poisson eloszlás ugyanazt a sorozatot adná, mint az NVIDIA
  libcurand-ja. Ma ezek más, de helyes eloszlású sorozatot adnak.
- **Miért nem clean-room:**
  - A scrambled Sobol NVIDIA-féle scramble-konstansai csak az NVIDIA könyvtárából nyerhetők ki
    (`curandGetScrambleConstants*`).
  - A Poisson-generátor algoritmusa nem dokumentált: csak a kimenetéből lehetne visszafejteni.
  - Ugyanez vonatkozik az MT19937 és az MTGP32 szálankénti elrendezésének minden olyan részére, amelyet a
    dokumentáció nem ír le.
- **Clean-room rész (marad a `plan.md` 6.11-ben):**
  - MT19937 és MTGP32 a nyilvános algoritmus és paramétertáblák alapján, és a dokumentált elrendezéssel.
  - A hivatalos kimenettel csak ellenőrzünk, abból nem vezetünk le semmit.
  - `curand_discrete`: a host készíti el a struktúrát a nyilvános header elrendezése szerint.
