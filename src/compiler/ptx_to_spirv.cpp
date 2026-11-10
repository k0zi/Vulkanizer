#include "compiler/ptx_to_spirv.h"

#include "compiler/cfg.h"
#include "compiler/spirv_builder.h"
#include "common/device_address.h"
#include "common/log.h"
#include "ptx/instr.h"

#include <spirv/unified1/GLSL.std.450.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace vulkanizer::compiler {

namespace {

enum class Base { Pred, Bit, Unsigned, Signed, Float };

struct PtxType {
    Base base = Base::Bit;
    uint32_t width = 32;  // bits; 1 for .pred
    bool isInt() const { return base == Base::Bit || base == Base::Unsigned || base == Base::Signed; }
    bool isFloat() const { return base == Base::Float; }
    bool isSigned() const { return base == Base::Signed; }
};

bool parseType(std::string_view m, PtxType& t) {
    if (m == ".pred") return t = {Base::Pred, 1}, true;
    if (m.size() < 3 || m[0] != '.') return false;
    Base b;
    switch (m[1]) {
    case 'b': b = Base::Bit; break;
    case 'u': b = Base::Unsigned; break;
    case 's': b = Base::Signed; break;
    case 'f': b = Base::Float; break;
    default: return false;
    }
    const std::string_view w = m.substr(2);
    uint32_t width;
    if (w == "8") width = 8;
    else if (w == "16") width = 16;
    else if (w == "32") width = 32;
    else if (w == "64") width = 64;
    else return false;
    if (b == Base::Float && width == 8) return false;
    t = {b, width};
    return true;
}

// Builds the error text for an instruction.
std::string at(const ptx::Instruction& ins, const std::string& msg) {
    return "line " + std::to_string(ins.line) + ": " + msg + " in '" + ptx::toString(ins) + "'";
}

struct CompileError {
    std::string message;
};

class Translator {
public:
    Translator(const ptx::Module& m, const ptx::Entry& e, const Target& t) : m_(m), e_(e), t_(t) {}

    bool run(KernelBinary& out, std::string& error) {
        ptx::ParseError perr;
        if (!ptx::parseFunctionBody(m_, e_, fn_, perr)) {
            error = "line " + std::to_string(perr.line) + ": " + perr.message;
            return false;
        }
        if (!buildCfg(fn_, cfg_, error)) return false;
        if (!structurize(cfg_, st_, error)) return false;
        try {
            emitModule();
        } catch (const CompileError& ce) {
            error = ce.message;
            return false;
        }
        out.spirv = b_.finish();
        out.entryPoint = e_.name;
        out.pushConstantSize = pushBytes_;
        out.globalsOffset = usesGlobals_ ? globalsOffset_ : KernelBinary::kNone;
        out.gridOffset = usesGrid_ ? gridOffset_ : KernelBinary::kNone;
        out.paramsInBuffer = paramsInBuffer_;
        out.printfOffset = usesPrintf_ ? printfOffset_ : KernelBinary::kNone;
        out.assertOffset = usesAssert_ ? assertOffset_ : KernelBinary::kNone;
        out.workspaceOffset = usesWorkspace_ ? workspaceOffset_ : KernelBinary::kNone;
        out.staticSharedBytes = sharedStatic_;
        out.localBytes = localBytes_;
        out.packable = packable_;
        return true;
    }

private:
    [[noreturn]] void fail(const ptx::Instruction& ins, const std::string& msg) { throw CompileError{at(ins, msg)}; }
    [[noreturn]] void fail(const std::string& msg) { throw CompileError{msg}; }

    // ---- types -------------------------------------------------------------------------------

    uint32_t tBool() { return b_.typeBool(); }
    uint32_t tU32() { return b_.typeInt(32); }
    uint32_t tU64() { return b_.typeInt(64); }
    uint32_t tInt(uint32_t w) {
        if (w == 8) b_.capability(spv::CapabilityInt8);
        if (w == 16) b_.capability(spv::CapabilityInt16);
        return b_.typeInt(w);
    }
    uint32_t tFloat(uint32_t w) {
        if (w == 64) b_.capability(spv::CapabilityFloat64);
        return b_.typeFloat(w);
    }
    uint32_t cU32(uint64_t v) { return b_.constInt(32, v); }
    uint32_t cU64(uint64_t v) { return b_.constInt(64, v); }
    uint32_t cInt(uint32_t w, uint64_t v) { return w == 64 ? cU64(v) : cU32(v); }

    // SPIR-V type an instruction computes with. Integers narrower than 32 bits are held in a
    // 32-bit value (sign-extended for signed types); f16 is not supported yet.
    uint32_t compType(const PtxType& t, const ptx::Instruction& ins) {
        switch (t.base) {
        case Base::Pred: return tBool();
        case Base::Float:
            if (t.width == 16) fail(ins, "f16 arithmetic is not supported yet");
            return tFloat(t.width);
        default: return t.width == 64 ? tU64() : tU32();
        }
    }
    uint32_t compIntWidth(const PtxType& t) const { return t.width == 64 ? 64 : 32; }

    uint32_t structPair(uint32_t t) {
        auto it = pairTypes_.find(t);
        if (it != pairTypes_.end()) return it->second;
        return pairTypes_[t] = b_.typeStruct({t, t});
    }

    // ---- registers ---------------------------------------------------------------------------

    struct Reg {
        PtxType type;
        uint32_t var = 0;
        uint32_t storage = 0;  // SPIR-V type of the variable: bool, u32 or u64
        uint32_t storageWidth = 0;
    };

    void declareRegisters() {
        for (const ptx::RegDecl& d : fn_.regs) {
            Reg r;
            if (!parseType(d.type, r.type)) {
                if (d.type == ".f16x2" || d.type == ".bf16" || d.type == ".bf16x2") r.type = {Base::Bit, 32};
                else fail("unsupported register type " + std::string(d.type) + " for " + d.name);
                if (d.type == ".bf16") r.type = {Base::Bit, 16};
            }
            if (r.type.base == Base::Pred) {
                r.storage = tBool();
                r.storageWidth = 1;
            } else {
                r.storageWidth = r.type.width == 64 ? 64 : 32;
                r.storage = b_.typeInt(r.storageWidth);
            }
            // Zero-initialized: PTX may read a register on paths that never wrote it (the value is
            // then simply unused or irrelevant on NVIDIA), and an uninitialized variable would let
            // the driver's optimizer assume whatever value suits it.
            r.var = b_.localVariable(r.storage, b_.constNull(r.storage));
            b_.name(r.var, d.name);
            regs_[d.name] = r;
        }
    }

    Reg& reg(const ptx::Instruction& ins, std::string_view name) {
        auto it = regs_.find(name);
        if (it == regs_.end()) fail(ins, "undeclared register " + std::string(name));
        return it->second;
    }

    uint32_t truncTo32(uint32_t v) { return b_.op(spv::OpUConvert, tU32(), {v}); }
    uint32_t extendTo64(uint32_t v, bool isSigned) {
        return b_.op(isSigned ? spv::OpSConvert : spv::OpUConvert, tU64(), {v});
    }
    // Canonical 32-bit form of a narrow integer: low `w` bits, sign- or zero-extended.
    uint32_t narrow(uint32_t v32, uint32_t w, bool isSigned) {
        if (w >= 32) return v32;
        if (isSigned) {
            const uint32_t sh = cU32(32 - w);
            return b_.op(spv::OpShiftRightArithmetic, tU32(), {b_.op(spv::OpShiftLeftLogical, tU32(), {v32, sh}), sh});
        }
        return b_.op(spv::OpBitwiseAnd, tU32(), {v32, cU32((uint64_t(1) << w) - 1)});
    }

    uint32_t loadReg(const ptx::Instruction& ins, std::string_view name, const PtxType& t) {
        Reg& r = reg(ins, name);
        uint32_t v = b_.op(spv::OpLoad, r.storage, {r.var});
        if (t.base == Base::Pred) {
            if (r.type.base != Base::Pred) fail(ins, std::string(name) + " is not a predicate");
            return v;
        }
        if (r.type.base == Base::Pred) fail(ins, "predicate " + std::string(name) + " used as a value");
        if (t.isFloat()) {
            if (t.width == 16) fail(ins, "f16 arithmetic is not supported yet");
            if (r.storageWidth != t.width) fail(ins, "register width does not match ." + std::to_string(t.width));
            return b_.op(spv::OpBitcast, tFloat(t.width), {v});
        }
        if (t.width == 64) return r.storageWidth == 64 ? v : extendTo64(v, t.isSigned());
        if (r.storageWidth == 64) v = truncTo32(v);
        return narrow(v, t.width, t.isSigned());
    }

    // Stores `v` (in compType(t)) into a register, following PTX's rules for registers that are
    // wider or narrower than the instruction type.
    void storeReg(const ptx::Instruction& ins, std::string_view name, const PtxType& t, uint32_t v) {
        Reg& r = reg(ins, name);
        if (t.base == Base::Pred || r.type.base == Base::Pred) {
            if (t.base != r.type.base) fail(ins, "predicate/value mismatch for " + std::string(name));
            b_.opVoid(spv::OpStore, {r.var, v});
            return;
        }
        PtxType it = t;
        if (t.isFloat()) {
            v = b_.op(spv::OpBitcast, b_.typeInt(t.width), {v});
            it.base = Base::Bit;
        }
        // A wider register receives the value extended according to the instruction type,
        // a narrower one its low bits. Narrow registers keep their bits zero-extended.
        if (it.width < 32) v = narrow(v, it.width, it.isSigned());
        if (r.storageWidth == 64 && it.width <= 32) v = extendTo64(v, it.isSigned());
        if (r.storageWidth == 32 && it.width == 64) v = truncTo32(v);
        if (r.type.width < 32) v = narrow(v, r.type.width, false);
        b_.opVoid(spv::OpStore, {r.var, v});
    }

    // ---- operands ----------------------------------------------------------------------------

    uint32_t immediate(const ptx::Instruction& ins, const ptx::Operand& o, const PtxType& t) {
        if (t.base == Base::Pred) {
            if (o.kind != ptx::OperandKind::Immediate) fail(ins, "invalid predicate immediate");
            return b_.constBool(o.imm != 0);
        }
        if (t.isFloat()) {
            if (t.width == 16) fail(ins, "f16 immediates are not supported yet");
            if (o.kind == ptx::OperandKind::Immediate) {
                const double d = double(int64_t(o.imm));
                return t.width == 64 ? b_.constFloat(64, std::bit_cast<uint64_t>(d))
                                     : b_.constFloat(32, std::bit_cast<uint32_t>(float(d)));
            }
            if (o.floatWidth == t.width) return b_.constFloat(t.width, o.imm);
            if (t.width == 32) return b_.constFloat(32, std::bit_cast<uint32_t>(float(std::bit_cast<double>(o.imm))));
            return b_.constFloat(64, std::bit_cast<uint64_t>(double(std::bit_cast<float>(uint32_t(o.imm)))));
        }
        uint64_t v = o.imm;  // FloatImm used with an integer type: raw bits
        if (t.width < 32) {
            const uint64_t mask = (uint64_t(1) << t.width) - 1;
            v &= mask;
            if (t.isSigned() && (v >> (t.width - 1))) v |= ~mask;
        }
        return cInt(compIntWidth(t), v);
    }

    uint32_t value(const ptx::Instruction& ins, const ptx::Operand& o, const PtxType& t) {
        uint32_t v = 0;
        switch (o.kind) {
        case ptx::OperandKind::Register: v = loadReg(ins, o.name, t); break;
        case ptx::OperandKind::Special: v = special(ins, o, t); break;
        case ptx::OperandKind::Immediate:
        case ptx::OperandKind::FloatImm: v = immediate(ins, o, t); break;
        case ptx::OperandKind::Symbol: {
            // mov.u64 %rd, globalVar / mov.u32 %r, sharedVar / mov.u64 %SPL, __local_depot0
            if (!t.isInt()) fail(ins, "variable address used with a non-integer type");
            if (o.name == "WARP_SZ") {  // the predefined constant (NVRTC output uses it)
                v = t.width == 64 ? cU64(32) : cU32(32);
                break;
            }
            const Space own = symbol(ins, o.name).space;
            v = symbolAddress(ins, o.name, own);
            if (t.width == 64 && isWordSpace(own)) v = extendTo64(v, false);
            if (t.width <= 32 && !isWordSpace(own)) v = truncTo32(v);
            break;
        }
        default: fail(ins, "unsupported operand");
        }
        if (o.negated) {
            if (t.base != Base::Pred) fail(ins, "'!' on a non-predicate operand");
            v = b_.op(spv::OpLogicalNot, tBool(), {v});
        }
        return v;
    }

    uint32_t builtinInput(uint32_t& var, spv::BuiltIn builtin) {
        if (!var) {
            const uint32_t vec = b_.typeVector(tU32(), 3);
            var = b_.globalVariable(b_.typePointer(spv::StorageClassInput, vec), spv::StorageClassInput);
            b_.decorate(var, spv::DecorationBuiltIn, {uint32_t(builtin)});
            interface_.push_back(var);
        }
        return var;
    }
    uint32_t builtinComponent(uint32_t& var, spv::BuiltIn builtin, uint32_t c) {
        const uint32_t ptr = b_.op(spv::OpAccessChain, b_.typePointer(spv::StorageClassInput, tU32()),
                                   {builtinInput(var, builtin), cU32(c)});
        return b_.op(spv::OpLoad, tU32(), {ptr});
    }

    // With block packing (KernelBinary::packable) the workgroup holds pack_ blocks side by side
    // along x; pack_ is a specialization constant, so for pack_ = 1 all of this folds away.
    uint32_t tidX() {
        const uint32_t x = builtinComponent(localId_, spv::BuiltInLocalInvocationId, 0);
        if (!pack_) return x;
        return b_.op(spv::OpSelect, tU32(), {packed_, b_.op(spv::OpUMod, tU32(), {x, blockDim_[0]}), x});
    }
    uint32_t ctaidX() {
        const uint32_t g = builtinComponent(groupId_, spv::BuiltInWorkgroupId, 0);
        if (!pack_) return g;
        const uint32_t sub = b_.op(spv::OpUDiv, tU32(), {builtinComponent(localId_, spv::BuiltInLocalInvocationId, 0), blockDim_[0]});
        const uint32_t v = b_.op(spv::OpIAdd, tU32(), {b_.op(spv::OpIMul, tU32(), {g, pack_}), sub});
        return b_.op(spv::OpSelect, tU32(), {packed_, v, g});
    }

    // Thread index within the block in CUDA's linear order (x fastest).
    uint32_t linearTid() {
        const uint32_t x = tidX();
        const uint32_t y = builtinComponent(localId_, spv::BuiltInLocalInvocationId, 1);
        const uint32_t z = builtinComponent(localId_, spv::BuiltInLocalInvocationId, 2);
        const uint32_t zy = b_.op(spv::OpIAdd, tU32(), {b_.op(spv::OpIMul, tU32(), {z, blockDim_[1]}), y});
        return b_.op(spv::OpIAdd, tU32(), {b_.op(spv::OpIMul, tU32(), {zy, blockDim_[0]}), x});
    }

    uint32_t special(const ptx::Instruction& ins, const ptx::Operand& o, const PtxType& t) {
        if (!t.isInt()) fail(ins, "special register read with a non-integer type");
        const std::string_view n = o.name;
        const uint32_t c = o.component ? uint32_t(o.component - 'x') : 0;
        if ((n == "%tid" || n == "%ntid" || n == "%ctaid" || n == "%nctaid") && (!o.component || c > 2))
            fail(ins, std::string(n) + " needs a .x/.y/.z component");
        uint32_t v;
        if (n == "%tid") v = c == 0 ? tidX() : builtinComponent(localId_, spv::BuiltInLocalInvocationId, c);
        else if (n == "%ntid") v = blockDim_[c];
        else if (n == "%ctaid") v = c == 0 ? ctaidX() : builtinComponent(groupId_, spv::BuiltInWorkgroupId, c);
        else if (n == "%nctaid") v = pushWord(gridOffset_ / 4 + c);  // the whole grid, even when split
        else if (n == "%laneid") v = laneId();
        else if (n == "%warpid") v = warpId();
        else if (n.starts_with("%lanemask_")) {
            const uint32_t lane = laneId();
            const uint32_t eq = b_.op(spv::OpShiftLeftLogical, tU32(), {cU32(1), lane});
            const uint32_t lt = b_.op(spv::OpISub, tU32(), {eq, cU32(1)});
            const uint32_t le = b_.op(spv::OpBitwiseOr, tU32(), {lt, eq});
            if (n == "%lanemask_eq") v = eq;
            else if (n == "%lanemask_lt") v = lt;
            else if (n == "%lanemask_le") v = le;
            else if (n == "%lanemask_gt") v = b_.op(spv::OpNot, tU32(), {le});
            else v = b_.op(spv::OpNot, tU32(), {lt});
        }
        else if (n == "%nwarpid") {
            const uint32_t size = b_.op(spv::OpIMul, tU32(),
                                        {b_.op(spv::OpIMul, tU32(), {blockDim_[0], blockDim_[1]}), blockDim_[2]});
            v = b_.op(spv::OpShiftRightLogical, tU32(), {b_.op(spv::OpIAdd, tU32(), {size, cU32(31)}), cU32(5)});
        } else if (n == "%warpsz") v = cU32(32);
        // The driver's environment: %envreg2:%envreg1 (low:high) is the grid workspace of cooperative
        // launches (cooperative groups' grid.sync()); the others are not used.
        else if (n == "%envreg1" || n == "%envreg2") v = pushWord(workspaceOffset_ / 4 + (n == "%envreg1"));
        else if (n.starts_with("%envreg")) v = cU32(0);
        else if (n == "%dynamic_smem_size") v = dynamicWords_ ? b_.op(spv::OpShiftLeftLogical, tU32(), {dynamicWords_, cU32(2)}) : cU32(0);
        else if (n == "%total_smem_size")
            v = b_.op(spv::OpIAdd, tU32(), {cU32(sharedStatic_), dynamicWords_ ? b_.op(spv::OpShiftLeftLogical, tU32(), {dynamicWords_, cU32(2)}) : cU32(0)});
        else fail(ins, "special register " + std::string(n) + " is not supported yet");
        return t.width == 64 ? extendTo64(v, false) : narrow(v, t.width, false);
    }

    // ---- memory ------------------------------------------------------------------------------
    //
    // Address spaces (see plan.md):
    //  - global: 64-bit buffer device addresses (PhysicalStorageBuffer pointers)
    //  - .global/.const module variables live in one buffer per module whose address is
    //    appended to the push constants after the parameters
    //  - shared: one Workgroup array of 32-bit words, addressed by byte offset. Static variables
    //    come first, the dynamic (extern) area follows, sized by specialization constant 3
    //  - local: one per-thread Function array of words, addressed by byte offset
    //  - generic: global addresses unchanged, shared/local offsets tagged in the top byte

    enum class Space { Global, Shared, Local, Param, Generic };
    static constexpr uint64_t kSharedTag = 0xFE, kLocalTag = 0xFD;

    struct Symbol {
        Space space;
        uint64_t offset;
    };

    static uint64_t alignTo(uint64_t v, uint64_t a) { return a > 1 ? (v + a - 1) / a * a : v; }

    // Assigns offsets to every variable the kernel can reach.
    void layoutMemory() {
        std::unordered_set<std::string_view> referenced;
        for (const ptx::Instruction& ins : fn_.instructions) {
            auto visit = [&](auto& self, const ptx::Operand& o) -> void {
                if ((o.kind == ptx::OperandKind::Symbol || o.kind == ptx::OperandKind::Address) && !o.baseIsRegister)
                    referenced.insert(o.name);
                for (const ptx::Operand& e : o.elems) self(self, e);
            };
            for (const ptx::Operand& o : ins.operands) visit(visit, o);
        }
        for (const ptx::Param& p : e_.params) symbols_[p.name] = {Space::Param, p.offset};

        std::vector<std::string_view> dynamicShared;
        uint64_t shared = 0;
        auto addShared = [&](std::string_view name, uint64_t size, uint32_t align) {
            shared = alignTo(shared, std::max<uint32_t>(align, 4));
            symbols_[name] = {Space::Shared, shared};
            shared += size;
        };
        for (const ptx::GlobalVar& g : m_.globals) {
            if (!referenced.count(g.name)) continue;
            if (g.space == ".shared") {
                if (g.isExtern || g.size == 0) dynamicShared.push_back(g.name);
                else addShared(g.name, g.size, g.align);
            } else if (g.isExtern) {
                fail("extern variable '" + g.name + "' is not supported (no linking of modules)");
            } else {
                symbols_[g.name] = {Space::Global, g.offset};
                usesGlobals_ = true;
            }
        }
        uint64_t local = 0;
        for (const ptx::LocalVar& v : fn_.locals) {
            if (v.space == ".shared") {
                if (v.size == 0) dynamicShared.push_back(v.name);
                else addShared(v.name, v.size, v.align);
            } else {
                local = alignTo(local, std::max<uint32_t>(v.align, 4));
                symbols_[v.name] = {Space::Local, local};
                local += v.size;
            }
        }
        bool usesBarrierReduction = false;
        for (const ptx::Instruction& ins : fn_.instructions)
            usesBarrierReduction |= (ins.opcode == "bar" || ins.opcode == "barrier") && ins.has(".red");
        if (usesBarrierReduction) {
            shared = alignTo(shared, 4);
            barrierWord_ = uint32_t(shared / 4);
            shared += 4;
        }
        if (!nativeWarps() && usesWarpOps()) {
            // 32 words per warp for up to 1024 threads
            shared = alignTo(shared, 4);
            warpScratchWord_ = uint32_t(shared / 4);
            shared += 32 * 32 * 4;
            VULKANIZER_WARN_ONCE("warp operations are emulated with workgroup barriers (subgroup size %u)", t_.subgroupSize);
        }
        sharedStatic_ = uint32_t(alignTo(shared, 16));
        for (std::string_view name : dynamicShared) symbols_[name] = {Space::Shared, sharedStatic_};
        hasShared_ = shared > 0 || !dynamicShared.empty() || usesSpace(".shared");
        localBytes_ = uint32_t(alignTo(local, 4));
        if (sharedStatic_ > t_.maxSharedMemorySize)
            fail(e_.name + ": " + std::to_string(sharedStatic_) + " bytes of shared memory exceed the device limit (" +
                 std::to_string(t_.maxSharedMemorySize) + ")");
    }

    bool usesSpace(std::string_view space) const {
        for (const ptx::Instruction& ins : fn_.instructions)
            if (ins.has(space)) return true;
        return false;
    }

    void declareMemory() {
        if (hasShared_) {
            dynamicWords_ = b_.specConstInt(32, 0, 3);
            const uint32_t len = b_.specConstOp(tU32(), spv::OpIAdd, {cU32(sharedStatic_ / 4 + 1), dynamicWords_});
            const uint32_t arr = b_.typeArrayOf(tU32(), len);
            sharedVar_ = b_.globalVariable(b_.typePointer(spv::StorageClassWorkgroup, arr), spv::StorageClassWorkgroup);
            b_.name(sharedVar_, "shared");
            interface_.push_back(sharedVar_);
        }
        if (localBytes_) {
            localVar_ = b_.localVariable(b_.typeArray(tU32(), localBytes_ / 4));
            b_.name(localVar_, "local");
        }
    }

    // Push constant block: the CUDA parameter buffer (or, when it does not fit, the address of a
    // per-launch copy of it), then the module variable buffer's address, then gridDim.
    void setupPushConstants() {
        for (const ptx::Instruction& ins : fn_.instructions) {
            usesPrintf_ |= ins.opcode == "vprintf";
            usesAssert_ |= ins.opcode == "assertfail";
            for (const ptx::Operand& o : ins.operands) {
                usesGrid_ |= o.kind == ptx::OperandKind::Special && o.name == "%nctaid";
                usesWorkspace_ |= o.kind == ptx::OperandKind::Special && (o.name == "%envreg1" || o.name == "%envreg2");
            }
        }
        auto layout = [&](uint32_t paramArea) {
            uint32_t at = paramArea;
            if (usesGlobals_) {
                globalsOffset_ = uint32_t(alignTo(at, 8));
                at = globalsOffset_ + 8;
            }
            if (usesGrid_) {
                gridOffset_ = uint32_t(alignTo(at, 4));
                at = gridOffset_ + 12;
            }
            if (usesPrintf_) {
                printfOffset_ = uint32_t(alignTo(at, 8));
                at = printfOffset_ + 8;
            }
            if (usesAssert_) {
                assertOffset_ = uint32_t(alignTo(at, 8));
                at = assertOffset_ + 8;
            }
            if (usesWorkspace_) {
                workspaceOffset_ = uint32_t(alignTo(at, 8));
                at = workspaceOffset_ + 8;
            }
            return uint32_t(alignTo(at, 4));
        };
        pushBytes_ = layout(e_.paramBytes);
        if (pushBytes_ > t_.maxPushConstantsSize) {
            paramsInBuffer_ = true;
            pushBytes_ = layout(8);
        }
        if (!pushBytes_) return;
        const uint32_t arr = b_.typeArray(tU32(), pushBytes_ / 4);
        b_.decorate(arr, spv::DecorationArrayStride, {4});
        const uint32_t st = b_.typeStruct({arr});
        b_.decorate(st, spv::DecorationBlock);
        b_.memberDecorate(st, 0, spv::DecorationOffset, {0});
        pushVar_ = b_.globalVariable(b_.typePointer(spv::StorageClassPushConstant, st), spv::StorageClassPushConstant);
        b_.name(pushVar_, "params");
        interface_.push_back(pushVar_);
    }

    uint32_t pushWord(uint32_t index) {
        const uint32_t ptr = b_.op(spv::OpAccessChain, b_.typePointer(spv::StorageClassPushConstant, tU32()),
                                   {pushVar_, cU32(0), cU32(index)});
        return b_.op(spv::OpLoad, tU32(), {ptr});
    }
    uint32_t pushU64(uint32_t byteOffset) {
        const uint32_t lo = pushWord(byteOffset / 4), hi = pushWord(byteOffset / 4 + 1);
        return b_.op(spv::OpBitcast, tU64(), {b_.op(spv::OpCompositeConstruct, b_.typeVector(tU32(), 2), {lo, hi})});
    }

    uint32_t pushWordAt(uint32_t indexId) {
        const uint32_t ptr = b_.op(spv::OpAccessChain, b_.typePointer(spv::StorageClassPushConstant, tU32()),
                                   {pushVar_, cU32(0), indexId});
        return b_.op(spv::OpLoad, tU32(), {ptr});
    }

    // Reads a scalar of type t at byte offset `off` (a u32 value, possibly dynamic) of the
    // parameter buffer.
    uint32_t paramAt(const ptx::Instruction& ins, uint32_t off, const PtxType& t) {
        const uint32_t width = t.base == Base::Pred ? 8 : t.width;
        if (paramsInBuffer_) {
            const uint32_t base = canonical(pushU64(0));
            Access a;
            a.type = t;
            if (width < 32 && !(width == 8 ? t_.storage8 : t_.storage16)) {
                // read the containing word instead of a sub-word load
                const uint32_t wordOff = b_.op(spv::OpBitwiseAnd, tU32(), {off, cU32(~3u)});
                a.type = {Base::Bit, 32};
                const uint32_t word = loadAt(ins, Space::Global, b_.op(spv::OpIAdd, tU64(), {base, extendTo64(wordOff, false)}), a)[0];
                return narrow(b_.op(spv::OpShiftRightLogical, tU32(), {word, byteShift(off)}), width, t.isSigned());
            }
            return loadAt(ins, Space::Global, b_.op(spv::OpIAdd, tU64(), {base, extendTo64(off, false)}), a)[0];
        }
        const uint32_t index = b_.op(spv::OpShiftRightLogical, tU32(), {off, cU32(2)});
        uint32_t bits;
        if (width == 64) {
            const uint32_t lo = pushWordAt(index), hi = pushWordAt(b_.op(spv::OpIAdd, tU32(), {index, cU32(1)}));
            bits = b_.op(spv::OpBitcast, tU64(), {b_.op(spv::OpCompositeConstruct, b_.typeVector(tU32(), 2), {lo, hi})});
        } else {
            bits = pushWordAt(index);
            if (width < 32)
                bits = narrow(b_.op(spv::OpShiftRightLogical, tU32(), {bits, byteShift(off)}), width, t.isSigned());
        }
        if (t.isFloat()) return b_.op(spv::OpBitcast, tFloat(t.width), {bits});
        return bits;
    }
    uint32_t byteShift(uint32_t byteAddr) {
        return b_.op(spv::OpShiftLeftLogical, tU32(), {b_.op(spv::OpBitwiseAnd, tU32(), {byteAddr, cU32(3)}), cU32(3)});
    }

    uint32_t paramScalar(const ptx::Instruction& ins, uint64_t offset, const PtxType& t) {
        const uint32_t size = t.base == Base::Pred ? 1 : t.width / 8;
        if (offset + size > e_.paramBytes) fail(ins, "parameter access out of range");
        if (offset % size) fail(ins, "misaligned parameter access");
        return paramAt(ins, cU32(offset), t);
    }

    const Symbol& symbol(const ptx::Instruction& ins, std::string_view name) {
        auto it = symbols_.find(name);
        if (it == symbols_.end()) fail(ins, "unknown variable '" + std::string(name) + "'");
        return it->second;
    }

    static bool isWordSpace(Space s) { return s == Space::Shared || s == Space::Local; }

    uint32_t tag(uint32_t offset32, uint64_t tagByte) {
        return b_.op(spv::OpBitwiseOr, tU64(), {extendTo64(offset32, false), cU64(tagByte << 56)});
    }

    // Address of a variable as seen from `want`: a 64-bit address for Global/Generic, a 32-bit
    // byte offset for Shared/Local.
    uint32_t symbolAddress(const ptx::Instruction& ins, std::string_view name, Space want) {
        const Symbol& s = symbol(ins, name);
        switch (s.space) {
        case Space::Global:
            if (isWordSpace(want)) fail(ins, "global variable used as a shared/local address");
            return b_.op(spv::OpIAdd, tU64(), {pushU64(globalsOffset_), cU64(s.offset)});
        case Space::Shared:
        case Space::Local:
            if (want == Space::Generic) return tag(cU32(s.offset), s.space == Space::Shared ? kSharedTag : kLocalTag);
            if (want != s.space) fail(ins, "variable used in the wrong address space");
            return cU32(s.offset);
        case Space::Param:
            // kernel parameter address (mov.b64 %rd, param): an offset in the parameter space
            if (want != Space::Param) fail(ins, "kernel parameters cannot be used as a generic address yet");
            return cU64(s.offset);
        default:
            fail(ins, "this variable cannot be addressed here");
        }
    }

    // Value of an address operand: u64 for Global/Generic, u32 byte offset for Shared/Local.
    uint32_t address(const ptx::Instruction& ins, const ptx::Operand& a, Space sp) {
        if (a.kind != ptx::OperandKind::Address) fail(ins, "expected an address operand");
        const bool word = isWordSpace(sp);
        uint32_t v;
        if (a.baseIsRegister) v = loadReg(ins, a.name, {Base::Unsigned, word ? 32u : 64u});
        else if (a.name.empty()) v = word ? cU32(0) : cU64(0);
        else v = symbolAddress(ins, a.name, sp);
        if (sp == Space::Global) v = canonical(v);  // generic ones once the space is known (withSpace)
        if (a.offset) v = b_.op(spv::OpIAdd, word ? tU32() : tU64(), {v, word ? cU32(uint64_t(a.offset)) : cU64(uint64_t(a.offset))});
        return v;
    }

    void selection(uint32_t cond, const std::function<void()>& thenBody, const std::function<void()>& elseBody,
                   spv::SelectionControlMask control = spv::SelectionControlMaskNone) {
        const uint32_t thenL = b_.newId(), elseL = b_.newId(), merge = b_.newId();
        b_.opVoid(spv::OpSelectionMerge, {merge, uint32_t(control)});
        b_.opVoid(spv::OpBranchConditional, {cond, thenL, elseL});
        b_.label(thenL);
        thenBody();
        b_.opVoid(spv::OpBranch, {merge});
        b_.label(elseL);
        elseBody();
        b_.opVoid(spv::OpBranch, {merge});
        b_.label(merge);
    }

    // Runs `body` for the concrete space(s) an address can be in; generic addresses branch at
    // run time on their tag (only between the spaces this kernel actually has).
    void withSpace(Space sp, uint32_t addr, const std::function<void(Space, uint32_t)>& body) {
        if (sp != Space::Generic) return body(sp, addr);
        auto global = [&] { body(Space::Global, canonical(addr)); };
        if (!hasShared_ && !localBytes_) return global();
        const uint32_t tagValue = truncTo32(b_.op(spv::OpShiftRightLogical, tU64(), {addr, cU32(56)}));
        auto local = [&] {
            if (!localBytes_) return global();
            selection(b_.op(spv::OpIEqual, tBool(), {tagValue, cU32(kLocalTag)}),
                      [&] { body(Space::Local, truncTo32(addr)); }, global);
        };
        if (!hasShared_) return local();
        selection(b_.op(spv::OpIEqual, tBool(), {tagValue, cU32(kSharedTag)}),
                  [&] { body(Space::Shared, truncTo32(addr)); }, local);
    }

    uint32_t wordPointer(Space sp, uint32_t wordIndex) {
        const spv::StorageClass sc = sp == Space::Shared ? spv::StorageClassWorkgroup : spv::StorageClassFunction;
        return b_.op(spv::OpAccessChain, b_.typePointer(sc, tU32()), {sp == Space::Shared ? sharedVar_ : localVar_, wordIndex});
    }
    uint32_t wordIndex(uint32_t byteAddr, uint32_t extraWords = 0) {
        uint32_t idx = b_.op(spv::OpShiftRightLogical, tU32(), {byteAddr, cU32(2)});
        return extraWords ? b_.op(spv::OpIAdd, tU32(), {idx, cU32(extraWords)}) : idx;
    }

    // Shared/local scalar load: returns the value in compType(t).
    uint32_t loadWords(Space sp, uint32_t byteAddr, const PtxType& t) {
        uint32_t bits;
        if (t.width == 64) {
            const uint32_t lo = b_.op(spv::OpLoad, tU32(), {wordPointer(sp, wordIndex(byteAddr))});
            const uint32_t hi = b_.op(spv::OpLoad, tU32(), {wordPointer(sp, wordIndex(byteAddr, 1))});
            bits = b_.op(spv::OpBitcast, tU64(), {b_.op(spv::OpCompositeConstruct, b_.typeVector(tU32(), 2), {lo, hi})});
        } else {
            bits = b_.op(spv::OpLoad, tU32(), {wordPointer(sp, wordIndex(byteAddr))});
            if (t.width < 32) {
                const uint32_t shift = b_.op(spv::OpShiftLeftLogical, tU32(),
                                             {b_.op(spv::OpBitwiseAnd, tU32(), {byteAddr, cU32(3)}), cU32(3)});
                bits = narrow(b_.op(spv::OpShiftRightLogical, tU32(), {bits, shift}), t.width, t.isSigned());
            }
        }
        return t.isFloat() ? b_.op(spv::OpBitcast, tFloat(t.width), {bits}) : bits;
    }

    void storeWords(Space sp, uint32_t byteAddr, const PtxType& t, uint32_t v) {
        uint32_t bits = t.isFloat() ? b_.op(spv::OpBitcast, b_.typeInt(t.width), {v}) : v;
        if (t.width == 64) {
            const uint32_t lo = truncTo32(bits);
            const uint32_t hi = truncTo32(b_.op(spv::OpShiftRightLogical, tU64(), {bits, cU32(32)}));
            b_.opVoid(spv::OpStore, {wordPointer(sp, wordIndex(byteAddr)), lo});
            b_.opVoid(spv::OpStore, {wordPointer(sp, wordIndex(byteAddr, 1)), hi});
            return;
        }
        const uint32_t ptr = wordPointer(sp, wordIndex(byteAddr));
        if (t.width == 32) {
            b_.opVoid(spv::OpStore, {ptr, bits});
            return;
        }
        // Sub-word store. Other threads may write the neighbouring bytes of the same shared word
        // concurrently, so shared memory clears and sets the bits with atomics.
        const uint32_t shift = b_.op(spv::OpShiftLeftLogical, tU32(),
                                     {b_.op(spv::OpBitwiseAnd, tU32(), {byteAddr, cU32(3)}), cU32(3)});
        const uint32_t low = cU32((uint64_t(1) << t.width) - 1);
        const uint32_t mask = b_.op(spv::OpShiftLeftLogical, tU32(), {low, shift});
        const uint32_t part =
            b_.op(spv::OpShiftLeftLogical, tU32(), {b_.op(spv::OpBitwiseAnd, tU32(), {bits, low}), shift});
        const uint32_t keep = b_.op(spv::OpNot, tU32(), {mask});
        if (sp == Space::Shared) {
            b_.op(spv::OpAtomicAnd, tU32(), {ptr, cU32(spv::ScopeWorkgroup), cU32(0), keep});
            b_.op(spv::OpAtomicOr, tU32(), {ptr, cU32(spv::ScopeWorkgroup), cU32(0), part});
        } else {
            const uint32_t old = b_.op(spv::OpLoad, tU32(), {ptr});
            b_.opVoid(spv::OpStore, {ptr, b_.op(spv::OpBitwiseOr, tU32(), {b_.op(spv::OpBitwiseAnd, tU32(), {old, keep}), part})});
        }
    }

    // Memory type for a global access of `count` elements of type t.
    uint32_t memType(const ptx::Instruction& ins, const PtxType& t, uint32_t count) {
        if (t.width == 8 && !t_.storage8) fail(ins, "8-bit global memory access needs storageBuffer8BitAccess");
        if (t.width == 16 && !t_.storage16) fail(ins, "16-bit global memory access needs storageBuffer16BitAccess");
        if (t.width == 8) b_.capability(spv::CapabilityStorageBuffer8BitAccess);
        if (t.width == 16) b_.capability(spv::CapabilityStorageBuffer16BitAccess);
        const uint32_t elem = tInt(t.width);
        return count > 1 ? b_.typeVector(elem, count) : elem;
    }

    // `addr` must be canonical (see canonical()).
    uint32_t globalPointer(uint32_t addr, uint32_t type) {
        return b_.op(spv::OpConvertUToPtr, b_.typePointer(spv::StorageClassPhysicalStorageBuffer, type), {addr});
    }

    // Device addresses come without their top bits (common/device_address.h): the device's bits
    // 63..47 (specialization constant 5) are set again, which also turns a host pointer into the
    // address its pages have on the device (pinned, registered and managed memory). Done on the high
    // word alone (one 32-bit OR), and to the base before a constant offset is added, so the GPU
    // compiler can still fold offsets into its loads and stores.
    uint32_t canonical(uint32_t addr) {
        if (!addressHigh_) addressHigh_ = b_.specConstInt(32, 0xFFFF8000u, 5);
        const uint32_t v2 = b_.typeVector(tU32(), 2);
        const uint32_t words = b_.op(spv::OpBitcast, v2, {addr});
        const uint32_t lo = b_.op(spv::OpCompositeExtract, tU32(), {words, 0});
        const uint32_t hi = b_.op(spv::OpCompositeExtract, tU32(), {words, 1});
        const uint32_t hiExt = b_.op(spv::OpBitwiseOr, tU32(), {hi, addressHigh_});
        return b_.op(spv::OpBitcast, tU64(), {b_.op(spv::OpCompositeConstruct, v2, {lo, hiExt})});
    }

    // Memory bits (int of t.width) -> computation value.
    uint32_t fromMemory(uint32_t bits, const PtxType& t) {
        if (t.width < 32) return b_.op(t.isSigned() ? spv::OpSConvert : spv::OpUConvert, tU32(), {bits});
        if (t.isFloat()) return b_.op(spv::OpBitcast, tFloat(t.width), {bits});
        return bits;
    }
    uint32_t toMemory(uint32_t v, const PtxType& t) {
        if (t.isFloat()) return b_.op(spv::OpBitcast, b_.typeInt(t.width), {v});
        if (t.width < 32) return b_.op(spv::OpUConvert, tInt(t.width), {v});
        return v;
    }

    struct Access {
        PtxType type;
        uint32_t count = 1;  // .v2 / .v4
        Space space = Space::Generic;
        bool isVolatile = false;
    };

    Access accessOf(const ptx::Instruction& ins) {
        Access a;
        bool typed = false;
        for (std::string_view m : ins.mods) {
            if (m == ".v2") a.count = 2;
            else if (m == ".v4") a.count = 4;
            else if (m == ".global" || m == ".const") a.space = Space::Global;
            else if (m == ".shared") a.space = Space::Shared;
            else if (m == ".local") a.space = Space::Local;
            else if (m == ".param") a.space = Space::Param;
            // Accesses other threads may be racing with (spin-waits on flags, e.g. CUB's decoupled
            // look-back): never served from a stale non-coherent cache line.
            else if (m == ".volatile" || m == ".relaxed" || m == ".acquire" || m == ".release" || m == ".cg" ||
                     m == ".cv")
                a.isVolatile = true;
            else if (parseType(m, a.type)) typed = true;
        }
        if (!typed) fail(ins, "missing type");
        if (a.type.base == Base::Float && a.type.width == 16) a.type.base = Base::Bit;
        return a;
    }

    std::vector<const ptx::Operand*> dataOperands(const ptx::Instruction& ins, const ptx::Operand& o, uint32_t count) {
        std::vector<const ptx::Operand*> out;
        if (count == 1 && o.kind == ptx::OperandKind::Vector && o.elems.size() == 1) {
            out.push_back(&o.elems[0]);  // Triton writes scalar ld/st data as {%r1}
        } else if (count == 1) {
            out.push_back(&o);
        } else {
            if (o.kind != ptx::OperandKind::Vector || o.elems.size() != count) fail(ins, "vector operand expected");
            for (const ptx::Operand& e : o.elems) out.push_back(&e);
        }
        return out;
    }

    std::vector<uint32_t> loadAt(const ptx::Instruction& ins, Space sp, uint32_t addr, const Access& a) {
        std::vector<uint32_t> values;
        const uint32_t elemBytes = a.type.width / 8;
        if (sp == Space::Global) {
            const uint32_t type = memType(ins, a.type, a.count);
            std::vector<uint32_t> ops{globalPointer(addr, type), uint32_t(spv::MemoryAccessAlignedMask),
                                      elemBytes * a.count};
            // Across grid.sync() a block reads what other blocks wrote, at addresses it may have read
            // before: without Volatile the driver reuses the earlier loads (or their cache lines).
            if (a.isVolatile || usesWorkspace_) ops[1] |= spv::MemoryAccessVolatileMask;
            const uint32_t loaded = b_.op(spv::OpLoad, type, ops);
            for (uint32_t i = 0; i < a.count; ++i) {
                const uint32_t bits =
                    a.count > 1 ? b_.op(spv::OpCompositeExtract, tInt(a.type.width), {loaded, i}) : loaded;
                values.push_back(fromMemory(bits, a.type));
            }
        } else {
            for (uint32_t i = 0; i < a.count; ++i) {
                const uint32_t at = i ? b_.op(spv::OpIAdd, tU32(), {addr, cU32(i * elemBytes)}) : addr;
                values.push_back(loadWords(sp, at, a.type));
            }
        }
        return values;
    }

    void storeAt(const ptx::Instruction& ins, Space sp, uint32_t addr, const Access& a,
                 const std::vector<uint32_t>& values) {
        const uint32_t elemBytes = a.type.width / 8;
        if (sp == Space::Global) {
            const uint32_t type = memType(ins, a.type, a.count);
            std::vector<uint32_t> parts;
            for (uint32_t v : values) parts.push_back(toMemory(v, a.type));
            const uint32_t data = a.count > 1 ? b_.op(spv::OpCompositeConstruct, type, parts) : parts[0];
            std::vector<uint32_t> ops{globalPointer(addr, type), data, uint32_t(spv::MemoryAccessAlignedMask),
                                      elemBytes * a.count};
            if (a.isVolatile || usesWorkspace_) ops[2] |= spv::MemoryAccessVolatileMask;
            b_.opVoid(spv::OpStore, ops);
            return;
        }
        for (uint32_t i = 0; i < a.count; ++i) {
            const uint32_t at = i ? b_.op(spv::OpIAdd, tU32(), {addr, cU32(i * elemBytes)}) : addr;
            storeWords(sp, at, a.type, values[i]);
        }
    }

    void emitLd(const ptx::Instruction& ins) {
        if (ins.operands.size() != 2) fail(ins, "ld needs 2 operands");
        const Access a = accessOf(ins);
        const auto dst = dataOperands(ins, ins.operands[0], a.count);
        const ptx::Operand& addr = ins.operands[1];
        auto writeBack = [&](const std::vector<uint32_t>& values) {
            for (uint32_t i = 0; i < a.count; ++i) {
                if (dst[i]->kind == ptx::OperandKind::Sink) continue;
                if (dst[i]->kind != ptx::OperandKind::Register) fail(ins, "ld destination must be a register");
                storeReg(ins, dst[i]->name, a.type, values[i]);
            }
        };
        Access acc = a;
        if (paramIsLocal(ins, addr)) acc.space = Space::Local;  // call arguments of inlined functions
        if (acc.space == Space::Param && addr.kind == ptx::OperandKind::Address && addr.baseIsRegister) {
            // ld.param through a register holding a parameter-space address
            uint32_t off = truncTo32(address(ins, addr, Space::Param));
            std::vector<uint32_t> values;
            for (uint32_t i = 0; i < a.count; ++i)
                values.push_back(paramAt(ins, i ? b_.op(spv::OpIAdd, tU32(), {off, cU32(i * (a.type.width / 8))}) : off, a.type));
            writeBack(values);
            return;
        }
        if (acc.space == Space::Param) {
            if (addr.kind != ptx::OperandKind::Address) fail(ins, "unsupported .param address");
            const Symbol& p = symbol(ins, addr.name);
            if (p.space != Space::Param) fail(ins, std::string(addr.name) + " is not a parameter");
            std::vector<uint32_t> values;
            for (uint32_t i = 0; i < a.count; ++i)
                values.push_back(paramScalar(ins, p.offset + uint64_t(addr.offset) + i * (a.type.width / 8), a.type));
            writeBack(values);
            return;
        }
        withSpace(acc.space, address(ins, addr, acc.space),
                  [&](Space sp, uint32_t at) { writeBack(loadAt(ins, sp, at, a)); });
    }

    // ld/st.param on a variable declared in the body: an inlined call's argument or result.
    bool paramIsLocal(const ptx::Instruction& ins, const ptx::Operand& addr) {
        if (!ins.has(".param") || addr.kind != ptx::OperandKind::Address || addr.baseIsRegister) return false;
        auto it = symbols_.find(addr.name);
        return it != symbols_.end() && it->second.space == Space::Local;
    }

    void emitSt(const ptx::Instruction& ins) {
        if (ins.operands.size() != 2) fail(ins, "st needs 2 operands");
        Access a = accessOf(ins);
        if (paramIsLocal(ins, ins.operands[0])) a.space = Space::Local;
        if (a.space == Space::Param) fail(ins, "st.param to a kernel parameter");
        std::vector<uint32_t> values;
        for (const ptx::Operand* o : dataOperands(ins, ins.operands[1], a.count)) values.push_back(value(ins, *o, a.type));
        withSpace(a.space, address(ins, ins.operands[0], a.space),
                  [&](Space sp, uint32_t at) { storeAt(ins, sp, at, a, values); });
    }

    // cp.async.{ca,cg}.shared{::cta}.global{.L2::cache_hint}{.L2::NB} [dst], [src], cp-size
    //     {, src-size | ignore-src}{, cache-policy}
    // Done right away as a global load and a shared store, so commit_group, wait_group and wait_all
    // have nothing to wait for. Bytes from src-size on are zero (and not read); ignore-src zeroes all.
    void emitCpAsync(const ptx::Instruction& ins) {
        if (ins.has(".commit_group") || ins.has(".wait_group") || ins.has(".wait_all")) return;
        if (ins.has(".bulk") || ins.has(".mbarrier")) fail(ins, "cp.async.bulk / cp.async.mbarrier are not supported");
        if (!ins.has(".shared") || !ins.has(".global")) fail(ins, "cp.async must copy from .global to .shared");
        size_t n = ins.operands.size();
        if (ins.has(".L2::cache_hint")) --n;  // the cache policy
        if (n < 3 || ins.operands[2].kind != ptx::OperandKind::Immediate) fail(ins, "malformed cp.async");
        const uint32_t bytes = uint32_t(ins.operands[2].imm);
        if (bytes != 4 && bytes != 8 && bytes != 16) fail(ins, "cp.async size must be 4, 8 or 16");
        const PtxType u32t{Base::Unsigned, 32};
        uint32_t srcSize = cU32(bytes);
        if (n >= 4) {
            const ptx::Operand& o = ins.operands[3];
            if (o.kind == ptx::OperandKind::Register && reg(ins, o.name).type.base == Base::Pred) {
                const uint32_t ignore = value(ins, o, {Base::Pred, 1});
                srcSize = b_.op(spv::OpSelect, tU32(), {ignore, cU32(0), cU32(bytes)});
            } else {
                srcSize = glsl(tU32(), GLSLstd450UMin, {value(ins, o, u32t), cU32(bytes)});
            }
        }
        const uint32_t dst = address(ins, ins.operands[0], Space::Shared);
        const uint32_t src = address(ins, ins.operands[1], Space::Global);
        Access a32;
        a32.type = {Base::Bit, 32};
        a32.isVolatile = ins.has(".cg");  // .cg bypasses L1, as ld.cg does
        const uint32_t word = b_.localVariable(tU32());
        for (uint32_t w = 0; w < bytes / 4; ++w) {
            // bytes of this word below src-size: 0..4
            const uint32_t valid = glsl(tU32(), GLSLstd450UMin,
                                        {b_.op(spv::OpISub, tU32(), {glsl(tU32(), GLSLstd450UMax, {srcSize, cU32(4 * w)}),
                                                                     cU32(4 * w)}),
                                         cU32(4)});
            b_.opVoid(spv::OpStore, {word, cU32(0)});
            selection(b_.op(spv::OpUGreaterThan, tBool(), {valid, cU32(0)}), [&] {
                uint32_t v = loadAt(ins, Space::Global, b_.op(spv::OpIAdd, tU64(), {src, cU64(4 * w)}), a32)[0];
                // keep the low `valid` bytes: mask = (1 << 8 * valid) - 1, computed in 64 bits for valid = 4
                const uint32_t mask = truncTo32(b_.op(
                    spv::OpISub, tU64(),
                    {b_.op(spv::OpShiftLeftLogical, tU64(), {cU64(1), b_.op(spv::OpIMul, tU32(), {valid, cU32(8)})}), cU64(1)}));
                v = b_.op(spv::OpBitwiseAnd, tU32(), {v, mask});
                b_.opVoid(spv::OpStore, {word, v});
            }, [] {});
            storeAt(ins, Space::Shared, b_.op(spv::OpIAdd, tU32(), {dst, cU32(4 * w)}), a32,
                    {b_.op(spv::OpLoad, tU32(), {word})});
        }
    }

    // Space named by a cvta/isspacep modifier.
    Space spaceModifier(const ptx::Instruction& ins) {
        if (ins.has(".shared")) return Space::Shared;
        if (ins.has(".local")) return Space::Local;
        if (ins.has(".global") || ins.has(".const")) return Space::Global;
        fail(ins, "unsupported address space");
    }

    // cvta.<space> (to generic) and cvta.to.<space> (from generic).
    void emitCvta(const ptx::Instruction& ins) {
        expectOperands(ins, 2);
        typeOf(ins);  // validates the type modifier
        const Space sp = spaceModifier(ins);
        const bool toSpace = ins.has(".to");
        const ptx::Operand& src = ins.operands[1];
        uint32_t r;
        if (sp == Space::Global) {
            r = src.kind == ptx::OperandKind::Symbol ? symbolAddress(ins, src.name, Space::Global)
                                                     : value(ins, src, {Base::Unsigned, 64});
        } else if (toSpace) {
            r = b_.op(spv::OpBitwiseAnd, tU64(), {value(ins, src, {Base::Unsigned, 64}), cU64(0xFFFFFFFFull)});
        } else {
            const uint32_t off = src.kind == ptx::OperandKind::Symbol ? symbolAddress(ins, src.name, sp)
                                                                       : value(ins, src, {Base::Unsigned, 32});
            r = tag(off, sp == Space::Shared ? kSharedTag : kLocalTag);
        }
        storeReg(ins, dstReg(ins).name, {Base::Unsigned, 64}, r);
    }

    // isspacep.<space> p, a
    void emitIsspacep(const ptx::Instruction& ins) {
        expectOperands(ins, 2);
        const Space sp = spaceModifier(ins);
        const uint32_t addr = value(ins, ins.operands[1], {Base::Unsigned, 64});
        const uint32_t tagValue = truncTo32(b_.op(spv::OpShiftRightLogical, tU64(), {addr, cU32(56)}));
        uint32_t r;
        if (sp == Space::Global)
            r = b_.op(spv::OpLogicalAnd, tBool(), {b_.op(spv::OpINotEqual, tBool(), {tagValue, cU32(kSharedTag)}),
                                                   b_.op(spv::OpINotEqual, tBool(), {tagValue, cU32(kLocalTag)})});
        else
            r = b_.op(spv::OpIEqual, tBool(), {tagValue, cU32(sp == Space::Shared ? kSharedTag : kLocalTag)});
        storeReg(ins, dstReg(ins).name, {Base::Pred, 1}, r);
    }

    // ---- synchronization and atomics ---------------------------------------------------------

    static constexpr uint32_t kSemAcqRelAll = uint32_t(spv::MemorySemanticsAcquireReleaseMask) |
                                              uint32_t(spv::MemorySemanticsUniformMemoryMask) |
                                              uint32_t(spv::MemorySemanticsWorkgroupMemoryMask);

    // Memory order of ld/st/atom/red (.relaxed when not given): .release orders the earlier accesses
    // before it, .acquire the later ones after it, as fences of the instruction's scope around the
    // access (cooperative groups' grid.sync(): atom.add.release.gpu, then a spin on ld.acquire.gpu).
    void emitOrdered(const ptx::Instruction& ins) {
        const bool acqRel = ins.has(".acq_rel");
        const bool release = acqRel || ins.has(".release"), acquire = acqRel || ins.has(".acquire");
        const uint32_t scope = ins.has(".cta") ? spv::ScopeWorkgroup : spv::ScopeDevice;
        constexpr uint32_t kStorage =
            uint32_t(spv::MemorySemanticsUniformMemoryMask) | uint32_t(spv::MemorySemanticsWorkgroupMemoryMask);
        if (release)
            b_.opVoid(spv::OpMemoryBarrier, {cU32(scope), cU32(uint32_t(spv::MemorySemanticsReleaseMask) | kStorage)});
        const std::string_view op = ins.opcode;
        if (op == "ld") emitLd(ins);
        else if (op == "st") emitSt(ins);
        else emitAtom(ins);
        if (acquire)
            b_.opVoid(spv::OpMemoryBarrier, {cU32(scope), cU32(uint32_t(spv::MemorySemanticsAcquireMask) | kStorage)});
    }

    // bar.sync / barrier.sync / bar.warp.sync / membar / fence
    void emitBarrier(const ptx::Instruction& ins) {
        const std::string_view op = ins.opcode;
        if (op == "membar" || op == "fence") {
            const uint32_t scope = ins.has(".cta") ? spv::ScopeWorkgroup : spv::ScopeDevice;
            b_.opVoid(spv::OpMemoryBarrier, {cU32(scope), cU32(kSemAcqRelAll)});
            return;
        }
        if (ins.has(".warp")) {  // bar.warp.sync: warps are always converged here
            b_.opVoid(spv::OpMemoryBarrier, {cU32(spv::ScopeWorkgroup), cU32(kSemAcqRelAll)});
            return;
        }
        if (ins.has(".red")) return emitBarrierReduction(ins);
        if (ins.has(".arrive")) fail(ins, "bar.arrive is not supported yet");
        if (ins.operands.size() > 1) VULKANIZER_WARN_ONCE("bar.sync with a thread count is treated as a full block barrier");
        // In grid.sync() only one thread per block orders the block's global accesses at GPU scope
        // (atom.release / ld.acquire around bar.sync), which PTX makes cumulative over the barrier.
        // Vulkan's workgroup-scope barrier is not (on AMD, the other waves' stores may still be in
        // flight), so kernels using the grid workspace get device-scope barriers.
        const uint32_t memoryScope = usesWorkspace_ ? spv::ScopeDevice : spv::ScopeWorkgroup;
        b_.opVoid(spv::OpControlBarrier, {cU32(spv::ScopeWorkgroup), cU32(memoryScope), cU32(kSemAcqRelAll)});
    }

    // bar.red.popc.u32 d, a{, b}, {!}c   bar.red.{and,or}.pred p, a{, b}, {!}c
    // (__syncthreads_count/_and/_or): a block-wide count of true predicates in a reserved word.
    void emitBarrierReduction(const ptx::Instruction& ins) {
        if (ins.operands.size() < 3) fail(ins, "bar.red needs a destination, a barrier and a predicate");
        const uint32_t pred = value(ins, ins.operands.back(), {Base::Pred, 1});
        const uint32_t word = wordPointer(Space::Shared, cU32(barrierWord_));
        workgroupBarrier();  // everyone has read the previous result
        selection(b_.op(spv::OpIEqual, tBool(), {linearTid(), cU32(0)}), [&] { b_.opVoid(spv::OpStore, {word, cU32(0)}); },
                  [] {});
        workgroupBarrier();
        selection(pred, [&] { b_.op(spv::OpAtomicIAdd, tU32(), {word, cU32(spv::ScopeWorkgroup), cU32(0), cU32(1)}); },
                  [] {});
        workgroupBarrier();
        const uint32_t count = b_.op(spv::OpLoad, tU32(), {word});
        const ptx::Operand& d = dstReg(ins);
        if (ins.has(".popc")) {
            storeReg(ins, d.name, {Base::Unsigned, 32}, count);
        } else if (ins.has(".or")) {
            storeReg(ins, d.name, {Base::Pred, 1}, b_.op(spv::OpINotEqual, tBool(), {count, cU32(0)}));
        } else if (ins.has(".and")) {
            const uint32_t threads = b_.op(spv::OpIMul, tU32(), {b_.op(spv::OpIMul, tU32(), {blockDim_[0], blockDim_[1]}), blockDim_[2]});
            storeReg(ins, d.name, {Base::Pred, 1}, b_.op(spv::OpIEqual, tBool(), {count, threads}));
        } else {
            fail(ins, "unsupported barrier reduction");
        }
    }

    // Compare-and-swap loop applying `update` to the current value; returns the old value.
    uint32_t casLoop(uint32_t ptr, uint32_t type, uint32_t scope, const std::function<uint32_t(uint32_t)>& update) {
        const uint32_t var = b_.localVariable(type);
        b_.opVoid(spv::OpStore, {var, b_.op(spv::OpAtomicLoad, type, {ptr, cU32(scope), cU32(0)})});
        const uint32_t header = b_.newId(), body = b_.newId(), cont = b_.newId(), merge = b_.newId();
        b_.opVoid(spv::OpBranch, {header});
        b_.label(header);
        b_.opVoid(spv::OpLoopMerge, {merge, cont, uint32_t(spv::LoopControlMaskNone)});
        b_.opVoid(spv::OpBranch, {body});
        b_.label(body);
        const uint32_t old = b_.op(spv::OpLoad, type, {var});
        const uint32_t got =
            b_.op(spv::OpAtomicCompareExchange, type, {ptr, cU32(scope), cU32(0), cU32(0), update(old), old});
        b_.opVoid(spv::OpStore, {var, got});
        b_.opVoid(spv::OpBranchConditional, {b_.op(spv::OpIEqual, tBool(), {got, old}), merge, cont});
        b_.label(cont);
        b_.opVoid(spv::OpBranch, {header});
        b_.label(merge);
        return old;
    }

    // A read-modify-write of one word: atomic (a CAS loop) in shared and global memory, plain in
    // local memory, which no other thread can see (Vulkan has no atomics on Function storage).
    uint32_t rmwWord(Space sp, uint32_t ptr, uint32_t type, uint32_t scope,
                     const std::function<uint32_t(uint32_t)>& update) {
        if (sp != Space::Local) return casLoop(ptr, type, scope, update);
        const uint32_t old = b_.op(spv::OpLoad, type, {ptr});
        b_.opVoid(spv::OpStore, {ptr, update(old)});
        return old;
    }

    // atom.{space}.{op}.{type} d, [a], b{, c}   red.{space}.{op}.{type} [a], b
    void emitAtom(const ptx::Instruction& ins) {
        const bool isRed = ins.opcode == "red";
        const size_t first = isRed ? 0 : 1;
        if (ins.operands.size() < first + 2) fail(ins, "missing operands");
        static constexpr std::string_view ops[] = {".add", ".min", ".max", ".inc", ".dec", ".exch",
                                                   ".cas", ".and", ".or",  ".xor"};
        std::string_view op;
        for (std::string_view m : ins.mods)
            if (std::find(std::begin(ops), std::end(ops), m) != std::end(ops)) op = m;
        if (op.empty()) fail(ins, "missing atomic operation");
        if (op == ".cas" && ins.operands.size() != first + 3) fail(ins, "cas needs 3 source operands");
        if (const Half h = halfOf(ins); h != Half::None) return emitHalfAtom(ins, h, op, isRed, first);
        Access a = accessOf(ins);
        const PtxType t = a.type;
        if (t.width != 32 && t.width != 64) fail(ins, "only 32- and 64-bit atomics are supported");
        if (t.width == 64) b_.capability(spv::CapabilityInt64Atomics);
        const uint32_t itype = b_.typeInt(t.width);
        const uint32_t b = value(ins, ins.operands[first + 1], t);
        const uint32_t c = op == ".cas" ? value(ins, ins.operands[first + 2], t) : 0;
        const bool cta = ins.has(".cta");

        withSpace(a.space, address(ins, ins.operands[first], a.space), [&](Space sp, uint32_t at) {
            uint32_t ptr;
            if (sp == Space::Local) {
                // the thread's own memory: a plain read-modify-write is atomic
                auto asBits = [&](uint32_t v) { return t.isFloat() ? b_.op(spv::OpBitcast, itype, {v}) : v; };
                const uint32_t bv = asBits(b), ft = tFloat(t.width);
                auto update = [&](uint32_t cur) {
                    if (t.isFloat()) {
                        const uint32_t f = b_.op(spv::OpBitcast, ft, {cur});
                        uint32_t r = b;
                        if (op == ".add") r = b_.op(spv::OpFAdd, ft, {f, b});
                        else if (op == ".min") r = glsl(ft, GLSLstd450NMin, {f, b});
                        else if (op == ".max") r = glsl(ft, GLSLstd450NMax, {f, b});
                        else if (op != ".exch") fail(ins, "unsupported float atomic");
                        return b_.op(spv::OpBitcast, itype, {r});
                    }
                    const bool sgn = t.isSigned();
                    if (op == ".add") return b_.op(spv::OpIAdd, itype, {cur, bv});
                    if (op == ".and") return b_.op(spv::OpBitwiseAnd, itype, {cur, bv});
                    if (op == ".or") return b_.op(spv::OpBitwiseOr, itype, {cur, bv});
                    if (op == ".xor") return b_.op(spv::OpBitwiseXor, itype, {cur, bv});
                    if (op == ".exch") return bv;
                    if (op == ".min") return glsl(itype, sgn ? GLSLstd450SMin : GLSLstd450UMin, {cur, bv});
                    if (op == ".max") return glsl(itype, sgn ? GLSLstd450SMax : GLSLstd450UMax, {cur, bv});
                    if (op == ".cas")
                        return b_.op(spv::OpSelect, itype, {b_.op(spv::OpIEqual, tBool(), {cur, asBits(c)}), bv, cur});
                    const uint32_t one = cInt(32, 1), zero = cInt(32, 0);
                    if (op == ".inc")
                        return b_.op(spv::OpSelect, itype, {b_.op(spv::OpUGreaterThanEqual, tBool(), {cur, bv}), zero,
                                                            b_.op(spv::OpIAdd, itype, {cur, one})});
                    const uint32_t wrap = b_.op(spv::OpLogicalOr, tBool(), {b_.op(spv::OpIEqual, tBool(), {cur, zero}),
                                                                         b_.op(spv::OpUGreaterThan, tBool(), {cur, bv})});
                    return b_.op(spv::OpSelect, itype, {wrap, bv, b_.op(spv::OpISub, itype, {cur, one})});
                };
                uint32_t old;
                if (t.width == 64) {  // two words
                    const uint32_t index = wordIndex(at), v2 = b_.typeVector(tU32(), 2);
                    const uint32_t p0 = wordPointer(sp, index), p1 = wordPointer(sp, b_.op(spv::OpIAdd, tU32(), {index, cU32(1)}));
                    old = b_.op(spv::OpBitcast, itype, {b_.op(spv::OpCompositeConstruct, v2,
                                                              {b_.op(spv::OpLoad, tU32(), {p0}), b_.op(spv::OpLoad, tU32(), {p1})})});
                    const uint32_t words = b_.op(spv::OpBitcast, v2, {update(old)});
                    b_.opVoid(spv::OpStore, {p0, b_.op(spv::OpCompositeExtract, tU32(), {words, 0})});
                    b_.opVoid(spv::OpStore, {p1, b_.op(spv::OpCompositeExtract, tU32(), {words, 1})});
                } else {
                    old = rmwWord(sp, wordPointer(sp, wordIndex(at)), itype, spv::ScopeInvocation, update);
                }
                if (!isRed)
                    storeReg(ins, dstReg(ins).name, t, t.isFloat() ? b_.op(spv::OpBitcast, ft, {old}) : old);
                return;
            }
            if (sp == Space::Shared) {
                if (t.width == 64) fail(ins, "64-bit shared memory atomics are not supported yet");
                ptr = wordPointer(sp, wordIndex(at));
            } else {
                ptr = globalPointer(at, itype);
            }
            const uint32_t scope = sp == Space::Shared || cta ? spv::ScopeWorkgroup : spv::ScopeDevice;
            const uint32_t sc = cU32(scope), sem = cU32(0);
            auto bits = [&](uint32_t v) { return t.isFloat() ? b_.op(spv::OpBitcast, itype, {v}) : v; };
            const uint32_t bb = bits(b);
            uint32_t old;
            if (t.isFloat()) {
                if (op != ".add" && op != ".exch" && op != ".min" && op != ".max")
                    fail(ins, "unsupported float atomic");
                if (op == ".exch") {
                    old = b_.op(spv::OpAtomicExchange, itype, {ptr, sc, sem, bb});
                } else {
                    const uint32_t ft = tFloat(t.width);
                    old = casLoop(ptr, itype, scope, [&](uint32_t cur) {
                        const uint32_t f = b_.op(spv::OpBitcast, ft, {cur});
                        uint32_t r;
                        if (op == ".add") r = b_.op(spv::OpFAdd, ft, {f, b});
                        else r = glsl(ft, op == ".min" ? GLSLstd450NMin : GLSLstd450NMax, {f, b});
                        return b_.op(spv::OpBitcast, itype, {r});
                    });
                }
            } else if (op == ".add") old = b_.op(spv::OpAtomicIAdd, itype, {ptr, sc, sem, bb});
            else if (op == ".and") old = b_.op(spv::OpAtomicAnd, itype, {ptr, sc, sem, bb});
            else if (op == ".or") old = b_.op(spv::OpAtomicOr, itype, {ptr, sc, sem, bb});
            else if (op == ".xor") old = b_.op(spv::OpAtomicXor, itype, {ptr, sc, sem, bb});
            else if (op == ".exch") old = b_.op(spv::OpAtomicExchange, itype, {ptr, sc, sem, bb});
            else if (op == ".min") old = b_.op(t.isSigned() ? spv::OpAtomicSMin : spv::OpAtomicUMin, itype, {ptr, sc, sem, bb});
            else if (op == ".max") old = b_.op(t.isSigned() ? spv::OpAtomicSMax : spv::OpAtomicUMax, itype, {ptr, sc, sem, bb});
            else if (op == ".cas") old = b_.op(spv::OpAtomicCompareExchange, itype, {ptr, sc, sem, sem, bits(c), bb});
            else {
                // inc: old >= b ? 0 : old + 1     dec: (old == 0 || old > b) ? b : old - 1
                const bool inc = op == ".inc";
                old = casLoop(ptr, itype, scope, [&](uint32_t cur) {
                    const uint32_t one = cInt(t.width, 1), zero = cInt(t.width, 0);
                    if (inc) {
                        const uint32_t wrap = b_.op(spv::OpUGreaterThanEqual, tBool(), {cur, bb});
                        return b_.op(spv::OpSelect, itype, {wrap, zero, b_.op(spv::OpIAdd, itype, {cur, one})});
                    }
                    const uint32_t wrap = b_.op(spv::OpLogicalOr, tBool(), {b_.op(spv::OpIEqual, tBool(), {cur, zero}),
                                                                         b_.op(spv::OpUGreaterThan, tBool(), {cur, bb})});
                    return b_.op(spv::OpSelect, itype, {wrap, bb, b_.op(spv::OpISub, itype, {cur, one})});
                });
            }
            if (!isRed) {
                const uint32_t result = t.isFloat() ? b_.op(spv::OpBitcast, tFloat(t.width), {old}) : old;
                storeReg(ins, dstReg(ins).name, t, result);
            }
        });
    }

    // ---- arithmetic --------------------------------------------------------------------------

    // The (last) type modifier of an instruction.
    PtxType typeOf(const ptx::Instruction& ins) {
        PtxType t;
        bool found = false;
        for (std::string_view m : ins.mods) {
            PtxType x;
            if (parseType(m, x)) t = x, found = true;
        }
        if (!found) fail(ins, "missing type");
        return t;
    }

    void expectOperands(const ptx::Instruction& ins, size_t n) {
        if (ins.operands.size() != n) fail(ins, "expected " + std::to_string(n) + " operands");
    }

    const ptx::Operand& dstReg(const ptx::Instruction& ins) {
        const ptx::Operand& d = ins.operands[0];
        if (d.kind != ptx::OperandKind::Register) fail(ins, "destination must be a register");
        return d;
    }

    uint32_t glsl(uint32_t type, GLSLstd450 inst, std::initializer_list<uint32_t> args) {
        return b_.extInst(glsl_, type, uint32_t(inst), args);
    }

    void emitMov(const ptx::Instruction& ins) {
        expectOperands(ins, 2);
        const PtxType t = typeOf(ins);
        const ptx::Operand& d = ins.operands[0];
        const ptx::Operand& s = ins.operands[1];
        if (d.kind == ptx::OperandKind::Register && s.kind == ptx::OperandKind::Vector) {
            // Pack: mov.b64 %rd, {%r1, %r2}
            const uint32_t n = uint32_t(s.elems.size());
            if (!t.isInt() || t.width % n) fail(ins, "invalid vector pack");
            const PtxType et{Base::Bit, t.width / n};
            const uint32_t w = compIntWidth(t), type = b_.typeInt(w);
            uint32_t acc = cInt(w, 0);
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t e = value(ins, s.elems[i], et);
                if (w == 64 && et.width <= 32) e = extendTo64(e, false);
                if (i) e = b_.op(spv::OpShiftLeftLogical, type, {e, cU32(i * et.width)});
                acc = b_.op(spv::OpBitwiseOr, type, {acc, e});
            }
            storeReg(ins, d.name, {Base::Bit, t.width}, acc);
            return;
        }
        if (d.kind == ptx::OperandKind::Vector) {
            // Unpack: mov.b64 {%r1, %r2}, %rd
            const uint32_t n = uint32_t(d.elems.size());
            if (!t.isInt() || t.width % n) fail(ins, "invalid vector unpack");
            const PtxType et{Base::Bit, t.width / n};
            const uint32_t w = compIntWidth(t), type = b_.typeInt(w);
            const uint32_t v = value(ins, s, {Base::Bit, t.width});
            for (uint32_t i = 0; i < n; ++i) {
                if (d.elems[i].kind == ptx::OperandKind::Sink) continue;
                if (d.elems[i].kind != ptx::OperandKind::Register) fail(ins, "invalid unpack destination");
                uint32_t e = i ? b_.op(spv::OpShiftRightLogical, type, {v, cU32(i * et.width)}) : v;
                if (w == 64 && et.width <= 32) e = truncTo32(e);
                storeReg(ins, d.elems[i].name, et, narrow(e, et.width, false));
            }
            return;
        }
        dstReg(ins);
        storeReg(ins, d.name, t, value(ins, s, t));
    }

    void emitBinary(const ptx::Instruction& ins) {
        expectOperands(ins, 3);
        const std::string_view op = ins.opcode;
        const PtxType t = typeOf(ins);
        const uint32_t type = compType(t, ins);
        const uint32_t a = value(ins, ins.operands[1], t);
        const uint32_t bv = value(ins, ins.operands[2], t);
        uint32_t r = 0;
        if (t.base == Base::Pred) {
            if (op == "and") r = b_.op(spv::OpLogicalAnd, type, {a, bv});
            else if (op == "or") r = b_.op(spv::OpLogicalOr, type, {a, bv});
            else if (op == "xor") r = b_.op(spv::OpLogicalNotEqual, type, {a, bv});
            else fail(ins, "invalid predicate operation");
        } else if (t.isFloat()) {
            if (op == "add") r = b_.op(spv::OpFAdd, type, {a, bv});
            else if (op == "sub") r = b_.op(spv::OpFSub, type, {a, bv});
            else if (op == "mul") r = b_.op(spv::OpFMul, type, {a, bv});
            else if (op == "div") r = b_.op(spv::OpFDiv, type, {a, bv});
            else if (op == "min") r = glsl(type, GLSLstd450NMin, {a, bv});
            else if (op == "max") r = glsl(type, GLSLstd450NMax, {a, bv});
            else fail(ins, "unsupported float operation");
            if ((op == "min" || op == "max") && ins.has(".NaN")) {  // NaN if either input is NaN
                const uint32_t anyNan = b_.op(spv::OpLogicalOr, tBool(), {b_.op(spv::OpIsNan, tBool(), {a}),
                                                                       b_.op(spv::OpIsNan, tBool(), {bv})});
                const uint64_t qnan = t.width == 64 ? 0x7FF8000000000000ull : 0x7FC00000ull;
                r = b_.op(spv::OpSelect, type, {anyNan, b_.constFloat(t.width, qnan), r});
            }
            if (ins.has(".sat")) r = saturate(r, t);
        } else {
            const bool s = t.isSigned();
            if (ins.has(".sat")) fail(ins, "saturating integer arithmetic is not supported yet");
            if (op == "add") r = b_.op(spv::OpIAdd, type, {a, bv});
            else if (op == "sub") r = b_.op(spv::OpISub, type, {a, bv});
            else if ((op == "div" || op == "rem") && t.width == 64) r = divRem64(op == "div", s, a, bv);
            else if (op == "div") r = b_.op(s ? spv::OpSDiv : spv::OpUDiv, type, {a, bv});
            else if (op == "rem") r = b_.op(s ? spv::OpSRem : spv::OpUMod, type, {a, bv});
            else if (op == "min") r = glsl(type, s ? GLSLstd450SMin : GLSLstd450UMin, {a, bv});
            else if (op == "max") r = glsl(type, s ? GLSLstd450SMax : GLSLstd450UMax, {a, bv});
            else if (op == "and") r = b_.op(spv::OpBitwiseAnd, type, {a, bv});
            else if (op == "or") r = b_.op(spv::OpBitwiseOr, type, {a, bv});
            else if (op == "xor") r = b_.op(spv::OpBitwiseXor, type, {a, bv});
            else fail(ins, "unsupported integer operation");
        }
        storeReg(ins, dstReg(ins).name, t, r);
    }

    // 64-bit division and remainder. GPUs without native 64-bit division emulate it at great
    // cost, yet the operands are usually small indices: when both fit in 31 bits (non-negative
    // for signed division) a 32-bit division gives the same result. DontFlatten: drivers would
    // otherwise turn the branch into a select while the 64-bit division is still one
    // instruction, then lower that into a ~400-instruction loop that always runs.
    uint32_t divRem64(bool isDiv, bool isSigned, uint32_t a, uint32_t b) {
        const uint32_t result = b_.localVariable(tU64());
        const uint32_t small = b_.op(spv::OpIEqual, tBool(), {b_.op(spv::OpShiftRightLogical, tU64(),
                                                                     {b_.op(spv::OpBitwiseOr, tU64(), {a, b}), cU32(31)}),
                                                               cU64(0)});
        selection(
            small,
            [&] {
                const uint32_t q = b_.op(isDiv ? spv::OpUDiv : spv::OpUMod, tU32(), {truncTo32(a), truncTo32(b)});
                b_.opVoid(spv::OpStore, {result, extendTo64(q, false)});
            },
            [&] {
                const spv::Op op = isDiv ? (isSigned ? spv::OpSDiv : spv::OpUDiv) : (isSigned ? spv::OpSRem : spv::OpUMod);
                b_.opVoid(spv::OpStore, {result, b_.op(op, tU64(), {a, b})});
            },
            spv::SelectionControlDontFlattenMask);
        return b_.op(spv::OpLoad, tU64(), {result});
    }

    uint32_t saturate(uint32_t v, const PtxType& t) {
        return glsl(tFloat(t.width), GLSLstd450NClamp,
                    {v, b_.constFloat(t.width, 0),
                     t.width == 64 ? b_.constFloat(64, std::bit_cast<uint64_t>(1.0)) : b_.constFloat(32, 0x3F800000)});
    }

    // mul / mad with .lo / .hi / .wide
    void emitMul(const ptx::Instruction& ins, bool isMad) {
        expectOperands(ins, isMad ? 4 : 3);
        const PtxType t = typeOf(ins);
        const uint32_t type = compType(t, ins);
        const uint32_t a = value(ins, ins.operands[1], t);
        const uint32_t bv = value(ins, ins.operands[2], t);
        if (t.isFloat()) {
            uint32_t r;
            if (isMad) r = fusedMulAdd(t, a, bv, value(ins, ins.operands[3], t), exactFma(ins));
            else r = b_.op(spv::OpFMul, type, {a, bv});
            if (ins.has(".sat")) r = saturate(r, t);
            storeReg(ins, dstReg(ins).name, t, r);
            return;
        }
        if (ins.has(".sat")) fail(ins, "saturating integer arithmetic is not supported yet");
        const bool s = t.isSigned();
        PtxType rt = t;
        uint32_t r;
        if (ins.has(".wide")) {
            if (t.width == 64) fail(ins, "mul.wide.64 is not supported");
            rt.width = t.width * 2;
            const uint32_t wt = compType(rt, ins);
            const uint32_t wa = rt.width == 64 ? extendTo64(a, s) : a;
            const uint32_t wb = rt.width == 64 ? extendTo64(bv, s) : bv;
            r = b_.op(spv::OpIMul, wt, {wa, wb});
        } else if (ins.has(".hi")) {
            if (t.width < 32) {
                const uint32_t p = b_.op(spv::OpIMul, type, {a, bv});
                r = b_.op(s ? spv::OpShiftRightArithmetic : spv::OpShiftRightLogical, type, {p, cU32(t.width)});
            } else {
                const uint32_t pair = b_.op(s ? spv::OpSMulExtended : spv::OpUMulExtended, structPair(type), {a, bv});
                r = b_.op(spv::OpCompositeExtract, type, {pair, 1});
            }
        } else {
            r = b_.op(spv::OpIMul, type, {a, bv});
        }
        if (isMad && (ins.has(".cc") || ins.opcode == "madc")) {
            // mad{.lo,.hi}.cc / madc{.lo,.hi}{.cc}: the addition takes and/or sets the carry flag
            if (ins.has(".wide") || (t.width != 32 && t.width != 64)) fail(ins, "32/64-bit mad.lo/hi expected");
            r = addWithCarry(t, r, value(ins, ins.operands[3], t), false, ins.opcode == "madc", ins.has(".cc"));
        } else if (isMad) {
            r = b_.op(spv::OpIAdd, compType(rt, ins), {r, value(ins, ins.operands[3], rt)});
        }
        storeReg(ins, dstReg(ins).name, rt, r);
    }

    // a * b + c with a single rounding. Vulkan's Fma may be evaluated as a multiply and an add
    // (RADV on GFX9, llvmpipe), which breaks the error-free transformations CUDA's math library
    // is built on. The driver probes this per device (Target::fusedFma); where Fma is unfused,
    // f32 fma is emulated from f32 operations (fmaEmulated), or with VULKANIZER_FMA_F64=1 computed in
    // f64, where the product is exact, and rounded once to f32 (up to a rare double rounding).
    // f64 fma is emulated the same way where the driver's f64 Fma is unfused (llvmpipe).
    // VULKANIZER_FAST_FMA=1 uses the driver's Fma anyway: fastest, least precise.
    // `exact` false: the driver's Fma suffices (exactFma()).
    uint32_t fusedMulAdd(const PtxType& t, uint32_t a, uint32_t b, uint32_t c, bool exact) {
        if (t.width == 64 && !t_.fusedFma64) return fmaEmulated(64, a, b, c);
        if (t.width != 32 || t_.fusedFma || !exact) return glsl(tFloat(t.width), GLSLstd450Fma, {a, b, c});
        if (t_.fmaInF64 && t_.float64) {
            const uint32_t d = tFloat(64);
            auto wide = [&](uint32_t x) { return b_.op(spv::OpFConvert, d, {x}); };
            const uint32_t r = glsl(d, GLSLstd450Fma, {wide(a), wide(b), wide(c)});
            return b_.op(spv::OpFConvert, tFloat(32), {r});
        }
        return fmaEmulated(32, a, b, c);
    }

    // Whether an f32 fma/mad needs the single rounding. Not with .ftz: nvcc emits it for code built
    // with -ftz=true (--use_fast_math, as ggml-cuda is), which gave up IEEE results anyway, and the
    // emulation costs some 30 instructions (half of llama.cpp's matrix-vector kernels on RADV GFX9).
    // The error-free transformations of math libraries live in IEEE code (fma.rn.f32).
    // VULKANIZER_EXACT_FMA=1 emulates the .ftz ones too.
    bool exactFma(const ptx::Instruction& ins) const { return !ins.has(".ftz") || t_.exactFtzFma; }

    // fma from f32 (f64) operations (all NoContraction): the product's exact error by Veltkamp
    // splitting (Dekker's TwoProduct), the sum's by TwoSum, then one final rounding of the
    // small terms. Exact where error-free transformations need it (c = -RN(a b): the result is
    // the product's error), faithful otherwise; outside the safe range (|a|, |b| >= 2^100,
    // 2^500 for f64, infinities, NaN) it degrades to a * b + c.
    uint32_t fmaEmulated(uint32_t width, uint32_t a, uint32_t b, uint32_t c) {
        const uint32_t f = tFloat(width);
        auto mul = [&](uint32_t x, uint32_t y) { return b_.op(spv::OpFMul, f, {x, y}); };
        auto add = [&](uint32_t x, uint32_t y) { return b_.op(spv::OpFAdd, f, {x, y}); };
        auto sub = [&](uint32_t x, uint32_t y) { return b_.op(spv::OpFSub, f, {x, y}); };
        const uint32_t splitter = width == 64 ? b_.constFloat(64, 0x41A0000002000000ull)  // 2^27 + 1
                                              : b_.constFloat(32, 0x45801000);           // 2^12 + 1
        auto split = [&](uint32_t x, uint32_t& hi, uint32_t& lo) {
            const uint32_t s = mul(splitter, x);
            hi = sub(s, sub(s, x));
            lo = sub(x, hi);
        };
        uint32_t ah, al, bh, bl;
        split(a, ah, al);
        split(b, bh, bl);
        const uint32_t p = mul(a, b);
        const uint32_t e = add(add(add(sub(mul(ah, bh), p), mul(ah, bl)), mul(al, bh)), mul(al, bl));
        const uint32_t s = add(p, c);  // TwoSum(p, c)
        const uint32_t bb = sub(s, p);
        const uint32_t err = add(sub(p, sub(s, bb)), sub(c, bb));
        const uint32_t r = add(s, add(e, err));
        // safe range: |a|, |b| < 2^100 and a finite result (else the plain a * b + c)
        const uint32_t limit = width == 64 ? b_.constFloat(64, 0x5F30000000000000ull)  // 2^500
                                           : b_.constFloat(32, 0x71800000);           // 2^100
        auto below = [&](uint32_t x) {
            return b_.op(spv::OpFOrdLessThan, tBool(), {glsl(f, GLSLstd450FAbs, {x}), limit});
        };
        const uint32_t finite = b_.op(spv::OpLogicalNot, tBool(),
                                      {b_.op(spv::OpLogicalOr, tBool(), {b_.op(spv::OpIsInf, tBool(), {s}),
                                                                         b_.op(spv::OpIsNan, tBool(), {s})})});
        const uint32_t safe = b_.op(spv::OpLogicalAnd, tBool(),
                                    {b_.op(spv::OpLogicalAnd, tBool(), {below(a), below(b)}), finite});
        return b_.op(spv::OpSelect, f, {safe, r, s});
    }

    void emitFma(const ptx::Instruction& ins) {
        expectOperands(ins, 4);
        const PtxType t = typeOf(ins);
        if (!t.isFloat()) fail(ins, "fma needs a float type");
        compType(t, ins);  // rejects f16
        uint32_t r = fusedMulAdd(t, value(ins, ins.operands[1], t), value(ins, ins.operands[2], t),
                                 value(ins, ins.operands[3], t), exactFma(ins));
        if (ins.has(".sat")) r = saturate(r, t);
        storeReg(ins, dstReg(ins).name, t, r);
    }

    void emitUnary(const ptx::Instruction& ins) {
        expectOperands(ins, 2);
        const std::string_view op = ins.opcode;
        const PtxType t = typeOf(ins);
        const uint32_t type = compType(t, ins);
        const uint32_t a = value(ins, ins.operands[1], t);
        uint32_t r = 0;
        if (op == "not") r = b_.op(t.base == Base::Pred ? spv::OpLogicalNot : spv::OpNot, type, {a});
        else if (op == "neg") r = b_.op(t.isFloat() ? spv::OpFNegate : spv::OpSNegate, type, {a});
        else if (op == "abs") r = glsl(type, t.isFloat() ? GLSLstd450FAbs : GLSLstd450SAbs, {a});
        storeReg(ins, dstReg(ins).name, t, r);
    }

    void emitShift(const ptx::Instruction& ins) {
        expectOperands(ins, 3);
        const PtxType t = typeOf(ins);
        if (!t.isInt()) fail(ins, "shift needs an integer type");
        const uint32_t type = compType(t, ins);
        const uint32_t a = value(ins, ins.operands[1], t);
        const uint32_t amt = value(ins, ins.operands[2], {Base::Unsigned, 32});
        const uint32_t w = t.width;
        // PTX clamps the shift amount; SPIR-V leaves shifts >= width undefined.
        const uint32_t clamped = glsl(tU32(), GLSLstd450UMin, {amt, cU32(w - 1)});
        uint32_t r;
        if (ins.opcode == "shr" && t.isSigned()) {
            r = b_.op(spv::OpShiftRightArithmetic, type, {a, clamped});
        } else {
            const spv::Op op = ins.opcode == "shl" ? spv::OpShiftLeftLogical : spv::OpShiftRightLogical;
            const uint32_t shifted = b_.op(op, type, {a, clamped});
            const uint32_t big = b_.op(spv::OpUGreaterThan, tBool(), {amt, cU32(w - 1)});
            r = b_.op(spv::OpSelect, type, {big, cInt(compIntWidth(t), 0), shifted});
        }
        storeReg(ins, dstReg(ins).name, t, r);
    }

    uint32_t compare(const ptx::Instruction& ins, std::string_view cmp, const PtxType& t, uint32_t a, uint32_t bv) {
        const uint32_t bt = tBool();
        if (t.isFloat()) {
            struct { std::string_view name; spv::Op op; } table[] = {
                {".eq", spv::OpFOrdEqual},          {".ne", spv::OpFOrdNotEqual},
                {".lt", spv::OpFOrdLessThan},       {".le", spv::OpFOrdLessThanEqual},
                {".gt", spv::OpFOrdGreaterThan},    {".ge", spv::OpFOrdGreaterThanEqual},
                {".equ", spv::OpFUnordEqual},       {".neu", spv::OpFUnordNotEqual},
                {".ltu", spv::OpFUnordLessThan},    {".leu", spv::OpFUnordLessThanEqual},
                {".gtu", spv::OpFUnordGreaterThan}, {".geu", spv::OpFUnordGreaterThanEqual},
            };
            for (auto& e : table)
                if (e.name == cmp) return b_.op(e.op, bt, {a, bv});
            const uint32_t nanA = b_.op(spv::OpIsNan, bt, {a}), nanB = b_.op(spv::OpIsNan, bt, {bv});
            const uint32_t anyNan = b_.op(spv::OpLogicalOr, bt, {nanA, nanB});
            if (cmp == ".nan") return anyNan;
            if (cmp == ".num") return b_.op(spv::OpLogicalNot, bt, {anyNan});
        } else if (t.base == Base::Pred) {
            if (cmp == ".eq") return b_.op(spv::OpLogicalEqual, bt, {a, bv});
            if (cmp == ".ne") return b_.op(spv::OpLogicalNotEqual, bt, {a, bv});
        } else {
            const bool s = t.isSigned();
            struct { std::string_view name; spv::Op sop, uop; } table[] = {
                {".eq", spv::OpIEqual, spv::OpIEqual},
                {".ne", spv::OpINotEqual, spv::OpINotEqual},
                {".lt", spv::OpSLessThan, spv::OpULessThan},
                {".le", spv::OpSLessThanEqual, spv::OpULessThanEqual},
                {".gt", spv::OpSGreaterThan, spv::OpUGreaterThan},
                {".ge", spv::OpSGreaterThanEqual, spv::OpUGreaterThanEqual},
                {".lo", spv::OpULessThan, spv::OpULessThan},
                {".ls", spv::OpULessThanEqual, spv::OpULessThanEqual},
                {".hi", spv::OpUGreaterThan, spv::OpUGreaterThan},
                {".hs", spv::OpUGreaterThanEqual, spv::OpUGreaterThanEqual},
            };
            for (auto& e : table)
                if (e.name == cmp) return b_.op(s ? e.sop : e.uop, bt, {a, bv});
        }
        fail(ins, "unsupported comparison " + std::string(cmp));
    }

    void emitSetp(const ptx::Instruction& ins) {
        if (ins.operands.size() != 3 && ins.operands.size() != 4) fail(ins, "setp needs 3 or 4 operands");
        if (ins.mods.size() < 2) fail(ins, "setp needs a comparison and a type");
        const std::string_view cmp = ins.mods[0];
        std::string_view boolOp;
        if (ins.mods.size() >= 3) boolOp = ins.mods[1];
        const PtxType t = typeOf(ins);
        const uint32_t bt = tBool();
        const ptx::Operand& d = ins.operands[0];
        if (d.kind != ptx::OperandKind::Register) fail(ins, "setp destination must be a predicate");
        uint32_t p = compare(ins, cmp, t, value(ins, ins.operands[1], t), value(ins, ins.operands[2], t));
        uint32_t q = d.pairName.empty() ? 0 : b_.op(spv::OpLogicalNot, bt, {p});
        if (ins.operands.size() == 4) {
            const uint32_t c = value(ins, ins.operands[3], {Base::Pred, 1});
            spv::Op op;
            if (boolOp == ".and") op = spv::OpLogicalAnd;
            else if (boolOp == ".or") op = spv::OpLogicalOr;
            else if (boolOp == ".xor") op = spv::OpLogicalNotEqual;
            else fail(ins, "setp with 4 operands needs .and/.or/.xor");
            p = b_.op(op, bt, {p, c});
            if (q) q = b_.op(op, bt, {q, c});
        }
        storeReg(ins, d.name, {Base::Pred, 1}, p);
        if (q) storeReg(ins, d.pairName, {Base::Pred, 1}, q);
    }

    void emitSelp(const ptx::Instruction& ins) {
        expectOperands(ins, 4);
        const PtxType t = typeOf(ins);
        const uint32_t r = b_.op(spv::OpSelect, compType(t, ins),
                                 {value(ins, ins.operands[3], {Base::Pred, 1}), value(ins, ins.operands[1], t),
                                  value(ins, ins.operands[2], t)});
        storeReg(ins, dstReg(ins).name, t, r);
    }

    // Float -> integer with PTX semantics: saturate to the destination range, NaN -> 0.
    uint32_t floatToInt(uint32_t v, const PtxType& from, const PtxType& to) {
        const uint32_t ft = tFloat(from.width);
        const uint32_t w = to.width;
        auto fconst = [&](double d) {
            if (from.width == 64) return b_.constFloat(64, std::bit_cast<uint64_t>(d));
            return b_.constFloat(32, std::bit_cast<uint32_t>(float(d)));
        };
        const double lo = to.isSigned() ? -std::ldexp(1.0, int(w) - 1) : 0.0;
        const double limit = std::ldexp(1.0, to.isSigned() ? int(w) - 1 : int(w));  // first value out of range
        // Largest value below the limit that the source type can represent.
        const double below = from.width == 64 ? std::nextafter(limit, 0.0) : double(std::nextafter(float(limit), 0.0f));
        const uint32_t it = w == 64 ? tU64() : tU32();
        const uint64_t maxInt = to.isSigned() ? (uint64_t(1) << (w - 1)) - 1 : (w == 64 ? ~uint64_t(0) : (uint64_t(1) << w) - 1);
        const uint32_t c = glsl(ft, GLSLstd450FClamp, {v, fconst(lo), fconst(below)});
        uint32_t r = b_.op(to.isSigned() ? spv::OpConvertFToS : spv::OpConvertFToU, it, {c});
        const uint32_t tooBig = b_.op(spv::OpFOrdGreaterThanEqual, tBool(), {v, fconst(limit)});
        r = b_.op(spv::OpSelect, it, {tooBig, cInt(w == 64 ? 64 : 32, maxInt), r});
        const uint32_t isNan = b_.op(spv::OpIsNan, tBool(), {v});
        return b_.op(spv::OpSelect, it, {isNan, cInt(w == 64 ? 64 : 32, 0), r});
    }

    // f32 -> f16 bits with round-to-nearest-even. packHalf2x16 would be shorter, but its
    // rounding is implementation-defined (RADV truncates).
    uint32_t floatToHalf(uint32_t f) {
        const uint32_t u32 = tU32(), bt = tBool();
        auto op = [&](spv::Op o, uint32_t x, uint32_t y) { return b_.op(o, o >= spv::OpIEqual && o <= spv::OpSLessThanEqual ? bt : u32, {x, y}); };
        auto sel = [&](uint32_t c, uint32_t x, uint32_t y) { return b_.op(spv::OpSelect, u32, {c, x, y}); };
        // Rounds x >> s to nearest even (s in 1..31).
        auto shiftRound = [&](uint32_t x, uint32_t s) {
            const uint32_t r = op(spv::OpShiftRightLogical, x, s);
            const uint32_t rem = op(spv::OpBitwiseAnd, x, op(spv::OpISub, op(spv::OpShiftLeftLogical, cU32(1), s), cU32(1)));
            const uint32_t half = op(spv::OpShiftLeftLogical, cU32(1), op(spv::OpISub, s, cU32(1)));
            const uint32_t odd = op(spv::OpINotEqual, op(spv::OpBitwiseAnd, r, cU32(1)), cU32(0));
            const uint32_t up = b_.op(spv::OpLogicalOr, bt, {op(spv::OpUGreaterThan, rem, half),
                                                           b_.op(spv::OpLogicalAnd, bt, {op(spv::OpIEqual, rem, half), odd})});
            return op(spv::OpIAdd, r, sel(up, cU32(1), cU32(0)));
        };
        const uint32_t u = b_.op(spv::OpBitcast, u32, {f});
        const uint32_t sign = op(spv::OpBitwiseAnd, op(spv::OpShiftRightLogical, u, cU32(16)), cU32(0x8000));
        const uint32_t a = op(spv::OpBitwiseAnd, u, cU32(0x7FFFFFFF));
        const uint32_t e = op(spv::OpShiftRightLogical, a, cU32(23));
        // normal half: rebias the exponent and round away 13 mantissa bits (carries may reach inf)
        const uint32_t normal = shiftRound(op(spv::OpISub, a, cU32(112u << 23)), cU32(13));
        // subnormal half: value / 2^-24 = mantissa >> (126 - e)
        const uint32_t mant = op(spv::OpBitwiseOr, op(spv::OpBitwiseAnd, a, cU32(0x7FFFFF)), cU32(0x800000));
        const uint32_t sub = shiftRound(mant, glsl(u32, GLSLstd450UMin, {op(spv::OpISub, cU32(126), e), cU32(31)}));
        uint32_t h = sel(op(spv::OpULessThan, a, cU32(0x38800000)), sel(op(spv::OpULessThan, e, cU32(102)), cU32(0), sub), normal);
        h = sel(op(spv::OpUGreaterThanEqual, a, cU32(0x477FF000)), cU32(0x7C00), h);  // overflow -> inf
        h = sel(op(spv::OpUGreaterThan, a, cU32(0x7F800000)), cU32(0x7E00), h);       // NaN
        return op(spv::OpBitwiseOr, h, sign);
    }

    // f32 -> f16 or bf16 bits (in a u32), rounded to nearest even; NaN stays a (quiet) NaN.
    uint32_t toHalfBits(uint32_t f, bool brain) {
        if (!brain) return floatToHalf(f);
        const uint32_t u = b_.op(spv::OpBitcast, tU32(), {f});
        const uint32_t lsb = b_.op(spv::OpBitwiseAnd, tU32(), {b_.op(spv::OpShiftRightLogical, tU32(), {u, cU32(16)}), cU32(1)});
        const uint32_t rounded = b_.op(spv::OpShiftRightLogical, tU32(),
                                       {b_.op(spv::OpIAdd, tU32(), {u, b_.op(spv::OpIAdd, tU32(), {cU32(0x7FFF), lsb})}), cU32(16)});
        return b_.op(spv::OpSelect, tU32(), {b_.op(spv::OpIsNan, tBool(), {f}), cU32(0x7FC0), rounded});
    }

    // cvt.rn{.relu}.{f16x2,bf16x2}.f32 d, a, b: a converted into the upper half of d, b into the lower.
    void emitCvtPacked(const ptx::Instruction& ins, bool brain) {
        expectOperands(ins, 3);
        if (!ins.has(".rn")) fail(ins, "packed cvt needs .rn");
        const PtxType f32t{Base::Float, 32};
        auto half = [&](const ptx::Operand& o) {
            uint32_t f = value(ins, o, f32t);
            if (ins.has(".relu"))  // negative -> 0; NaN stays NaN
                f = b_.op(spv::OpSelect, tFloat(32),
                          {b_.op(spv::OpFOrdLessThan, tBool(), {f, b_.constFloat(32, 0)}), b_.constFloat(32, 0), f});
            return toHalfBits(f, brain);
        };
        const uint32_t hi = half(ins.operands[1]), lo = half(ins.operands[2]);
        storeReg(ins, dstReg(ins).name, {Base::Bit, 32},
                 b_.op(spv::OpBitwiseOr, tU32(), {b_.op(spv::OpShiftLeftLogical, tU32(), {hi, cU32(16)}), lo}));
    }

    template <class Round>
    void emitCvtHalf(const ptx::Instruction& ins, PtxType to, std::string_view toHalf, PtxType from,
                     std::string_view fromHalf, Round roundInt) {
        const uint32_t f32 = tFloat(32);
        const PtxType bits16{Base::Bit, 16};
        uint32_t f;
        if (!fromHalf.empty()) {
            const uint32_t bits = value(ins, ins.operands[1], bits16);
            if (fromHalf == ".bf16")
                f = b_.op(spv::OpBitcast, f32, {b_.op(spv::OpShiftLeftLogical, tU32(), {bits, cU32(16)})});
            else
                f = b_.op(spv::OpCompositeExtract, f32, {glsl(b_.typeVector(f32, 2), GLSLstd450UnpackHalf2x16, {bits}), 0});
        } else {
            const uint32_t v = value(ins, ins.operands[1], from);
            if (from.isFloat()) f = from.width == 32 ? v : b_.op(spv::OpFConvert, f32, {v});
            else f = b_.op(from.isSigned() ? spv::OpConvertSToF : spv::OpConvertUToF, f32, {v});
        }
        if (!toHalf.empty()) {
            storeReg(ins, dstReg(ins).name, bits16, toHalfBits(f, toHalf == ".bf16"));
            return;
        }
        uint32_t r;
        if (to.isFloat()) r = to.width == 32 ? roundInt(f, f32) : roundInt(b_.op(spv::OpFConvert, tFloat(64), {f}), tFloat(64));
        else r = floatToInt(roundInt(f, f32), {Base::Float, 32}, to);
        if (to.isInt() && to.width < 32) r = narrow(r, to.width, to.isSigned());
        storeReg(ins, dstReg(ins).name, to, r);
    }

    void emitCvt(const ptx::Instruction& ins) {
        if (ins.has(".f16x2") || ins.has(".bf16x2")) return emitCvtPacked(ins, ins.has(".bf16x2"));
        expectOperands(ins, 2);
        std::vector<PtxType> types;
        std::string_view rnd;
        for (std::string_view m : ins.mods) {
            PtxType x;
            if (parseType(m, x)) types.push_back(x);
            else if (m == ".rn" || m == ".rz" || m == ".rm" || m == ".rp" || m == ".rni" || m == ".rzi" ||
                     m == ".rmi" || m == ".rpi")
                rnd = m;
        }
        auto roundInt = [&](uint32_t x, uint32_t ft) {
            if (rnd == ".rni") return glsl(ft, GLSLstd450RoundEven, {x});
            if (rnd == ".rzi") return glsl(ft, GLSLstd450Trunc, {x});
            if (rnd == ".rmi") return glsl(ft, GLSLstd450Floor, {x});
            if (rnd == ".rpi") return glsl(ft, GLSLstd450Ceil, {x});
            return x;
        };
        // Half types (f16, bf16) are held as 16 bits in an integer and converted through f32.
        std::vector<std::string_view> halves;  // per type position: "", ".f16", ".bf16"
        types.clear();
        for (std::string_view m : ins.mods) {
            PtxType x;
            if (m == ".bf16") {
                halves.push_back(m);
                types.push_back({Base::Float, 16});
            } else if (parseType(m, x)) {
                halves.push_back(x.isFloat() && x.width == 16 ? ".f16" : "");
                types.push_back(x);
            }
        }
        if (types.size() != 2) fail(ins, "cvt needs a destination and a source type");
        if (!halves[0].empty() || !halves[1].empty()) {
            emitCvtHalf(ins, types[0], halves[0], types[1], halves[1], roundInt);
            return;
        }
        const PtxType to = types[0], from = types[1];
        uint32_t v = value(ins, ins.operands[1], from);

        uint32_t r;
        if (from.isFloat() && to.isFloat()) {
            r = from.width == to.width ? v : b_.op(spv::OpFConvert, tFloat(to.width), {v});
            r = roundInt(r, tFloat(to.width));
            if (ins.has(".sat")) r = saturate(r, to);
        } else if (from.isFloat()) {
            if (!to.isInt()) fail(ins, "invalid conversion");
            r = floatToInt(roundInt(v, tFloat(from.width)), from, to);
            if (to.width < 32) r = narrow(r, to.width, to.isSigned());
        } else if (to.isFloat()) {
            r = b_.op(from.isSigned() ? spv::OpConvertSToF : spv::OpConvertUToF, tFloat(to.width), {v});
            if (ins.has(".sat")) r = saturate(r, to);
        } else {
            if (ins.has(".sat")) fail(ins, "saturating integer conversion is not supported yet");
            // Integer -> integer: extend according to the source type, then truncate.
            if (to.width == 64 && from.width <= 32) r = extendTo64(v, from.isSigned());
            else if (to.width <= 32 && from.width == 64) r = truncTo32(v);
            else r = v;
            if (to.width < 32) r = narrow(r, to.width, to.isSigned());
        }
        storeReg(ins, dstReg(ins).name, to, r);
    }

    // ---- math and bit manipulation -----------------------------------------------------------

    // sqrt, rsqrt, rcp, ex2, lg2, sin, cos, tanh
    void emitMathUnary(const ptx::Instruction& ins) {
        expectOperands(ins, 2);
        const std::string_view op = ins.opcode;
        const PtxType t = typeOf(ins);
        if (!t.isFloat()) fail(ins, "float type expected");
        const uint32_t type = compType(t, ins);
        const uint32_t a = value(ins, ins.operands[1], t);
        const uint32_t one = t.width == 64 ? b_.constFloat(64, std::bit_cast<uint64_t>(1.0)) : b_.constFloat(32, 0x3F800000);
        uint32_t r;
        if (op == "sqrt") r = glsl(type, GLSLstd450Sqrt, {a});
        else if (op == "rsqrt") r = glsl(type, GLSLstd450InverseSqrt, {a});
        else if (op == "rcp") r = b_.op(spv::OpFDiv, type, {one, a});
        else {
            if (t.width != 32) fail(ins, "only f32 is supported");
            if (op == "ex2") r = glsl(type, GLSLstd450Exp2, {a});
            else if (op == "lg2") r = glsl(type, GLSLstd450Log2, {a});
            else if (op == "sin") r = glsl(type, GLSLstd450Sin, {a});
            else if (op == "cos") r = glsl(type, GLSLstd450Cos, {a});
            else r = glsl(type, GLSLstd450Tanh, {a});
        }
        storeReg(ins, dstReg(ins).name, t, r);
    }

    // copysign.f32 d, a, b: magnitude of b, sign of a
    void emitCopysign(const ptx::Instruction& ins) {
        expectOperands(ins, 3);
        const PtxType t = typeOf(ins);
        const PtxType bt{Base::Bit, t.width};
        const uint64_t sign = uint64_t(1) << (t.width - 1);
        const uint32_t it = b_.typeInt(t.width);
        const uint32_t a = value(ins, ins.operands[1], bt), b = value(ins, ins.operands[2], bt);
        const uint32_t r = b_.op(spv::OpBitwiseOr, it, {b_.op(spv::OpBitwiseAnd, it, {b, cInt(t.width, sign - 1)}),
                                                        b_.op(spv::OpBitwiseAnd, it, {a, cInt(t.width, sign)})});
        storeReg(ins, dstReg(ins).name, bt, r);
    }

    // testp.{finite,infinite,number,notanumber,normal,subnormal}.f32 p, a
    void emitTestp(const ptx::Instruction& ins) {
        expectOperands(ins, 2);
        const PtxType t = typeOf(ins);
        const uint32_t a = value(ins, ins.operands[1], t), bt = tBool();
        const uint32_t nan = b_.op(spv::OpIsNan, bt, {a}), inf = b_.op(spv::OpIsInf, bt, {a});
        auto lnot = [&](uint32_t x) { return b_.op(spv::OpLogicalNot, bt, {x}); };
        const uint32_t finite = lnot(b_.op(spv::OpLogicalOr, bt, {nan, inf}));
        const uint32_t fabs = glsl(compType(t, ins), GLSLstd450FAbs, {a});
        const uint64_t minNormal = t.width == 64 ? 0x0010000000000000ull : 0x00800000ull;
        const uint32_t small = b_.op(spv::OpFOrdLessThan, bt, {fabs, b_.constFloat(t.width, minNormal)});
        const uint32_t zero = b_.op(spv::OpFOrdEqual, bt, {fabs, b_.constFloat(t.width, 0)});
        uint32_t r;
        if (ins.has(".finite")) r = finite;
        else if (ins.has(".infinite")) r = inf;
        else if (ins.has(".number")) r = lnot(nan);
        else if (ins.has(".notanumber")) r = nan;
        else if (ins.has(".normal")) r = b_.op(spv::OpLogicalAnd, bt, {finite, lnot(small)});
        else if (ins.has(".subnormal")) r = b_.op(spv::OpLogicalAnd, bt, {small, lnot(zero)});
        else fail(ins, "unsupported testp class");
        storeReg(ins, dstReg(ins).name, {Base::Pred, 1}, r);
    }

    // Shifts with PTX semantics for any amount (SPIR-V leaves amount >= width undefined).
    uint32_t shl(uint32_t v, uint32_t amt32, uint32_t w) {
        const uint32_t type = b_.typeInt(w);
        const uint32_t s = b_.op(spv::OpShiftLeftLogical, type, {v, glsl(tU32(), GLSLstd450UMin, {amt32, cU32(w - 1)})});
        return b_.op(spv::OpSelect, type, {b_.op(spv::OpUGreaterThanEqual, tBool(), {amt32, cU32(w)}), cInt(w, 0), s});
    }
    uint32_t shr(uint32_t v, uint32_t amt32, uint32_t w, bool arithmetic) {
        const uint32_t type = b_.typeInt(w);
        const uint32_t clamped = glsl(tU32(), GLSLstd450UMin, {amt32, cU32(w - 1)});
        if (arithmetic) return b_.op(spv::OpShiftRightArithmetic, type, {v, clamped});
        const uint32_t s = b_.op(spv::OpShiftRightLogical, type, {v, clamped});
        return b_.op(spv::OpSelect, type, {b_.op(spv::OpUGreaterThanEqual, tBool(), {amt32, cU32(w)}), cInt(w, 0), s});
    }
    // (1 << len) - 1, all ones for len >= w
    uint32_t lowMask(uint32_t len32, uint32_t w) {
        const uint32_t type = b_.typeInt(w);
        const uint32_t m = b_.op(spv::OpISub, type, {shl(cInt(w, 1), len32, w), cInt(w, 1)});
        return b_.op(spv::OpSelect, type,
                     {b_.op(spv::OpUGreaterThanEqual, tBool(), {len32, cU32(w)}), cInt(w, ~uint64_t(0)), m});
    }

    uint32_t bitCount32(uint32_t v) { return b_.op(spv::OpBitCount, tU32(), {v}); }
    uint32_t clz32(uint32_t v) {
        const uint32_t msb = glsl(tU32(), GLSLstd450FindUMsb, {v});
        return b_.op(spv::OpSelect, tU32(), {b_.op(spv::OpIEqual, tBool(), {v, cU32(0)}), cU32(32),
                                            b_.op(spv::OpISub, tU32(), {cU32(31), msb})});
    }
    uint32_t hiWord(uint32_t v64) { return truncTo32(b_.op(spv::OpShiftRightLogical, tU64(), {v64, cU32(32)})); }

    // popc, clz, brev
    void emitBitScan(const ptx::Instruction& ins) {
        expectOperands(ins, 2);
        const std::string_view op = ins.opcode;
        const PtxType t = typeOf(ins);
        if (!t.isInt() || (t.width != 32 && t.width != 64)) fail(ins, ".b32 or .b64 expected");
        const uint32_t a = value(ins, ins.operands[1], {Base::Bit, t.width});
        if (op == "brev") {
            uint32_t r;
            if (t.width == 32) {
                r = b_.op(spv::OpBitReverse, tU32(), {a});
            } else {
                const uint32_t lo = b_.op(spv::OpBitReverse, tU32(), {truncTo32(a)});
                const uint32_t hi = b_.op(spv::OpBitReverse, tU32(), {hiWord(a)});
                r = b_.op(spv::OpBitwiseOr, tU64(), {b_.op(spv::OpShiftLeftLogical, tU64(), {extendTo64(lo, false), cU32(32)}),
                                                     extendTo64(hi, false)});
            }
            storeReg(ins, dstReg(ins).name, {Base::Bit, t.width}, r);
            return;
        }
        uint32_t r;
        if (op == "popc") {
            r = t.width == 32 ? bitCount32(a)
                              : b_.op(spv::OpIAdd, tU32(), {bitCount32(truncTo32(a)), bitCount32(hiWord(a))});
        } else if (t.width == 32) {
            r = clz32(a);
        } else {
            const uint32_t hi = hiWord(a);
            r = b_.op(spv::OpSelect, tU32(), {b_.op(spv::OpIEqual, tBool(), {hi, cU32(0)}),
                                             b_.op(spv::OpIAdd, tU32(), {cU32(32), clz32(truncTo32(a))}), clz32(hi)});
        }
        storeReg(ins, dstReg(ins).name, {Base::Unsigned, 32}, r);
    }

    // bfind{.shiftamt}.{u,s}{32,64} d, a: bit position of the most significant non-sign bit
    // (or the left shift that normalizes it), 0xFFFFFFFF if there is none
    void emitBfind(const ptx::Instruction& ins) {
        expectOperands(ins, 2);
        const PtxType t = typeOf(ins);
        if (!t.isInt() || (t.width != 32 && t.width != 64)) fail(ins, "32/64-bit integer expected");
        const bool s = t.isSigned();
        const uint32_t a = value(ins, ins.operands[1], t);
        const GLSLstd450 find = s ? GLSLstd450FindSMsb : GLSLstd450FindUMsb;
        uint32_t pos;
        if (t.width == 32) {
            pos = glsl(tU32(), find, {a});
        } else {
            // Signed: look for the first bit that differs from the sign, i.e. the msb of a ^ (a >> 63).
            const uint32_t v = s ? b_.op(spv::OpBitwiseXor, tU64(), {a, b_.op(spv::OpShiftRightArithmetic, tU64(), {a, cU32(63)})}) : a;
            const uint32_t hi = hiWord(v), lo = truncTo32(v);
            const uint32_t hiPos = glsl(tU32(), GLSLstd450FindUMsb, {hi});
            pos = b_.op(spv::OpSelect, tU32(), {b_.op(spv::OpINotEqual, tBool(), {hi, cU32(0)}),
                                               b_.op(spv::OpIAdd, tU32(), {hiPos, cU32(32)}),
                                               glsl(tU32(), GLSLstd450FindUMsb, {lo})});
        }
        const uint32_t none = b_.op(spv::OpIEqual, tBool(), {pos, cU32(0xFFFFFFFF)});
        if (ins.has(".shiftamt"))
            pos = b_.op(spv::OpSelect, tU32(), {none, pos, b_.op(spv::OpISub, tU32(), {cU32(t.width - 1), pos})});
        storeReg(ins, dstReg(ins).name, {Base::Unsigned, 32}, pos);
    }

    // bfe.{u,s}{32,64} d, a, pos, len
    void emitBfe(const ptx::Instruction& ins) {
        expectOperands(ins, 4);
        const PtxType t = typeOf(ins);
        const uint32_t w = compIntWidth(t), type = b_.typeInt(w);
        const uint32_t a = value(ins, ins.operands[1], t);
        const uint32_t pos = b_.op(spv::OpBitwiseAnd, tU32(), {value(ins, ins.operands[2], {Base::Unsigned, 32}), cU32(0xFF)});
        const uint32_t len = b_.op(spv::OpBitwiseAnd, tU32(), {value(ins, ins.operands[3], {Base::Unsigned, 32}), cU32(0xFF)});
        uint32_t r;
        if (t.isSigned()) {
            const uint32_t x = shr(a, pos, w, true);  // bits beyond the msb become the sign
            const uint32_t up = b_.op(spv::OpISub, tU32(), {cU32(w), glsl(tU32(), GLSLstd450UMin, {len, cU32(w)})});
            const uint32_t ext = shr(shl(x, up, w), up, w, true);
            r = b_.op(spv::OpSelect, type, {b_.op(spv::OpUGreaterThanEqual, tBool(), {len, cU32(w)}), x, ext});
        } else {
            r = b_.op(spv::OpBitwiseAnd, type, {shr(a, pos, w, false), lowMask(len, w)});
        }
        r = b_.op(spv::OpSelect, type, {b_.op(spv::OpIEqual, tBool(), {len, cU32(0)}), cInt(w, 0), r});
        storeReg(ins, dstReg(ins).name, t, r);
    }

    // bfi.b{32,64} d, f, b, pos, len: insert the low len bits of f into b at pos
    void emitBfi(const ptx::Instruction& ins) {
        expectOperands(ins, 5);
        const PtxType t = typeOf(ins);
        const uint32_t w = compIntWidth(t), type = b_.typeInt(w);
        const uint32_t f = value(ins, ins.operands[1], t), b = value(ins, ins.operands[2], t);
        const uint32_t pos = b_.op(spv::OpBitwiseAnd, tU32(), {value(ins, ins.operands[3], {Base::Unsigned, 32}), cU32(0xFF)});
        const uint32_t len = b_.op(spv::OpBitwiseAnd, tU32(), {value(ins, ins.operands[4], {Base::Unsigned, 32}), cU32(0xFF)});
        const uint32_t mask = shl(lowMask(len, w), pos, w);
        const uint32_t r = b_.op(spv::OpBitwiseOr, type, {b_.op(spv::OpBitwiseAnd, type, {b, b_.op(spv::OpNot, type, {mask})}),
                                                          b_.op(spv::OpBitwiseAnd, type, {shl(f, pos, w), mask})});
        storeReg(ins, dstReg(ins).name, t, r);
    }

    // prmt.b32 d, a, b, c (default mode): byte permute of {b, a}
    void emitPrmt(const ptx::Instruction& ins) {
        expectOperands(ins, 4);
        for (std::string_view m : ins.mods)
            if (m != ".b32") fail(ins, "prmt mode " + std::string(m) + " is not supported yet");
        const PtxType bt{Base::Bit, 32};
        const uint32_t a = value(ins, ins.operands[1], bt), b = value(ins, ins.operands[2], bt);
        const uint32_t c = value(ins, ins.operands[3], bt);
        const uint32_t both = b_.op(spv::OpBitwiseOr, tU64(), {extendTo64(a, false),
                                                               b_.op(spv::OpShiftLeftLogical, tU64(), {extendTo64(b, false), cU32(32)})});
        uint32_t r = cU32(0);
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t sel = b_.op(spv::OpBitwiseAnd, tU32(), {b_.op(spv::OpShiftRightLogical, tU32(), {c, cU32(4 * i)}), cU32(0xF)});
            const uint32_t idx = b_.op(spv::OpBitwiseAnd, tU32(), {sel, cU32(7)});
            uint32_t byte = truncTo32(b_.op(spv::OpShiftRightLogical, tU64(), {both, b_.op(spv::OpShiftLeftLogical, tU32(), {idx, cU32(3)})}));
            byte = b_.op(spv::OpBitwiseAnd, tU32(), {byte, cU32(0xFF)});
            // msb mode: replicate the selected byte's sign bit
            const uint32_t sign = b_.op(spv::OpSelect, tU32(), {b_.op(spv::OpINotEqual, tBool(), {b_.op(spv::OpBitwiseAnd, tU32(), {byte, cU32(0x80)}), cU32(0)}), cU32(0xFF), cU32(0)});
            byte = b_.op(spv::OpSelect, tU32(), {b_.op(spv::OpINotEqual, tBool(), {b_.op(spv::OpBitwiseAnd, tU32(), {sel, cU32(8)}), cU32(0)}), sign, byte});
            r = b_.op(spv::OpBitwiseOr, tU32(), {r, b_.op(spv::OpShiftLeftLogical, tU32(), {byte, cU32(8 * i)})});
        }
        storeReg(ins, dstReg(ins).name, bt, r);
    }

    // lop3.b32 d, a, b, c, lut: any 3-input logic function (lut = F(0xF0, 0xCC, 0xAA))
    void emitLop3(const ptx::Instruction& ins) {
        expectOperands(ins, 5);
        const ptx::Operand& lutOp = ins.operands[4];
        if (lutOp.kind != ptx::OperandKind::Immediate) fail(ins, "lop3 needs an immediate lookup table");
        const uint32_t lut = uint32_t(lutOp.imm) & 0xFF;
        const PtxType bt{Base::Bit, 32};
        const uint32_t in[3] = {value(ins, ins.operands[1], bt), value(ins, ins.operands[2], bt),
                                value(ins, ins.operands[3], bt)};
        const uint32_t notIn[3] = {b_.op(spv::OpNot, tU32(), {in[0]}), b_.op(spv::OpNot, tU32(), {in[1]}),
                                   b_.op(spv::OpNot, tU32(), {in[2]})};
        uint32_t r = cU32(0);
        if (lut == 0xFF) r = cU32(0xFFFFFFFF);
        else
            for (uint32_t i = 0; i < 8; ++i) {
                if (!((lut >> i) & 1)) continue;
                uint32_t term = (i & 4) ? in[0] : notIn[0];
                term = b_.op(spv::OpBitwiseAnd, tU32(), {term, (i & 2) ? in[1] : notIn[1]});
                term = b_.op(spv::OpBitwiseAnd, tU32(), {term, (i & 1) ? in[2] : notIn[2]});
                r = b_.op(spv::OpBitwiseOr, tU32(), {r, term});
            }
        storeReg(ins, dstReg(ins).name, bt, r);
    }

    // shf.{l,r}.{wrap,clamp}.b32 d, a, b, n: funnel shift of {b:a}
    void emitShf(const ptx::Instruction& ins) {
        expectOperands(ins, 4);
        const PtxType bt{Base::Bit, 32};
        const uint32_t a = value(ins, ins.operands[1], bt), b = value(ins, ins.operands[2], bt);
        uint32_t n = value(ins, ins.operands[3], bt);
        n = ins.has(".wrap") ? b_.op(spv::OpBitwiseAnd, tU32(), {n, cU32(31)}) : glsl(tU32(), GLSLstd450UMin, {n, cU32(32)});
        const uint32_t v = b_.op(spv::OpBitwiseOr, tU64(), {extendTo64(a, false),
                                                            b_.op(spv::OpShiftLeftLogical, tU64(), {extendTo64(b, false), cU32(32)})});
        const uint32_t r = ins.has(".l") ? hiWord(b_.op(spv::OpShiftLeftLogical, tU64(), {v, n}))
                                         : truncTo32(b_.op(spv::OpShiftRightLogical, tU64(), {v, n}));
        storeReg(ins, dstReg(ins).name, bt, r);
    }

    // add.cc / addc{.cc} / sub.cc / subc{.cc}: the carry flag lives in a function variable
    void emitCarry(const ptx::Instruction& ins) {
        expectOperands(ins, 3);
        const PtxType t = typeOf(ins);
        if (!t.isInt() || (t.width != 32 && t.width != 64)) fail(ins, "32/64-bit integer expected");
        const bool sub = ins.opcode == "sub" || ins.opcode == "subc";
        const bool useCarry = ins.opcode == "addc" || ins.opcode == "subc";
        const uint32_t a = value(ins, ins.operands[1], t), b = value(ins, ins.operands[2], t);
        storeReg(ins, dstReg(ins).name, t, addWithCarry(t, a, b, sub, useCarry, ins.has(".cc")));
    }

    // a + b (or a - b), plus the carry flag (minus the borrow) if useCarry; stores the new flag if setCarry.
    uint32_t addWithCarry(PtxType t, uint32_t a, uint32_t b, bool sub, bool useCarry, bool setCarry) {
        if (!carryVar_) {
            carryVar_ = b_.localVariable(tU32());
            b_.name(carryVar_, "carry");
        }
        const uint32_t w = t.width, type = b_.typeInt(w), pair = structPair(type);
        auto widen = [&](uint32_t c32) { return w == 64 ? extendTo64(c32, false) : c32; };
        const spv::Op op = sub ? spv::OpISubBorrow : spv::OpIAddCarry;
        uint32_t res = b_.op(op, pair, {a, b});
        uint32_t r = b_.op(spv::OpCompositeExtract, type, {res, 0});
        uint32_t c = b_.op(spv::OpCompositeExtract, type, {res, 1});
        if (useCarry) {
            const uint32_t res2 = b_.op(op, pair, {r, widen(b_.op(spv::OpLoad, tU32(), {carryVar_}))});
            r = b_.op(spv::OpCompositeExtract, type, {res2, 0});
            c = b_.op(spv::OpBitwiseOr, type, {c, b_.op(spv::OpCompositeExtract, type, {res2, 1})});
        }
        if (setCarry) b_.opVoid(spv::OpStore, {carryVar_, w == 64 ? truncTo32(c) : c});
        return r;
    }

    // dp4a.atype.btype d, a, b, c = c + sum of the four byte products
    // dp2a.{lo,hi}.atype.btype d, a, b, c = c + sum of a's two 16-bit halves times two bytes of b
    void emitDotProduct(const ptx::Instruction& ins) {
        expectOperands(ins, 4);
        std::vector<PtxType> types;
        for (std::string_view m : ins.mods) {
            PtxType x;
            if (parseType(m, x)) types.push_back(x);
        }
        if (types.size() != 2) fail(ins, "dot product needs two operand types");
        const PtxType bt{Base::Bit, 32};
        const uint32_t a = value(ins, ins.operands[1], bt), b = value(ins, ins.operands[2], bt);
        uint32_t acc = value(ins, ins.operands[3], bt);
        // field `index` of width `w` bits, sign- or zero-extended
        auto field = [&](uint32_t v, uint32_t index, uint32_t w, bool isSigned) {
            return b_.op(isSigned ? spv::OpBitFieldSExtract : spv::OpBitFieldUExtract, tU32(),
                         {v, cU32(w * index), cU32(w)});
        };
        const bool sa = types[0].isSigned(), sb = types[1].isSigned();
        if (ins.opcode == "dp4a" && t_.integerDotProduct) {
            if (!dotProductDeclared_) {
                b_.extension("SPV_KHR_integer_dot_product");
                b_.capability(spv::CapabilityDotProduct);
                b_.capability(spv::CapabilityDotProductInput4x8BitPacked);
                dotProductDeclared_ = true;
            }
            const uint32_t packed = uint32_t(spv::PackedVectorFormatPackedVectorFormat4x8Bit);
            uint32_t dot;
            if (sa && sb) dot = b_.op(spv::OpSDot, tU32(), {a, b, packed});
            else if (!sa && !sb) dot = b_.op(spv::OpUDot, tU32(), {a, b, packed});
            else dot = b_.op(spv::OpSUDot, tU32(), {sa ? a : b, sa ? b : a, packed});  // signed operand first
            storeReg(ins, dstReg(ins).name, bt, b_.op(spv::OpIAdd, tU32(), {acc, dot}));
            return;
        }
        if (ins.opcode == "dp4a" && t_.int16) {
            // Packed 16-bit lanes: every byte product fits in 16 bits (|s8 * s8| <= 2^14,
            // s8 * u8 and u8 * u8 likewise), and GPUs with 16-bit ints multiply two per
            // full-rate instruction; the four products are summed in 32 bits.
            b_.capability(spv::CapabilityInt16);
            const uint32_t v2 = b_.typeVector(b_.typeInt(16), 2);
            const uint32_t eight = b_.constComposite(v2, {b_.constInt(16, 8), b_.constInt(16, 8)});
            // bytes 0, 2 (even) or 1, 3 (odd) of x in 16-bit lanes, extended
            auto lanes = [&](uint32_t x, bool odd, bool isSigned) {
                if (!isSigned) {
                    const uint32_t y = odd ? b_.op(spv::OpShiftRightLogical, tU32(), {x, cU32(8)}) : x;
                    return b_.op(spv::OpBitcast, v2, {b_.op(spv::OpBitwiseAnd, tU32(), {y, cU32(0x00FF00FF)})});
                }
                const uint32_t y = odd ? x : b_.op(spv::OpShiftLeftLogical, tU32(), {x, cU32(8)});
                return b_.op(spv::OpShiftRightArithmetic, v2, {b_.op(spv::OpBitcast, v2, {y}), eight});
            };
            const bool signedProducts = sa || sb;
            const uint32_t v2x32 = b_.typeVector(tU32(), 2);
            for (bool odd : {false, true}) {
                const uint32_t p = b_.op(spv::OpIMul, v2, {lanes(a, odd, sa), lanes(b, odd, sb)});
                const uint32_t wide = b_.op(signedProducts ? spv::OpSConvert : spv::OpUConvert, v2x32, {p});
                acc = b_.op(spv::OpIAdd, tU32(), {acc, b_.op(spv::OpCompositeExtract, tU32(), {wide, 0})});
                acc = b_.op(spv::OpIAdd, tU32(), {acc, b_.op(spv::OpCompositeExtract, tU32(), {wide, 1})});
            }
        } else if (ins.opcode == "dp4a") {
            for (uint32_t i = 0; i < 4; ++i)
                acc = b_.op(spv::OpIAdd, tU32(), {acc, b_.op(spv::OpIMul, tU32(), {field(a, i, 8, sa), field(b, i, 8, sb)})});
        } else {
            const uint32_t first = ins.has(".hi") ? 2 : 0;
            for (uint32_t i = 0; i < 2; ++i)
                acc = b_.op(spv::OpIAdd, tU32(), {acc, b_.op(spv::OpIMul, tU32(), {field(a, i, 16, sa), field(b, first + i, 8, sb)})});
        }
        storeReg(ins, dstReg(ins).name, bt, acc);
    }

    // sad d, a, b, c = |a - b| + c;  cnot d, a = a == 0;  slct d, a, b, c = c >= 0 ? a : b
    void emitMisc(const ptx::Instruction& ins) {
        const std::string_view op = ins.opcode;
        if (op == "cnot") {
            expectOperands(ins, 2);
            const PtxType t = typeOf(ins);
            const uint32_t w = compIntWidth(t);
            const uint32_t z = b_.op(spv::OpIEqual, tBool(), {value(ins, ins.operands[1], t), cInt(w, 0)});
            storeReg(ins, dstReg(ins).name, t, b_.op(spv::OpSelect, b_.typeInt(w), {z, cInt(w, 1), cInt(w, 0)}));
            return;
        }
        expectOperands(ins, 4);
        if (op == "sad") {
            const PtxType t = typeOf(ins);
            const uint32_t type = compType(t, ins);
            const uint32_t a = value(ins, ins.operands[1], t), b = value(ins, ins.operands[2], t);
            const bool s = t.isSigned();
            const uint32_t hi = glsl(type, s ? GLSLstd450SMax : GLSLstd450UMax, {a, b});
            const uint32_t lo = glsl(type, s ? GLSLstd450SMin : GLSLstd450UMin, {a, b});
            const uint32_t r = b_.op(spv::OpIAdd, type, {b_.op(spv::OpISub, type, {hi, lo}), value(ins, ins.operands[3], t)});
            storeReg(ins, dstReg(ins).name, t, r);
            return;
        }
        // slct.dtype.ctype
        std::vector<PtxType> types;
        for (std::string_view m : ins.mods) {
            PtxType x;
            if (parseType(m, x)) types.push_back(x);
        }
        if (types.size() != 2) fail(ins, "slct needs two types");
        const PtxType dt = types[0], ct = types[1];
        const uint32_t c = value(ins, ins.operands[3], ct);
        const uint32_t ge = ct.isFloat() ? b_.op(spv::OpFOrdGreaterThanEqual, tBool(), {c, b_.constFloat(ct.width, 0)})
                                         : b_.op(spv::OpSGreaterThanEqual, tBool(), {c, cInt(compIntWidth(ct), 0)});
        storeReg(ins, dstReg(ins).name, dt,
                 b_.op(spv::OpSelect, compType(dt, ins), {ge, value(ins, ins.operands[1], dt), value(ins, ins.operands[2], dt)}));
    }

    // set.CmpOp{.BoolOp}.dtype.stype d, a, b{, c}: like setp, but writes all ones / 1.0f
    void emitSet(const ptx::Instruction& ins) {
        if (ins.operands.size() != 3 && ins.operands.size() != 4) fail(ins, "set needs 3 or 4 operands");
        std::vector<PtxType> types;
        std::string_view boolOp;
        for (std::string_view m : ins.mods) {
            PtxType x;
            if (parseType(m, x)) types.push_back(x);
            else if (m == ".and" || m == ".or" || m == ".xor") boolOp = m;
        }
        if (types.size() != 2) fail(ins, "set needs a destination and a source type");
        const PtxType dt = types[0], st = types[1];
        uint32_t p = compare(ins, ins.mods[0], st, value(ins, ins.operands[1], st), value(ins, ins.operands[2], st));
        if (ins.operands.size() == 4) {
            const uint32_t c = value(ins, ins.operands[3], {Base::Pred, 1});
            p = b_.op(boolOp == ".and" ? spv::OpLogicalAnd : boolOp == ".or" ? spv::OpLogicalOr : spv::OpLogicalNotEqual,
                      tBool(), {p, c});
        }
        const uint32_t r = dt.isFloat()
                               ? b_.op(spv::OpSelect, tFloat(32), {p, b_.constFloat(32, 0x3F800000), b_.constFloat(32, 0)})
                               : b_.op(spv::OpSelect, tU32(), {p, cU32(0xFFFFFFFF), cU32(0)});
        storeReg(ins, dstReg(ins).name, dt.isFloat() ? PtxType{Base::Float, 32} : PtxType{Base::Bit, 32}, r);
    }

    // ---- half precision ----------------------------------------------------------------------
    //
    // f16/bf16 values are held as 16 bits in integer registers (x2: two in 32 bits). Arithmetic
    // runs in f32 and rounds back to nearest even: f16 sums and products are exact in f32, so the
    // results match native half arithmetic (fma up to a rare double rounding).

    enum class Half { None, F16, F16x2, BF16, BF16x2 };

    static Half halfOf(const ptx::Instruction& ins) {
        for (std::string_view m : ins.mods) {
            if (m == ".f16") return Half::F16;
            if (m == ".f16x2") return Half::F16x2;
            if (m == ".bf16") return Half::BF16;
            if (m == ".bf16x2") return Half::BF16x2;
        }
        return Half::None;
    }
    static bool isPair(Half h) { return h == Half::F16x2 || h == Half::BF16x2; }
    static bool isBrain(Half h) { return h == Half::BF16 || h == Half::BF16x2; }

    uint32_t halfToFloat(uint32_t bits16, bool brain) {
        const uint32_t f32 = tFloat(32);
        if (brain) return b_.op(spv::OpBitcast, f32, {b_.op(spv::OpShiftLeftLogical, tU32(), {bits16, cU32(16)})});
        return b_.op(spv::OpCompositeExtract, f32, {glsl(b_.typeVector(f32, 2), GLSLstd450UnpackHalf2x16, {bits16}), 0});
    }
    uint32_t floatToHalfBits(uint32_t f, bool brain) {
        if (!brain) return floatToHalf(f);
        const uint32_t u = b_.op(spv::OpBitcast, tU32(), {f});
        const uint32_t lsb = b_.op(spv::OpBitwiseAnd, tU32(), {b_.op(spv::OpShiftRightLogical, tU32(), {u, cU32(16)}), cU32(1)});
        const uint32_t rounded = b_.op(spv::OpShiftRightLogical, tU32(),
                                       {b_.op(spv::OpIAdd, tU32(), {u, b_.op(spv::OpIAdd, tU32(), {cU32(0x7FFF), lsb})}), cU32(16)});
        return b_.op(spv::OpSelect, tU32(), {b_.op(spv::OpIsNan, tBool(), {f}), cU32(0x7FC0), rounded});
    }

    // atom/red.add on .f16 .bf16 .f16x2 .bf16x2: a compare-and-swap loop on the aligned 32-bit
    // word, updating both halves (pairs) or the addressed one.
    void emitHalfAtom(const ptx::Instruction& ins, Half h, std::string_view op, bool isRed, size_t first) {
        if (op != ".add") fail(ins, "only add is supported for half-precision atomics");
        Space space = Space::Generic;
        for (std::string_view m : ins.mods) {
            if (m == ".global") space = Space::Global;
            else if (m == ".shared") space = Space::Shared;
        }
        const bool pair = isPair(h), brain = isBrain(h), cta = ins.has(".cta");
        const uint32_t u32 = tU32();
        withSpace(space, address(ins, ins.operands[first], space), [&](Space sp, uint32_t at) {
            uint32_t ptr, shift = cU32(0);
            if (!pair) {  // the half's bit position in its word
                const uint32_t low = truncTo32(at);
                shift = b_.op(spv::OpShiftLeftLogical, u32, {b_.op(spv::OpBitwiseAnd, u32, {low, cU32(2)}), cU32(3)});
            }
            if (sp == Space::Shared || sp == Space::Local) {
                ptr = wordPointer(sp, wordIndex(at));
            } else {
                const uint32_t aligned = b_.op(spv::OpBitwiseAnd, tU64(), {at, cU64(~uint64_t(3))});
                ptr = globalPointer(aligned, u32);
            }
            const uint32_t scope = sp == Space::Shared || cta ? spv::ScopeWorkgroup : spv::ScopeDevice;
            const uint32_t add0 = halfLane(ins, ins.operands[first + 1], h, 0);
            const uint32_t add1 = pair ? halfLane(ins, ins.operands[first + 1], h, 1) : 0;
            auto lane = [&](uint32_t word, uint32_t sh) {
                const uint32_t bits = b_.op(spv::OpBitwiseAnd, u32, {b_.op(spv::OpShiftRightLogical, u32, {word, sh}), cU32(0xFFFF)});
                return halfToFloat(bits, brain);
            };
            auto sum = [&](uint32_t x, uint32_t y) { return floatToHalfBits(b_.op(spv::OpFAdd, tFloat(32), {x, y}), brain); };
            const uint32_t old = rmwWord(sp, ptr, u32, scope, [&](uint32_t cur) {
                if (pair) {
                    const uint32_t lo = sum(lane(cur, cU32(0)), add0), hi = sum(lane(cur, cU32(16)), add1);
                    return b_.op(spv::OpBitwiseOr, u32, {lo, b_.op(spv::OpShiftLeftLogical, u32, {hi, cU32(16)})});
                }
                const uint32_t v = sum(lane(cur, shift), add0);
                const uint32_t mask = b_.op(spv::OpNot, u32, {b_.op(spv::OpShiftLeftLogical, u32, {cU32(0xFFFF), shift})});
                return b_.op(spv::OpBitwiseOr, u32, {b_.op(spv::OpBitwiseAnd, u32, {cur, mask}),
                                                     b_.op(spv::OpShiftLeftLogical, u32, {v, shift})});
            });
            if (isRed) return;
            if (pair) storeReg(ins, dstReg(ins).name, {Base::Bit, 32}, old);
            else
                storeReg(ins, dstReg(ins).name, {Base::Bit, 16},
                         b_.op(spv::OpBitwiseAnd, u32, {b_.op(spv::OpShiftRightLogical, u32, {old, shift}), cU32(0xFFFF)}));
        });
    }

    // The f32 values of lane `lane` of operand `o`.
    uint32_t halfLane(const ptx::Instruction& ins, const ptx::Operand& o, Half h, uint32_t lane) {
        if (!isPair(h)) return halfToFloat(value(ins, o, {Base::Bit, 16}), isBrain(h));
        const uint32_t v = value(ins, o, {Base::Bit, 32});
        const uint32_t bits = b_.op(spv::OpBitwiseAnd, tU32(), {lane ? b_.op(spv::OpShiftRightLogical, tU32(), {v, cU32(16)}) : v, cU32(0xFFFF)});
        return halfToFloat(bits, isBrain(h));
    }

    // add sub mul fma neg abs min max on .f16/.f16x2/.bf16/.bf16x2
    void emitHalfArith(const ptx::Instruction& ins, Half h) {
        const std::string_view op = ins.opcode;
        const size_t srcs = op == "fma" || op == "mad" ? 3 : op == "neg" || op == "abs" ? 1 : 2;
        expectOperands(ins, srcs + 1);
        const uint32_t f32 = tFloat(32);
        const uint32_t lanes = isPair(h) ? 2 : 1;
        uint32_t result = cU32(0);
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            uint32_t x[3] = {};
            for (size_t k = 0; k < srcs; ++k) x[k] = halfLane(ins, ins.operands[k + 1], h, lane);
            uint32_t r;
            if (op == "add") r = b_.op(spv::OpFAdd, f32, {x[0], x[1]});
            else if (op == "sub") r = b_.op(spv::OpFSub, f32, {x[0], x[1]});
            else if (op == "mul") r = b_.op(spv::OpFMul, f32, {x[0], x[1]});
            // the product of two halves is exact in f32: one rounding even where Fma is unfused
            else if (op == "fma" || op == "mad") r = fusedMulAdd({Base::Float, 32}, x[0], x[1], x[2], false);
            else if (op == "neg") r = b_.op(spv::OpFNegate, f32, {x[0]});
            else if (op == "abs") r = glsl(f32, GLSLstd450FAbs, {x[0]});
            else if (op == "min") r = glsl(f32, GLSLstd450NMin, {x[0], x[1]});
            else if (op == "max") r = glsl(f32, GLSLstd450NMax, {x[0], x[1]});
            else fail(ins, "unsupported half-precision instruction");
            if (ins.has(".sat")) r = saturate(r, {Base::Float, 32});
            uint32_t bits = floatToHalfBits(r, isBrain(h));
            if (lane) bits = b_.op(spv::OpShiftLeftLogical, tU32(), {bits, cU32(16)});
            result = b_.op(spv::OpBitwiseOr, tU32(), {result, bits});
        }
        storeReg(ins, dstReg(ins).name, {Base::Bit, isPair(h) ? 32u : 16u}, result);
    }

    // setp.cmp.f16 p, a, b / setp.cmp.f16x2 p|q, a, b   set.cmp.u32.f16x2 d, a, b
    void emitHalfCompare(const ptx::Instruction& ins, Half h) {
        const bool isSet = ins.opcode == "set";
        if (ins.operands.size() != 3) fail(ins, "half comparisons with a predicate operand are not supported yet");
        const PtxType f32t{Base::Float, 32};
        const uint32_t lanes = isPair(h) ? 2 : 1;
        uint32_t cmp[2];
        for (uint32_t lane = 0; lane < lanes; ++lane)
            cmp[lane] = compare(ins, ins.mods[0], f32t, halfLane(ins, ins.operands[1], h, lane), halfLane(ins, ins.operands[2], h, lane));
        const ptx::Operand& d = ins.operands[0];
        if (!isSet) {
            storeReg(ins, d.name, {Base::Pred, 1}, cmp[0]);
            if (lanes == 2 && !d.pairName.empty()) storeReg(ins, d.pairName, {Base::Pred, 1}, cmp[1]);
            return;
        }
        // integer destination: all-ones per lane; half destination: 1.0 per lane
        const bool halfDst = ins.has(".f16") && ins.mods.size() >= 2 && ins.mods[1] == ".f16";
        const uint32_t one = halfDst ? (isBrain(h) ? 0x3F80u : 0x3C00u) : 0xFFFFu;
        uint32_t r = cU32(0);
        for (uint32_t lane = 0; lane < lanes; ++lane)
            r = b_.op(spv::OpBitwiseOr, tU32(), {r, b_.op(spv::OpSelect, tU32(), {cmp[lane], cU32(one << (16 * lane)), cU32(0)})});
        if (lanes == 1 && !halfDst) r = b_.op(spv::OpSelect, tU32(), {cmp[0], cU32(0xFFFFFFFF), cU32(0)});
        storeReg(ins, d.name, {Base::Bit, 32}, r);
    }

    // ---- printf ------------------------------------------------------------------------------
    //
    // vprintf(format, args) appends a record to the launch's printf buffer, which the driver
    // prints after the kernel finished:
    //   buffer: u32 used bytes, u32 capacity, u64 reserved, records...
    //   record: u64 format address, kPrintfArgBytes bytes copied from the argument block
    // The argument block's size is only known from the format string, so a fixed amount is
    // copied (reads beyond the thread's local memory are clamped).

    void emitPrintf(const ptx::Instruction& ins) {
        const PtxType u64t{Base::Unsigned, 64};
        auto loadParamVar = [&](const ptx::Operand& o) { return loadWords(Space::Local, address(ins, o, Space::Local), u64t); };
        const uint32_t format = loadParamVar(ins.operands[0]);
        const uint32_t args = loadParamVar(ins.operands[1]);
        const uint32_t buf = canonical(pushU64(printfOffset_));
        const uint32_t recordBytes = 8 + KernelBinary::kPrintfArgBytes;
        const uint32_t used = b_.op(spv::OpAtomicIAdd, tU32(), {globalPointer(buf, tU32()), cU32(spv::ScopeDevice), cU32(0), cU32(recordBytes)});
        const uint32_t capacity = b_.op(spv::OpLoad, tU32(), {globalPointer(b_.op(spv::OpIAdd, tU64(), {buf, cU64(4)}), tU32()),
                                                              uint32_t(spv::MemoryAccessAlignedMask), 4});
        const uint32_t fits = b_.op(spv::OpULessThanEqual, tBool(), {b_.op(spv::OpIAdd, tU32(), {used, cU32(recordBytes)}), capacity});
        selection(fits, [&] {
            const uint32_t rec = b_.op(spv::OpIAdd, tU64(), {buf, extendTo64(b_.op(spv::OpIAdd, tU32(), {used, cU32(16)}), false)});
            Access a64;
            a64.type = u64t;
            storeAt(ins, Space::Global, rec, a64, {format});
            Access a32;
            a32.type = {Base::Bit, 32};
            const uint32_t argsVar = b_.localVariable(tU32());
            for (uint32_t w = 0; w < KernelBinary::kPrintfArgBytes / 4; ++w) {
                b_.opVoid(spv::OpStore, {argsVar, cU32(0)});
                const uint32_t at = b_.op(spv::OpIAdd, tU64(), {args, cU64(4 * w)});
                selection(b_.op(spv::OpINotEqual, tBool(), {args, cU64(0)}), [&] {
                    withSpace(Space::Generic, at, [&](Space sp, uint32_t addr) {
                        uint32_t v;
                        if (sp == Space::Local) {
                            const uint32_t idx = glsl(tU32(), GLSLstd450UMin, {wordIndex(addr), cU32(localBytes_ / 4 - 1)});
                            v = b_.op(spv::OpLoad, tU32(), {wordPointer(sp, idx)});
                        } else {
                            v = loadAt(ins, sp, addr, a32)[0];
                        }
                        b_.opVoid(spv::OpStore, {argsVar, v});
                    });
                }, [] {});
                const uint32_t dst = b_.op(spv::OpIAdd, tU64(), {rec, cU64(8 + 4 * w)});
                storeAt(ins, Space::Global, dst, a32, {b_.op(spv::OpLoad, tU32(), {argsVar})});
            }
        }, [] {});
    }

    // assertfail [message], [file], [line], [function]: the first failing thread of the device
    // claims the assert record (KernelBinary::kAssertRecordBytes) and fills it in.
    void emitAssert(const ptx::Instruction& ins) {
        const PtxType u64t{Base::Unsigned, 64}, u32t{Base::Unsigned, 32};
        auto loadParamVar = [&](const ptx::Operand& o, const PtxType& t) {
            return loadWords(Space::Local, address(ins, o, Space::Local), t);
        };
        const uint32_t message = loadParamVar(ins.operands[0], u64t);
        const uint32_t file = loadParamVar(ins.operands[1], u64t);
        const uint32_t line = loadParamVar(ins.operands[2], u32t);
        const uint32_t function = loadParamVar(ins.operands[3], u64t);
        const uint32_t rec = canonical(pushU64(assertOffset_));
        const uint32_t old = b_.op(spv::OpAtomicCompareExchange, tU32(),
                                   {globalPointer(rec, tU32()), cU32(spv::ScopeDevice), cU32(0), cU32(0), cU32(1), cU32(0)});
        selection(b_.op(spv::OpIEqual, tBool(), {old, cU32(0)}), [&] {
            Access a64, a32;
            a64.type = u64t;
            a32.type = u32t;
            auto at = [&](uint32_t offset) { return b_.op(spv::OpIAdd, tU64(), {rec, cU64(offset)}); };
            storeAt(ins, Space::Global, at(4), a32, {line});
            storeAt(ins, Space::Global, at(8), a64, {message});
            storeAt(ins, Space::Global, at(16), a64, {file});
            storeAt(ins, Space::Global, at(24), a64, {function});
            for (char c = 'x'; c <= 'z'; ++c) {
                ptx::Operand ctaid, tid;
                ctaid.kind = tid.kind = ptx::OperandKind::Special;
                ctaid.name = "%ctaid";
                tid.name = "%tid";
                ctaid.component = tid.component = c;
                storeAt(ins, Space::Global, at(32 + 4 * uint32_t(c - 'x')), a32, {special(ins, ctaid, u32t)});
                storeAt(ins, Space::Global, at(44 + 4 * uint32_t(c - 'x')), a32, {special(ins, tid, u32t)});
            }
            // complete: the host may look while the kernel still runs (a later event's synchronization)
            b_.opVoid(spv::OpMemoryBarrier, {cU32(spv::ScopeDevice), cU32(uint32_t(spv::MemorySemanticsReleaseMask) |
                                                                       uint32_t(spv::MemorySemanticsUniformMemoryMask))});
            storeAt(ins, Space::Global, at(60), a32, {cU32(1)});
        }, [] {});
    }

    // ---- warps -------------------------------------------------------------------------------
    //
    // A CUDA warp is always 32 threads. With a subgroup size that is a multiple of 32 a warp is
    // a 32-lane slice of a subgroup and warp operations map to subgroup operations. Smaller
    // subgroups (e.g. llvmpipe's 8) go through a per-warp scratch area in shared memory, which
    // needs workgroup barriers: correct only when the whole block executes the operation.

    // Every path through `nodes` ends in a return (conservatively: a Return last, or an If whose
    // arms both always return).
    static bool alwaysReturns(const std::vector<SNode>& nodes) {
        if (nodes.empty()) return false;
        const SNode& last = nodes.back();
        if (last.kind == SNode::Kind::Return) return true;
        return last.kind == SNode::Kind::If && alwaysReturns(last.body) && alwaysReturns(last.elseBody);
    }

    static bool isBlockBarrier(const ptx::Instruction& ins) {
        return (ins.opcode == "bar" || ins.opcode == "barrier") && !ins.has(".warp");
    }
    bool hasBlockBarrier(const std::vector<SNode>& nodes) const {
        for (const SNode& n : nodes) {
            if (n.kind == SNode::Kind::Code) {
                const BasicBlock& bb = cfg_.blocks[n.block];
                for (size_t i = bb.begin; i < bb.end; ++i)
                    if (isBlockBarrier(fn_.instructions[i])) return true;
            }
            if (hasBlockBarrier(n.body) || hasBlockBarrier(n.elseBody)) return true;
        }
        return false;
    }
    // No Return, and every Br targets a construct inside `nodes` or one in `tail` (ending where
    // `nodes` ends): control only leaves at the end.
    static bool selfContained(const std::vector<SNode>& nodes, const std::vector<uint32_t>& tail) {
        std::vector<uint32_t> inner = tail;
        std::function<bool(const std::vector<SNode>&)> walk = [&](const std::vector<SNode>& ns) {
            for (const SNode& n : ns) {
                if (n.kind == SNode::Kind::Return) return false;
                if (n.kind == SNode::Kind::Block || n.kind == SNode::Kind::Loop) inner.push_back(n.construct);
                if (n.kind == SNode::Kind::Br && std::find(inner.begin(), inner.end(), n.construct) == inner.end())
                    return false;
                if (!walk(n.body) || !walk(n.elseBody)) return false;
            }
            return true;
        };
        return walk(nodes);
    }

    // bar / barrier synchronize the whole block (bar.warp only the warp).
    bool usesBlockBarrier() const {
        for (const ptx::Instruction& ins : fn_.instructions)
            if (isBlockBarrier(ins)) return true;
        return false;
    }

    bool nativeWarps() const {
        return t_.subgroupSize >= 32 && t_.subgroupSize % 32 == 0 && t_.subgroupBallot && t_.subgroupShuffle;
    }

    static bool isWarpOp(std::string_view op) {
        return op == "shfl" || op == "vote" || op == "activemask" || op == "match" || op == "redux" || op == "mma" ||
               op == "ldmatrix";
    }
    bool usesWarpOps() const {
        for (const ptx::Instruction& ins : fn_.instructions)
            if (isWarpOp(ins.opcode)) return true;
        return false;
    }

    uint32_t subgroupInvocation() {
        if (!subgroupId_) {
            b_.capability(spv::CapabilityGroupNonUniform);
            subgroupId_ = b_.globalVariable(b_.typePointer(spv::StorageClassInput, tU32()), spv::StorageClassInput);
            b_.decorate(subgroupId_, spv::DecorationBuiltIn, {uint32_t(spv::BuiltInSubgroupLocalInvocationId)});
            interface_.push_back(subgroupId_);
        }
        return b_.op(spv::OpLoad, tU32(), {subgroupId_});
    }

    uint32_t laneId() {
        if (nativeWarps()) return b_.op(spv::OpBitwiseAnd, tU32(), {subgroupInvocation(), cU32(31)});
        return b_.op(spv::OpBitwiseAnd, tU32(), {linearTid(), cU32(31)});
    }
    uint32_t warpId() { return b_.op(spv::OpShiftRightLogical, tU32(), {linearTid(), cU32(5)}); }

    void workgroupBarrier() {
        b_.opVoid(spv::OpControlBarrier, {cU32(spv::ScopeWorkgroup), cU32(spv::ScopeWorkgroup), cU32(kSemAcqRelAll)});
    }
    // Word `index` of this warp's 32-word scratch area (emulated warps only).
    uint32_t warpScratch(uint32_t index) {
        const uint32_t base = b_.op(spv::OpIAdd, tU32(), {cU32(warpScratchWord_), b_.op(spv::OpShiftLeftLogical, tU32(), {warpId(), cU32(5)})});
        return wordPointer(Space::Shared, b_.op(spv::OpIAdd, tU32(), {base, index}));
    }

    // Bits of the warp's lanes for which `pred` is true.
    uint32_t warpBallot(uint32_t pred) {
        if (nativeWarps()) {
            b_.capability(spv::CapabilityGroupNonUniformBallot);
            const uint32_t ballot = b_.op(spv::OpGroupNonUniformBallot, b_.typeVector(tU32(), 4), {cU32(spv::ScopeSubgroup), pred});
            if (t_.subgroupSize == 32) return b_.op(spv::OpCompositeExtract, tU32(), {ballot, 0});
            const uint32_t which = b_.op(spv::OpShiftRightLogical, tU32(), {subgroupInvocation(), cU32(5)});
            return b_.op(spv::OpVectorExtractDynamic, tU32(), {ballot, which});
        }
        const uint32_t word = warpScratch(cU32(0));
        workgroupBarrier();
        selection(b_.op(spv::OpIEqual, tBool(), {laneId(), cU32(0)}), [&] { b_.opVoid(spv::OpStore, {word, cU32(0)}); }, [] {});
        workgroupBarrier();
        const uint32_t bit = b_.op(spv::OpShiftLeftLogical, tU32(), {cU32(1), laneId()});
        selection(pred, [&] { b_.op(spv::OpAtomicOr, tU32(), {warpScratch(cU32(0)), cU32(spv::ScopeWorkgroup), cU32(0), bit}); }, [] {});
        workgroupBarrier();
        return b_.op(spv::OpLoad, tU32(), {warpScratch(cU32(0))});
    }

    // Value of `v` (u32) in lane `srcLane` (0..31) of this warp.
    uint32_t warpShuffle(uint32_t v, uint32_t srcLane) {
        if (nativeWarps()) {
            b_.capability(spv::CapabilityGroupNonUniformShuffle);
            uint32_t id = srcLane;
            if (t_.subgroupSize > 32)
                id = b_.op(spv::OpIAdd, tU32(), {b_.op(spv::OpBitwiseAnd, tU32(), {subgroupInvocation(), cU32(~31u)}), srcLane});
            return b_.op(spv::OpGroupNonUniformShuffle, tU32(), {cU32(spv::ScopeSubgroup), v, id});
        }
        workgroupBarrier();
        b_.opVoid(spv::OpStore, {warpScratch(laneId()), v});
        workgroupBarrier();
        return b_.op(spv::OpLoad, tU32(), {warpScratch(srcLane)});
    }

    // shfl{.sync}.{up,down,bfly,idx}.b32 d{|p}, a, b, c{, membermask}
    void emitShfl(const ptx::Instruction& ins) {
        const bool sync = ins.has(".sync");
        if (ins.operands.size() != (sync ? 5u : 4u)) fail(ins, "wrong number of shfl operands");
        const PtxType bt{Base::Bit, 32};
        const uint32_t a = value(ins, ins.operands[1], bt);
        const uint32_t bval = b_.op(spv::OpBitwiseAnd, tU32(), {value(ins, ins.operands[2], bt), cU32(31)});
        const uint32_t c = value(ins, ins.operands[3], bt);
        const uint32_t cval = b_.op(spv::OpBitwiseAnd, tU32(), {c, cU32(31)});
        const uint32_t segmask = b_.op(spv::OpBitwiseAnd, tU32(), {b_.op(spv::OpShiftRightLogical, tU32(), {c, cU32(8)}), cU32(31)});
        const uint32_t notSeg = b_.op(spv::OpNot, tU32(), {segmask});
        const uint32_t lane = laneId();
        const uint32_t minLane = b_.op(spv::OpBitwiseAnd, tU32(), {lane, segmask});
        const uint32_t maxLane = b_.op(spv::OpBitwiseOr, tU32(), {minLane, b_.op(spv::OpBitwiseAnd, tU32(), {cval, notSeg})});
        uint32_t j, pval;
        if (ins.has(".up")) {
            j = b_.op(spv::OpISub, tU32(), {lane, bval});
            pval = b_.op(spv::OpSGreaterThanEqual, tBool(), {j, maxLane});
        } else {
            if (ins.has(".down")) j = b_.op(spv::OpIAdd, tU32(), {lane, bval});
            else if (ins.has(".bfly")) j = b_.op(spv::OpBitwiseXor, tU32(), {lane, bval});
            else if (ins.has(".idx")) j = b_.op(spv::OpBitwiseOr, tU32(), {minLane, b_.op(spv::OpBitwiseAnd, tU32(), {bval, notSeg})});
            else fail(ins, "missing shfl mode");
            pval = b_.op(spv::OpSLessThanEqual, tBool(), {j, maxLane});
        }
        j = b_.op(spv::OpSelect, tU32(), {pval, j, lane});
        const ptx::Operand& d = ins.operands[0];
        storeReg(ins, dstReg(ins).name, bt, warpShuffle(a, j));
        if (!d.pairName.empty()) storeReg(ins, d.pairName, {Base::Pred, 1}, pval);
    }

    // Registers of a {...} operand (or a single register).
    std::vector<const ptx::Operand*> vectorElems(const ptx::Instruction& ins, const ptx::Operand& o, size_t n) {
        std::vector<const ptx::Operand*> r;
        if (o.kind == ptx::OperandKind::Vector) {
            for (const ptx::Operand& e : o.elems) r.push_back(&e);
        } else {
            r.push_back(&o);
        }
        if (r.size() != n) fail(ins, "expected a vector of " + std::to_string(n) + " registers");
        return r;
    }

    // mma.sync.aligned.{m16n8k8,m16n8k16}.row.col.{f32,f16}.{f16,bf16}.{f16,bf16}.{f32,f16} d, a, b, c
    // Tensor core matrix multiply-accumulate of a warp, emulated with shuffles: each lane gathers
    // the A row and B column elements of its four outputs from the lanes holding them (fragment
    // layouts from the PTX ISA) and accumulates in f32.
    void emitMma(const ptx::Instruction& ins) {
        if (ins.operands.size() != 4) fail(ins, "mma needs 4 operands");
        uint32_t k = 0;
        if (ins.has(".m16n8k8")) k = 8;
        else if (ins.has(".m16n8k16")) k = 16;
        else fail(ins, "only the m16n8k8 and m16n8k16 mma shapes are supported");
        if (!ins.has(".row") || !ins.has(".col")) fail(ins, "mma layout must be .row.col");
        std::vector<std::string_view> types;  // d, a, b, c
        for (std::string_view m : ins.mods)
            if (m == ".f32" || m == ".f16" || m == ".bf16" || m == ".tf32" || m == ".s8" || m == ".u8" || m == ".s32" ||
                m == ".f64" || m == ".e4m3" || m == ".e5m2")
                types.push_back(m);
        if (types.size() == 4 && types[1] == ".tf32" && types[2] == ".tf32" && types[0] == ".f32" && types[3] == ".f32" &&
            k == 8)
            return emitMmaTf32(ins);
        if (types.size() != 4 || types[1] != types[2] || (types[1] != ".f16" && types[1] != ".bf16") ||
            (types[0] != ".f32" && types[0] != ".f16") || (types[3] != ".f32" && types[3] != ".f16"))
            fail(ins, "only f16/bf16 inputs with f32/f16 accumulators are supported by mma");
        const bool brain = types[1] == ".bf16", f16d = types[0] == ".f16", f16c = types[3] == ".f16";
        const uint32_t kBlocks = k / 8;
        const PtxType bits{Base::Bit, 32}, f32t{Base::Float, 32};
        const auto dOps = vectorElems(ins, ins.operands[0], f16d ? 2 : 4);
        const auto aOps = vectorElems(ins, ins.operands[1], 2 * kBlocks);
        const auto bOps = vectorElems(ins, ins.operands[2], kBlocks);
        const auto cOps = vectorElems(ins, ins.operands[3], f16c ? 2 : 4);

        const uint32_t f32 = tFloat(32);
        auto halves = [&](uint32_t word, uint32_t& lo, uint32_t& hi) {
            if (brain) {
                lo = b_.op(spv::OpBitcast, f32, {b_.op(spv::OpShiftLeftLogical, tU32(), {word, cU32(16)})});
                hi = b_.op(spv::OpBitcast, f32, {b_.op(spv::OpBitwiseAnd, tU32(), {word, cU32(0xFFFF0000u)})});
            } else {
                const uint32_t v = glsl(b_.typeVector(f32, 2), GLSLstd450UnpackHalf2x16, {word});
                lo = b_.op(spv::OpCompositeExtract, f32, {v, 0});
                hi = b_.op(spv::OpCompositeExtract, f32, {v, 1});
            }
        };
        const uint32_t lane = laneId();
        const uint32_t quad = b_.op(spv::OpBitwiseAnd, tU32(), {lane, cU32(~3u)});  // 4 * groupID
        const uint32_t t2 = b_.op(spv::OpShiftLeftLogical, tU32(), {b_.op(spv::OpBitwiseAnd, tU32(), {lane, cU32(3)}), cU32(3)});

        // aRow[r][k]: A[groupID + 8r][k]; bCol[c][k]: B[k][2 threadID_in_group + c]
        std::vector<uint32_t> aRow[2], bCol[2];
        for (uint32_t r = 0; r < 2; ++r) aRow[r].resize(k), bCol[r].resize(k);
        for (uint32_t kb = 0; kb < kBlocks; ++kb) {
            const uint32_t aRegs[2] = {value(ins, *aOps[2 * kb], bits), value(ins, *aOps[2 * kb + 1], bits)};
            const uint32_t bReg = value(ins, *bOps[kb], bits);
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t kk = 8 * kb + 2 * j;
                for (uint32_t r = 0; r < 2; ++r)
                    halves(warpShuffle(aRegs[r], b_.op(spv::OpIAdd, tU32(), {quad, cU32(j)})), aRow[r][kk], aRow[r][kk + 1]);
                for (uint32_t c = 0; c < 2; ++c)
                    halves(warpShuffle(bReg, b_.op(spv::OpIAdd, tU32(), {t2, cU32(4 * c + j)})), bCol[c][kk], bCol[c][kk + 1]);
            }
        }
        uint32_t acc[4];  // (row groupID, groupID + 8) x (col 2t, 2t + 1)
        if (f16c) {
            for (uint32_t r = 0; r < 2; ++r) halves(value(ins, *cOps[r], bits), acc[2 * r], acc[2 * r + 1]);
            if (brain) fail(ins, "mma with bf16 inputs accumulates in f32");
        } else {
            for (uint32_t i = 0; i < 4; ++i) acc[i] = value(ins, *cOps[i], f32t);
        }
        for (uint32_t i = 0; i < 4; ++i)
            for (uint32_t kk = 0; kk < k; ++kk)
                acc[i] = b_.op(spv::OpFAdd, f32, {acc[i], b_.op(spv::OpFMul, f32, {aRow[i / 2][kk], bCol[i % 2][kk]})});
        if (f16d) {
            for (uint32_t r = 0; r < 2; ++r) {
                const uint32_t lo = floatToHalfBits(acc[2 * r], false), hi = floatToHalfBits(acc[2 * r + 1], false);
                storeReg(ins, dOps[r]->name, bits,
                         b_.op(spv::OpBitwiseOr, tU32(), {lo, b_.op(spv::OpShiftLeftLogical, tU32(), {hi, cU32(16)})}));
            }
        } else {
            for (uint32_t i = 0; i < 4; ++i) storeReg(ins, dOps[i]->name, f32t, acc[i]);
        }
    }

    // mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 d, a, b, c: tf32 elements are f32 registers of
    // which the tensor core uses the top 19 bits. Lane (groupID g, threadID_in_group t) holds
    // A[g][t], A[g+8][t], A[g][t+4], A[g+8][t+4] (a0..a3) and B[t][g], B[t+4][g] (b0, b1); C and D as
    // for the f16 shapes.
    void emitMmaTf32(const ptx::Instruction& ins) {
        const PtxType bits{Base::Bit, 32}, f32t{Base::Float, 32};
        const auto dOps = vectorElems(ins, ins.operands[0], 4);
        const auto aOps = vectorElems(ins, ins.operands[1], 4);
        const auto bOps = vectorElems(ins, ins.operands[2], 2);
        const auto cOps = vectorElems(ins, ins.operands[3], 4);
        const uint32_t f32 = tFloat(32);
        auto tf32 = [&](uint32_t word) {
            return b_.op(spv::OpBitcast, f32, {b_.op(spv::OpBitwiseAnd, tU32(), {word, cU32(0xFFFFE000u)})});
        };
        const uint32_t lane = laneId();
        const uint32_t quad = b_.op(spv::OpBitwiseAnd, tU32(), {lane, cU32(~3u)});  // 4 * groupID
        const uint32_t t2 = b_.op(spv::OpShiftLeftLogical, tU32(), {b_.op(spv::OpBitwiseAnd, tU32(), {lane, cU32(3)}), cU32(3)});
        uint32_t a[4], bb[2];
        for (uint32_t i = 0; i < 4; ++i) a[i] = value(ins, *aOps[i], bits);
        for (uint32_t i = 0; i < 2; ++i) bb[i] = value(ins, *bOps[i], bits);
        // aRow[r][k] = A[g + 8r][k], held by lane 4g + k % 4 in a[r + 2 (k / 4)];
        // bCol[c][k] = B[k][2t + c], held by lane 4 (2t + c) + k % 4 in b[k / 4]
        uint32_t aRow[2][8], bCol[2][8];
        for (uint32_t kk = 0; kk < 8; ++kk) {
            const uint32_t j = kk % 4, half = kk / 4;
            for (uint32_t r = 0; r < 2; ++r)
                aRow[r][kk] = tf32(warpShuffle(a[r + 2 * half], b_.op(spv::OpIAdd, tU32(), {quad, cU32(j)})));
            for (uint32_t c = 0; c < 2; ++c)
                bCol[c][kk] = tf32(warpShuffle(bb[half], b_.op(spv::OpIAdd, tU32(), {t2, cU32(4 * c + j)})));
        }
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t acc = value(ins, *cOps[i], f32t);
            for (uint32_t kk = 0; kk < 8; ++kk)
                acc = b_.op(spv::OpFAdd, f32, {acc, b_.op(spv::OpFMul, f32, {aRow[i / 2][kk], bCol[i % 2][kk]})});
            storeReg(ins, dOps[i]->name, f32t, acc);
        }
    }

    // ldmatrix.sync.aligned.m8n8.{x1,x2,x4}{.trans}.shared.b16 d, [a]: lanes 8j..8j+7 hold the row
    // addresses of matrix j; each lane receives two 16-bit elements of every matrix.
    void emitLdmatrix(const ptx::Instruction& ins) {
        if (ins.operands.size() != 2) fail(ins, "ldmatrix needs 2 operands");
        if (!ins.has(".m8n8") || !ins.has(".b16") || !ins.has(".shared")) fail(ins, "only ldmatrix.m8n8.shared.b16 is supported");
        const uint32_t n = ins.has(".x4") ? 4 : ins.has(".x2") ? 2 : 1;
        const bool trans = ins.has(".trans");
        const auto dOps = vectorElems(ins, ins.operands[0], n);
        const uint32_t addr = address(ins, ins.operands[1], Space::Shared);
        const uint32_t lane = laneId();
        const uint32_t row = b_.op(spv::OpShiftRightLogical, tU32(), {lane, cU32(2)});  // 0..7
        const uint32_t quarter = b_.op(spv::OpBitwiseAnd, tU32(), {lane, cU32(3)});
        const PtxType b32{Base::Bit, 32}, b16{Base::Bit, 16};
        for (uint32_t j = 0; j < n; ++j) {
            uint32_t word;
            if (!trans) {  // row `row`, columns 2 quarter, 2 quarter + 1
                const uint32_t rowAddr = warpShuffle(addr, b_.op(spv::OpIAdd, tU32(), {row, cU32(8 * j)}));
                word = loadWords(Space::Shared, b_.op(spv::OpIAdd, tU32(), {rowAddr, b_.op(spv::OpShiftLeftLogical, tU32(), {quarter, cU32(2)})}), b32);
            } else {  // column `row`, rows 2 quarter and 2 quarter + 1
                const uint32_t r0 = b_.op(spv::OpIAdd, tU32(), {b_.op(spv::OpShiftLeftLogical, tU32(), {quarter, cU32(1)}), cU32(8 * j)});
                const uint32_t col = b_.op(spv::OpShiftLeftLogical, tU32(), {row, cU32(1)});
                const uint32_t a0 = warpShuffle(addr, r0), a1 = warpShuffle(addr, b_.op(spv::OpIAdd, tU32(), {r0, cU32(1)}));
                const uint32_t lo = loadWords(Space::Shared, b_.op(spv::OpIAdd, tU32(), {a0, col}), b16);
                const uint32_t hi = loadWords(Space::Shared, b_.op(spv::OpIAdd, tU32(), {a1, col}), b16);
                word = b_.op(spv::OpBitwiseOr, tU32(), {lo, b_.op(spv::OpShiftLeftLogical, tU32(), {hi, cU32(16)})});
            }
            storeReg(ins, dOps[j]->name, b32, word);
        }
    }

    // vote{.sync}.{all,any,uni}.pred d, {!}p{, mask}   vote{.sync}.ballot.b32 d, {!}p{, mask}
    void emitVote(const ptx::Instruction& ins) {
        const bool sync = ins.has(".sync");
        if (ins.operands.size() != (sync ? 3u : 2u)) fail(ins, "wrong number of vote operands");
        const uint32_t p = value(ins, ins.operands[1], {Base::Pred, 1});
        const uint32_t mask = sync ? value(ins, ins.operands[2], {Base::Bit, 32}) : cU32(0xFFFFFFFF);
        const uint32_t ballot = b_.op(spv::OpBitwiseAnd, tU32(), {warpBallot(p), mask});
        if (ins.has(".ballot")) {
            storeReg(ins, dstReg(ins).name, {Base::Bit, 32}, ballot);
            return;
        }
        const uint32_t active = b_.op(spv::OpBitwiseAnd, tU32(), {warpBallot(b_.constBool(true)), mask});
        const uint32_t all = b_.op(spv::OpIEqual, tBool(), {ballot, active});
        const uint32_t none = b_.op(spv::OpIEqual, tBool(), {ballot, cU32(0)});
        uint32_t r;
        if (ins.has(".all")) r = all;
        else if (ins.has(".any")) r = b_.op(spv::OpLogicalNot, tBool(), {none});
        else if (ins.has(".uni")) r = b_.op(spv::OpLogicalOr, tBool(), {all, none});
        else fail(ins, "missing vote mode");
        storeReg(ins, dstReg(ins).name, {Base::Pred, 1}, r);
    }

    void emitActivemask(const ptx::Instruction& ins) {
        expectOperands(ins, 1);
        storeReg(ins, dstReg(ins).name, {Base::Bit, 32}, warpBallot(b_.constBool(true)));
    }

    // match.{any,all}.sync.b32 d{|p}, a, mask: lanes holding the same value as this one
    void emitMatch(const ptx::Instruction& ins) {
        if (ins.operands.size() != 3) fail(ins, "match needs 3 operands");
        const PtxType t = typeOf(ins);
        if (t.width != 32) fail(ins, "only match.b32 is supported");
        const uint32_t a = value(ins, ins.operands[1], {Base::Bit, 32});
        const uint32_t mask = value(ins, ins.operands[2], {Base::Bit, 32});
        uint32_t same = cU32(0);
        for (uint32_t i = 0; i < 32; ++i) {
            const uint32_t eq = b_.op(spv::OpIEqual, tBool(), {warpShuffle(a, cU32(i)), a});
            same = b_.op(spv::OpBitwiseOr, tU32(), {same, b_.op(spv::OpSelect, tU32(), {eq, cU32(1u << i), cU32(0)})});
        }
        same = b_.op(spv::OpBitwiseAnd, tU32(), {same, mask});
        const ptx::Operand& d = ins.operands[0];
        if (ins.has(".all")) {
            const uint32_t all = b_.op(spv::OpIEqual, tBool(), {same, b_.op(spv::OpBitwiseAnd, tU32(), {warpBallot(b_.constBool(true)), mask})});
            storeReg(ins, d.name, {Base::Bit, 32}, b_.op(spv::OpSelect, tU32(), {all, mask, cU32(0)}));
            if (!d.pairName.empty()) storeReg(ins, d.pairName, {Base::Pred, 1}, all);
        } else {
            storeReg(ins, d.name, {Base::Bit, 32}, same);
        }
    }

    // redux.sync.{add,min,max}.{u32,s32} d, a, mask   redux.sync.{and,or,xor}.b32 d, a, mask
    // Lane by lane like match: only the lanes in mask are read, so a partial mask with the other
    // lanes inactive works too.
    void emitRedux(const ptx::Instruction& ins) {
        if (ins.operands.size() != 3) fail(ins, "redux needs 3 operands");
        const PtxType t = typeOf(ins);
        if (t.width != 32 || t.isFloat()) fail(ins, "only 32-bit integer redux is supported");
        const bool s = t.isSigned();
        std::function<uint32_t(uint32_t, uint32_t)> combine;
        auto bitwise = [&](spv::Op op) { combine = [this, op](uint32_t x, uint32_t y) { return b_.op(op, tU32(), {x, y}); }; };
        auto ext = [&](GLSLstd450 inst) { combine = [this, inst](uint32_t x, uint32_t y) { return glsl(tU32(), inst, {x, y}); }; };
        if (ins.has(".add")) bitwise(spv::OpIAdd);
        else if (ins.has(".min")) ext(s ? GLSLstd450SMin : GLSLstd450UMin);
        else if (ins.has(".max")) ext(s ? GLSLstd450SMax : GLSLstd450UMax);
        else if (ins.has(".and")) bitwise(spv::OpBitwiseAnd);
        else if (ins.has(".or")) bitwise(spv::OpBitwiseOr);
        else if (ins.has(".xor")) bitwise(spv::OpBitwiseXor);
        else fail(ins, "missing redux operation");
        const uint32_t a = value(ins, ins.operands[1], {Base::Bit, 32});
        const uint32_t mask = value(ins, ins.operands[2], {Base::Bit, 32});
        uint32_t acc = 0;  // starts at the first lane in mask: no identity per operation needed
        uint32_t seen = b_.constBool(false);
        for (uint32_t i = 0; i < 32; ++i) {
            const uint32_t v = warpShuffle(a, cU32(i));
            const uint32_t in = b_.op(spv::OpINotEqual, tBool(), {b_.op(spv::OpBitwiseAnd, tU32(), {mask, cU32(1u << i)}), cU32(0)});
            const uint32_t combined = i ? b_.op(spv::OpSelect, tU32(), {seen, combine(acc, v), v}) : v;
            acc = i ? b_.op(spv::OpSelect, tU32(), {in, combined, acc}) : v;
            seen = b_.op(spv::OpLogicalOr, tBool(), {seen, in});
        }
        storeReg(ins, dstReg(ins).name, {Base::Bit, 32}, acc);
    }

    void emitInstruction(const ptx::Instruction& ins) {
        const std::string_view op = ins.opcode;
        if (op != "cvt" && op != "ld" && op != "st" && op != "mov" && op != "atom" && op != "red" && op != "mma" &&
            op != "ldmatrix") {
            if (const Half h = halfOf(ins); h != Half::None) {
                if (op == "setp" || op == "set") emitHalfCompare(ins, h);
                else emitHalfArith(ins, h);
                return;
            }
        }
        if ((op == "add" || op == "sub") && ins.has(".cc")) emitCarry(ins);
        else if (op == "mov") emitMov(ins);
        else if (op == "ld" || op == "st" || op == "atom" || op == "red") emitOrdered(ins);
        else if (op == "add" || op == "sub" || op == "div" || op == "rem" || op == "min" || op == "max" ||
                 op == "and" || op == "or" || op == "xor")
            emitBinary(ins);
        else if (op == "mul") {
            if (typeOf(ins).isFloat()) emitBinary(ins);
            else emitMul(ins, false);
        } else if (op == "mad" || op == "madc") emitMul(ins, true);
        else if (op == "fma") emitFma(ins);
        else if (op == "not" || op == "neg" || op == "abs") emitUnary(ins);
        else if (op == "shl" || op == "shr") emitShift(ins);
        else if (op == "setp") emitSetp(ins);
        else if (op == "selp") emitSelp(ins);
        else if (op == "cvt") emitCvt(ins);
        else if (op == "cvta") emitCvta(ins);
        else if (op == "isspacep") emitIsspacep(ins);
        else if (op == "sqrt" || op == "rsqrt" || op == "rcp" || op == "ex2" || op == "lg2" || op == "sin" ||
                 op == "cos" || op == "tanh")
            emitMathUnary(ins);
        else if (op == "copysign") emitCopysign(ins);
        else if (op == "testp") emitTestp(ins);
        else if (op == "popc" || op == "clz" || op == "brev") emitBitScan(ins);
        else if (op == "bfe") emitBfe(ins);
        else if (op == "bfind") emitBfind(ins);
        else if (op == "bfi") emitBfi(ins);
        else if (op == "prmt") emitPrmt(ins);
        else if (op == "lop3") emitLop3(ins);
        else if (op == "shf") emitShf(ins);
        else if (op == "addc" || op == "subc" || ((op == "add" || op == "sub") && ins.has(".cc"))) emitCarry(ins);
        else if (op == "sad" || op == "cnot" || op == "slct") emitMisc(ins);
        else if (op == "dp4a" || op == "dp2a") emitDotProduct(ins);
        else if (op == "brkpt" || op == "pmevent" || op == "nanosleep") {}  // no effect here
        // Programmatic dependent launch (sm_90): kernels of a stream already run one after the other,
        // each seeing the previous one's writes, so waiting for the prerequisite grid and releasing
        // the dependent one are both no-ops.
        else if (op == "griddepcontrol") {}
        else if (op == "set") emitSet(ins);
        else if (op == "shfl") emitShfl(ins);
        else if (op == "mma") emitMma(ins);
        else if (op == "ldmatrix") emitLdmatrix(ins);
        else if (op == "cp" && ins.has(".async")) emitCpAsync(ins);
        else if (op == "vote") emitVote(ins);
        else if (op == "activemask") emitActivemask(ins);
        else if (op == "match") emitMatch(ins);
        else if (op == "redux") emitRedux(ins);
        else if (op == "vprintf") emitPrintf(ins);
        else if (op == "assertfail") emitAssert(ins);
        else if (op == "bar" || op == "barrier" || op == "membar" || op == "fence") emitBarrier(ins);
        else fail(ins, "unsupported instruction '" + std::string(op) + "'");
    }

    // Predicated instructions become a small selection construct.
    void emitGuarded(const ptx::Instruction& ins) {
        if (ins.guard.empty()) {
            emitInstruction(ins);
            return;
        }
        uint32_t cond = loadReg(ins, ins.guard, {Base::Pred, 1});
        if (ins.guardNegated) cond = b_.op(spv::OpLogicalNot, tBool(), {cond});
        if (!nativeWarps() && isWarpOp(ins.opcode)) {
            // Emulated warp operations synchronize the workgroup (barriers around a scratch area),
            // so every thread must reach them: all run the instruction, and the guard decides whose
            // destination registers take the result.
            std::vector<std::pair<const Reg*, uint32_t>> saved;
            const ptx::Operand& d = ins.operands[0];
            if (d.kind == ptx::OperandKind::Register)
                for (std::string_view name : {std::string_view(d.name), std::string_view(d.pairName)})
                    if (!name.empty()) {
                        const Reg& r = reg(ins, name);
                        saved.emplace_back(&r, b_.op(spv::OpLoad, r.storage, {r.var}));
                    }
            emitInstruction(ins);
            for (const auto& [r, old] : saved) {
                const uint32_t now = b_.op(spv::OpLoad, r->storage, {r->var});
                b_.opVoid(spv::OpStore, {r->var, b_.op(spv::OpSelect, r->storage, {cond, now, old})});
            }
            return;
        }
        const uint32_t then = b_.newId(), merge = b_.newId();
        b_.opVoid(spv::OpSelectionMerge, {merge, uint32_t(spv::SelectionControlMaskNone)});
        b_.opVoid(spv::OpBranchConditional, {cond, then, merge});
        b_.label(then);
        emitInstruction(ins);
        b_.opVoid(spv::OpBranch, {merge});
        b_.label(merge);
    }

    // ---- structured control flow -------------------------------------------------------------

    struct Construct {
        uint32_t id;
        bool isLoop;
        uint32_t merge, cont;
        bool mergeReached = false;
        std::vector<uint32_t> pendingExits;  // outer constructs some branch inside wants to reach
    };

    void branchTo(uint32_t label) {
        b_.opVoid(spv::OpBranch, {label});
        open_ = false;
    }

    uint32_t exitVar() {
        if (!exitVar_) {
            // Starts at 0: merge blocks also read it on paths that never set it.
            exitVar_ = b_.localVariable(tU32(), cU32(0));
            b_.name(exitVar_, "exit_target");
        }
        return exitVar_;
    }

    // Br to construct `target`. SPIR-V only allows leaving the innermost loop, so exits further
    // out record their target in a variable and are re-dispatched after each merge block.
    void emitBr(uint32_t target, bool fromDispatch) {
        Construct& inner = constructs_.back();
        if (inner.id == target) {
            if (fromDispatch) b_.opVoid(spv::OpStore, {exitVar(), cU32(0)});
            if (inner.isLoop) {
                branchTo(inner.cont);
            } else {
                inner.mergeReached = true;
                branchTo(inner.merge);
            }
            return;
        }
        if (!fromDispatch) b_.opVoid(spv::OpStore, {exitVar(), cU32(target)});
        if (std::find(inner.pendingExits.begin(), inner.pendingExits.end(), target) == inner.pendingExits.end())
            inner.pendingExits.push_back(target);
        inner.mergeReached = true;
        branchTo(inner.merge);
    }

    // A thread that ends publishes its shared memory writes. CUDA lets threads that exited
    // early count as arrived at later __syncthreads(), so what they wrote is read by others
    // after a barrier they never execute. Without a release here, Vulkan's memory model makes
    // those writes unobservable and compilers delete them (NIR does: both ACO and LLVM).
    void emitReturn() {
        if (hasShared_)
            b_.opVoid(spv::OpMemoryBarrier, {cU32(spv::ScopeWorkgroup),
                                            cU32(uint32_t(spv::MemorySemanticsReleaseMask) |
                                                 uint32_t(spv::MemorySemanticsWorkgroupMemoryMask))});
        b_.opVoid(spv::OpReturn, {});
    }

    void startBlock(uint32_t label) {
        b_.label(label);
        open_ = true;
    }

    // `tail`: Block constructs that end where `nodes` ends, so a Br to them only falls through.
    void emitNodes(const std::vector<SNode>& nodes, const std::vector<uint32_t>& tail = {}) {
        for (size_t index = 0; index < nodes.size(); ++index) {
            const SNode& n = nodes[index];
            if (!open_) return;
            const std::vector<uint32_t> noTail;
            const std::vector<uint32_t>& armTail = index + 1 == nodes.size() ? tail : noTail;
            switch (n.kind) {
            case SNode::Kind::Code: {
                const BasicBlock& bb = cfg_.blocks[n.block];
                for (size_t i = bb.begin; i < bb.end; ++i) emitGuarded(fn_.instructions[i]);
                break;
            }
            case SNode::Kind::Return:
                emitReturn();
                open_ = false;
                break;
            case SNode::Kind::Br:
                emitBr(n.construct, false);
                break;
            case SNode::Kind::If: {
                const ptx::Instruction fake{};
                uint32_t cond = loadReg(fake, n.pred, {Base::Pred, 1});
                if (n.predNegated) cond = b_.op(spv::OpLogicalNot, tBool(), {cond});
                // One arm returns, the other carries on (often the rest of the kernel): returning
                // lanes go first, in a selection of their own, and the rest follows it in program
                // order. Inside one selection, drivers may run the arms in either order, and with
                // CUDA warps sharing a wave the continuing arm's __syncthreads and shared reads
                // could then run before the returning warps' shared writes (plan.md 3.13).
                // Likewise when only one arm synchronizes the block (and neither leaves itself):
                // the other arm, typically warps writing partial results to shared memory for
                // the synchronizing warps to read, runs first, as `if (!c) E; if (c) T`.
                if (blockBarrier_ && hasBlockBarrier(n.body) != hasBlockBarrier(n.elseBody) &&
                    selfContained(n.body, armTail) && selfContained(n.elseBody, armTail)) {
                    const bool thenSyncs = hasBlockBarrier(n.body);
                    const uint32_t notCond = b_.op(spv::OpLogicalNot, tBool(), {cond});
                    for (int part = 0; part < 2; ++part) {
                        const bool first = part == 0;
                        const std::vector<SNode>& arm = first == thenSyncs ? n.elseBody : n.body;
                        const uint32_t armCond = (first == thenSyncs) ? notCond : cond;
                        const uint32_t armL = b_.newId(), merge = b_.newId();
                        b_.opVoid(spv::OpSelectionMerge, {merge, uint32_t(spv::SelectionControlMaskNone)});
                        b_.opVoid(spv::OpBranchConditional, {armCond, armL, merge});
                        startBlock(armL);
                        emitNodes(arm, armTail);
                        if (open_) branchTo(merge);
                        startBlock(merge);
                    }
                    break;
                }
                if (blockBarrier_) {
                    const bool thenReturns = alwaysReturns(n.body), elseReturns = alwaysReturns(n.elseBody);
                    if (thenReturns != elseReturns) {
                        const uint32_t retCond = thenReturns ? cond : b_.op(spv::OpLogicalNot, tBool(), {cond});
                        const uint32_t retL = b_.newId(), merge = b_.newId();
                        b_.opVoid(spv::OpSelectionMerge, {merge, uint32_t(spv::SelectionControlMaskNone)});
                        b_.opVoid(spv::OpBranchConditional, {retCond, retL, merge});
                        startBlock(retL);
                        emitNodes(thenReturns ? n.body : n.elseBody, armTail);
                        if (open_) branchTo(merge);  // not reached: the arm always returns
                        startBlock(merge);
                        emitNodes(thenReturns ? n.elseBody : n.body, armTail);
                        break;
                    }
                }
                const uint32_t thenL = b_.newId(), elseL = b_.newId(), merge = b_.newId();
                b_.opVoid(spv::OpSelectionMerge, {merge, uint32_t(spv::SelectionControlMaskNone)});
                b_.opVoid(spv::OpBranchConditional, {cond, thenL, elseL});
                bool reached = false;
                startBlock(thenL);
                emitNodes(n.body, armTail);
                if (open_) branchTo(merge), reached = true;
                startBlock(elseL);
                emitNodes(n.elseBody, armTail);
                if (open_) branchTo(merge), reached = true;
                startBlock(merge);
                if (!reached) {
                    b_.opVoid(spv::OpUnreachable, {});
                    open_ = false;
                }
                break;
            }
            case SNode::Kind::Block:
            case SNode::Kind::Loop: {
                const uint32_t header = b_.newId(), body = b_.newId(), cont = b_.newId(), merge = b_.newId();
                branchTo(header);
                startBlock(header);
                b_.opVoid(spv::OpLoopMerge, {merge, cont, uint32_t(spv::LoopControlMaskNone)});
                branchTo(body);
                startBlock(body);
                constructs_.push_back({n.construct, n.kind == SNode::Kind::Loop, merge, cont});
                std::vector<uint32_t> bodyTail;
                if (n.kind == SNode::Kind::Block) {
                    bodyTail = armTail;
                    bodyTail.push_back(n.construct);
                }
                emitNodes(n.body, bodyTail);
                if (open_) {
                    constructs_.back().mergeReached = true;
                    branchTo(merge);
                }
                Construct c = std::move(constructs_.back());
                constructs_.pop_back();
                startBlock(cont);  // a Block's continue target is never reached
                branchTo(header);
                startBlock(merge);
                if (!c.mergeReached) {
                    b_.opVoid(spv::OpUnreachable, {});
                    open_ = false;
                    break;
                }
                for (uint32_t target : c.pendingExits) {
                    const uint32_t v = b_.op(spv::OpLoad, tU32(), {exitVar()});
                    const uint32_t cond = b_.op(spv::OpIEqual, tBool(), {v, cU32(target)});
                    const uint32_t thenL = b_.newId(), next = b_.newId();
                    b_.opVoid(spv::OpSelectionMerge, {next, uint32_t(spv::SelectionControlMaskNone)});
                    b_.opVoid(spv::OpBranchConditional, {cond, thenL, next});
                    startBlock(thenL);
                    emitBr(target, true);
                    startBlock(next);
                }
                break;
            }
            }
        }
    }

    // ---- module ------------------------------------------------------------------------------

    void emitModule() {
        b_.setExactFloat(true);
        b_.capability(spv::CapabilityShader);
        b_.capability(spv::CapabilityInt64);
        b_.capability(spv::CapabilityPhysicalStorageBufferAddresses);
        glsl_ = b_.importExtInst("GLSL.std.450");
        b_.memoryModel(spv::AddressingModelPhysicalStorageBuffer64, spv::MemoryModelGLSL450);

        // blockDim from specialization constants 0/1/2, exposed as the WorkgroupSize builtin.
        for (uint32_t i = 0; i < 3; ++i) blockDim_[i] = b_.specConstInt(32, 1, i);

        layoutMemory();
        blockBarrier_ = usesBlockBarrier();
        packable_ = !hasShared_ && !blockBarrier_ && nativeWarps();
        uint32_t wgX = blockDim_[0];
        if (packable_) {
            pack_ = b_.specConstInt(32, 1, 4);
            packed_ = b_.specConstOp(tBool(), spv::OpUGreaterThan, {pack_, cU32(1)});
            wgX = b_.specConstOp(tU32(), spv::OpIMul, {blockDim_[0], pack_});
            usesGrid_ = true;  // the last workgroup's surplus blocks compare against gridDim.x
        }
        const uint32_t wgSize = b_.specConstComposite(b_.typeVector(tU32(), 3), {wgX, blockDim_[1], blockDim_[2]});
        b_.decorate(wgSize, spv::DecorationBuiltIn, {uint32_t(spv::BuiltInWorkgroupSize)});

        setupPushConstants();

        const uint32_t voidT = b_.typeVoid();
        const uint32_t fn = b_.beginFunction(voidT, b_.typeFunction(voidT));
        declareRegisters();
        declareMemory();
        open_ = true;
        if (packable_) {  // blocks past the grid in the last packed workgroup do nothing
            const uint32_t past = b_.op(spv::OpUGreaterThanEqual, tBool(), {ctaidX(), pushWord(gridOffset_ / 4)});
            const uint32_t cond = b_.op(spv::OpLogicalAnd, tBool(), {packed_, past});
            const uint32_t retL = b_.newId(), merge = b_.newId();
            b_.opVoid(spv::OpSelectionMerge, {merge, uint32_t(spv::SelectionControlMaskNone)});
            b_.opVoid(spv::OpBranchConditional, {cond, retL, merge});
            b_.label(retL);
            b_.opVoid(spv::OpReturn, {});
            b_.label(merge);
        }
        emitNodes(st_.body);
        if (open_) emitReturn();
        b_.endFunction();

        b_.entryPoint(spv::ExecutionModelGLCompute, fn, e_.name, interface_);
        b_.executionMode(fn, spv::ExecutionModeLocalSize, {1, 1, 1});
    }

    const ptx::Module& m_;
    const ptx::Entry& e_;
    const Target& t_;
    ptx::Function fn_;
    Cfg cfg_;
    Structured st_;
    SpirvBuilder b_;

    uint32_t glsl_ = 0;
    uint32_t blockDim_[3] = {};
    bool packable_ = false, blockBarrier_ = false;
    uint32_t pack_ = 0, packed_ = 0;  // spec constant 4 and pack_ > 1
    uint32_t pushVar_ = 0, pushBytes_ = 0;
    std::unordered_map<std::string_view, Symbol> symbols_;
    uint32_t sharedStatic_ = 0, localBytes_ = 0, globalsOffset_ = 0, gridOffset_ = 0;
    bool paramsInBuffer_ = false, usesGrid_ = false, usesPrintf_ = false, usesAssert_ = false, usesWorkspace_ = false;
    uint32_t printfOffset_ = 0, assertOffset_ = 0;
    uint32_t workspaceOffset_ = 0;
    uint32_t addressHigh_ = 0;  // specialization constant 5 (canonical())
    bool usesGlobals_ = false, hasShared_ = false;
    uint32_t sharedVar_ = 0, localVar_ = 0, dynamicWords_ = 0;
    uint32_t localId_ = 0, groupId_ = 0;
    uint32_t exitVar_ = 0, carryVar_ = 0, subgroupId_ = 0;
    bool dotProductDeclared_ = false;
    uint32_t warpScratchWord_ = 0;  // first word of the emulated warps' scratch area
    uint32_t barrierWord_ = 0;      // bar.red counter
    std::vector<uint32_t> interface_;
    std::unordered_map<std::string_view, Reg> regs_;
    std::unordered_map<uint32_t, uint32_t> pairTypes_;
    std::vector<Construct> constructs_;
    bool open_ = false;
};

}  // namespace

bool compileKernel(const ptx::Module& module, const ptx::Entry& entry, const Target& target, KernelBinary& out,
                   std::string& error) {
    Translator t(module, entry, target);
    return t.run(out, error);
}

}  // namespace vulkanizer::compiler
