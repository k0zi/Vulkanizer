#pragma once

#include <spirv/unified1/spirv.hpp>

#include <cstdint>
#include <initializer_list>
#include <map>
#include <string_view>
#include <vector>

namespace vulkanizer::compiler {

// Minimal SPIR-V module writer. Keeps each logical section in its own word
// stream and concatenates them in the order the spec requires. Types and
// constants are deduplicated. One function at a time; its local variables are
// hoisted to the start of the entry block as the spec requires.
class SpirvBuilder {
public:
    using Ops = std::initializer_list<uint32_t>;

    uint32_t newId() { return bound_++; }

    // Decorates every float arithmetic result with NoContraction, which keeps drivers from
    // fusing or reassociating it. PTX math relies on exact IEEE operations (e.g. the rounding
    // tricks in expf), while SPIR-V otherwise allows such rewrites.
    void setExactFloat(bool on) { exactFloat_ = on; }

    void capability(spv::Capability c);
    void extension(std::string_view name);
    uint32_t importExtInst(std::string_view name);
    void memoryModel(spv::AddressingModel a, spv::MemoryModel m);
    void entryPoint(spv::ExecutionModel model, uint32_t fn, std::string_view name, const std::vector<uint32_t>& iface);
    void executionMode(uint32_t fn, spv::ExecutionMode mode, Ops literals);
    void name(uint32_t id, std::string_view n);
    void decorate(uint32_t id, spv::Decoration d, Ops literals = {});
    void memberDecorate(uint32_t structId, uint32_t member, spv::Decoration d, Ops literals = {});

    uint32_t typeVoid();
    uint32_t typeBool();
    uint32_t typeInt(uint32_t width);  // always unsigned; signed ops do not need signed types
    uint32_t typeFloat(uint32_t width);
    uint32_t typeVector(uint32_t component, uint32_t count);
    uint32_t typeArray(uint32_t element, uint32_t length);  // length is a literal, a constant is created
    uint32_t typeArrayOf(uint32_t element, uint32_t lengthId);  // length is a (spec) constant id
    uint32_t typePointer(spv::StorageClass sc, uint32_t pointee);
    uint32_t typeFunction(uint32_t ret);
    uint32_t typeStruct(const std::vector<uint32_t>& members);  // not deduplicated (decorations differ)

    uint32_t constInt(uint32_t width, uint64_t value);
    uint32_t constFloat(uint32_t width, uint64_t bits);
    uint32_t constBool(bool v);
    uint32_t constNull(uint32_t type);
    uint32_t constComposite(uint32_t type, const std::vector<uint32_t>& parts);
    uint32_t specConstInt(uint32_t width, uint64_t defaultValue, uint32_t specId);
    uint32_t specConstComposite(uint32_t type, const std::vector<uint32_t>& parts);
    uint32_t specConstOp(uint32_t type, spv::Op op, const std::vector<uint32_t>& operands);
    uint32_t globalVariable(uint32_t ptrType, spv::StorageClass sc);

    // Function body. beginFunction opens the entry block.
    uint32_t beginFunction(uint32_t retType, uint32_t fnType);
    uint32_t localVariable(uint32_t valueType, uint32_t initializer = 0);  // Function storage, returns the pointer
    void endFunction();

    // Instructions go to the current block.
    uint32_t op(spv::Op op, uint32_t resultType, Ops operands);
    uint32_t op(spv::Op op, uint32_t resultType, const std::vector<uint32_t>& operands);
    void opVoid(spv::Op op, Ops operands);
    void opVoid(spv::Op op, const std::vector<uint32_t>& operands);
    uint32_t extInst(uint32_t set, uint32_t resultType, uint32_t instruction, Ops operands);
    void label(uint32_t id);

    std::vector<uint32_t> finish(uint32_t version = 0x00010500);

private:
    static void emit(std::vector<uint32_t>& out, spv::Op op, Ops words);
    static void emit(std::vector<uint32_t>& out, spv::Op op, const std::vector<uint32_t>& words);
    static void appendString(std::vector<uint32_t>& out, std::string_view s);
    uint32_t cachedType(std::vector<uint32_t> key, spv::Op op, Ops operands);

    void markExact(spv::Op op, uint32_t id);

    uint32_t bound_ = 1;
    bool exactFloat_ = false;
    std::vector<uint32_t> capabilities_, extensions_, imports_, memoryModel_, entryPoints_, executionModes_, debug_,
        annotations_, globals_, functions_;
    std::vector<uint32_t> fnHeader_, fnVars_, fnBody_;
    std::map<std::vector<uint32_t>, uint32_t> cache_;
    std::vector<spv::Capability> declaredCaps_;
};

}  // namespace vulkanizer::compiler
