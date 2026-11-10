#include "compiler/spirv_builder.h"

#include <algorithm>

namespace vulkanizer::compiler {

void SpirvBuilder::emit(std::vector<uint32_t>& out, spv::Op op, Ops words) {
    out.push_back(uint32_t(words.size() + 1) << 16 | uint32_t(op));
    out.insert(out.end(), words.begin(), words.end());
}

void SpirvBuilder::emit(std::vector<uint32_t>& out, spv::Op op, const std::vector<uint32_t>& words) {
    out.push_back(uint32_t(words.size() + 1) << 16 | uint32_t(op));
    out.insert(out.end(), words.begin(), words.end());
}

void SpirvBuilder::appendString(std::vector<uint32_t>& out, std::string_view s) {
    // Nul-terminated, little-endian, padded to a word boundary.
    for (size_t i = 0; i <= s.size(); i += 4) {
        uint32_t w = 0;
        for (size_t k = 0; k < 4 && i + k < s.size(); ++k) w |= uint32_t(uint8_t(s[i + k])) << (8 * k);
        out.push_back(w);
    }
}

void SpirvBuilder::capability(spv::Capability c) {
    if (std::find(declaredCaps_.begin(), declaredCaps_.end(), c) != declaredCaps_.end()) return;
    declaredCaps_.push_back(c);
    emit(capabilities_, spv::OpCapability, {uint32_t(c)});
}

void SpirvBuilder::extension(std::string_view name) {
    std::vector<uint32_t> w;
    appendString(w, name);
    emit(extensions_, spv::OpExtension, w);
}

uint32_t SpirvBuilder::importExtInst(std::string_view name) {
    const uint32_t id = newId();
    std::vector<uint32_t> w{id};
    appendString(w, name);
    emit(imports_, spv::OpExtInstImport, w);
    return id;
}

void SpirvBuilder::memoryModel(spv::AddressingModel a, spv::MemoryModel m) {
    memoryModel_.clear();
    emit(memoryModel_, spv::OpMemoryModel, {uint32_t(a), uint32_t(m)});
}

void SpirvBuilder::entryPoint(spv::ExecutionModel model, uint32_t fn, std::string_view n,
                              const std::vector<uint32_t>& iface) {
    std::vector<uint32_t> w{uint32_t(model), fn};
    appendString(w, n);
    w.insert(w.end(), iface.begin(), iface.end());
    emit(entryPoints_, spv::OpEntryPoint, w);
}

void SpirvBuilder::executionMode(uint32_t fn, spv::ExecutionMode mode, Ops literals) {
    std::vector<uint32_t> w{fn, uint32_t(mode)};
    w.insert(w.end(), literals.begin(), literals.end());
    emit(executionModes_, spv::OpExecutionMode, w);
}

void SpirvBuilder::name(uint32_t id, std::string_view n) {
    std::vector<uint32_t> w{id};
    appendString(w, n);
    emit(debug_, spv::OpName, w);
}

void SpirvBuilder::decorate(uint32_t id, spv::Decoration d, Ops literals) {
    std::vector<uint32_t> w{id, uint32_t(d)};
    w.insert(w.end(), literals.begin(), literals.end());
    emit(annotations_, spv::OpDecorate, w);
}

void SpirvBuilder::memberDecorate(uint32_t structId, uint32_t member, spv::Decoration d, Ops literals) {
    std::vector<uint32_t> w{structId, member, uint32_t(d)};
    w.insert(w.end(), literals.begin(), literals.end());
    emit(annotations_, spv::OpMemberDecorate, w);
}

uint32_t SpirvBuilder::cachedType(std::vector<uint32_t> key, spv::Op op, Ops operands) {
    key.insert(key.begin(), uint32_t(op));
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    const uint32_t id = newId();
    // Constants are (result type, result id, ...); types are (result id, ...).
    const bool typed = op == spv::OpConstant || op == spv::OpConstantTrue || op == spv::OpConstantFalse ||
                       op == spv::OpConstantNull;
    std::vector<uint32_t> w;
    auto rest = operands.begin();
    if (typed) w.push_back(*rest++);
    w.push_back(id);
    w.insert(w.end(), rest, operands.end());
    emit(globals_, op, w);
    cache_.emplace(std::move(key), id);
    return id;
}

uint32_t SpirvBuilder::typeVoid() { return cachedType({}, spv::OpTypeVoid, {}); }
uint32_t SpirvBuilder::typeBool() { return cachedType({}, spv::OpTypeBool, {}); }
uint32_t SpirvBuilder::typeInt(uint32_t width) { return cachedType({width}, spv::OpTypeInt, {width, 0}); }
uint32_t SpirvBuilder::typeFloat(uint32_t width) { return cachedType({width}, spv::OpTypeFloat, {width}); }
uint32_t SpirvBuilder::typeVector(uint32_t component, uint32_t count) {
    return cachedType({component, count}, spv::OpTypeVector, {component, count});
}
uint32_t SpirvBuilder::typeArray(uint32_t element, uint32_t length) {
    const uint32_t len = constInt(32, length);
    return cachedType({element, len}, spv::OpTypeArray, {element, len});
}
uint32_t SpirvBuilder::typeArrayOf(uint32_t element, uint32_t lengthId) {
    return cachedType({element, lengthId}, spv::OpTypeArray, {element, lengthId});
}
uint32_t SpirvBuilder::typePointer(spv::StorageClass sc, uint32_t pointee) {
    return cachedType({uint32_t(sc), pointee}, spv::OpTypePointer, {uint32_t(sc), pointee});
}
uint32_t SpirvBuilder::typeFunction(uint32_t ret) { return cachedType({ret}, spv::OpTypeFunction, {ret}); }

uint32_t SpirvBuilder::typeStruct(const std::vector<uint32_t>& members) {
    const uint32_t id = newId();
    std::vector<uint32_t> w{id};
    w.insert(w.end(), members.begin(), members.end());
    emit(globals_, spv::OpTypeStruct, w);
    return id;
}

uint32_t SpirvBuilder::constInt(uint32_t width, uint64_t value) {
    const uint32_t type = typeInt(width);
    if (width == 64)
        return cachedType({type, uint32_t(value), uint32_t(value >> 32)}, spv::OpConstant,
                          {type, uint32_t(value), uint32_t(value >> 32)});
    if (width < 32) value &= (uint64_t(1) << width) - 1;
    return cachedType({type, uint32_t(value)}, spv::OpConstant, {type, uint32_t(value)});
}

uint32_t SpirvBuilder::constFloat(uint32_t width, uint64_t bits) {
    const uint32_t type = typeFloat(width);
    if (width == 64)
        return cachedType({type, uint32_t(bits), uint32_t(bits >> 32)}, spv::OpConstant,
                          {type, uint32_t(bits), uint32_t(bits >> 32)});
    return cachedType({type, uint32_t(bits)}, spv::OpConstant, {type, uint32_t(bits)});
}

uint32_t SpirvBuilder::constBool(bool v) {
    const uint32_t type = typeBool();
    return cachedType({type}, v ? spv::OpConstantTrue : spv::OpConstantFalse, {type});
}

uint32_t SpirvBuilder::constNull(uint32_t type) { return cachedType({type}, spv::OpConstantNull, {type}); }

uint32_t SpirvBuilder::specConstInt(uint32_t width, uint64_t defaultValue, uint32_t specId) {
    const uint32_t type = typeInt(width);
    const uint32_t id = newId();
    if (width == 64) emit(globals_, spv::OpSpecConstant, {type, id, uint32_t(defaultValue), uint32_t(defaultValue >> 32)});
    else emit(globals_, spv::OpSpecConstant, {type, id, uint32_t(defaultValue)});
    decorate(id, spv::DecorationSpecId, {specId});
    return id;
}

uint32_t SpirvBuilder::constComposite(uint32_t type, const std::vector<uint32_t>& parts) {
    std::vector<uint32_t> key{uint32_t(spv::OpConstantComposite), type};
    key.insert(key.end(), parts.begin(), parts.end());
    if (auto it = cache_.find(key); it != cache_.end()) return it->second;
    const uint32_t id = newId();
    std::vector<uint32_t> w{type, id};
    w.insert(w.end(), parts.begin(), parts.end());
    emit(globals_, spv::OpConstantComposite, w);
    cache_.emplace(std::move(key), id);
    return id;
}

uint32_t SpirvBuilder::specConstComposite(uint32_t type, const std::vector<uint32_t>& parts) {
    const uint32_t id = newId();
    std::vector<uint32_t> w{type, id};
    w.insert(w.end(), parts.begin(), parts.end());
    emit(globals_, spv::OpSpecConstantComposite, w);
    return id;
}

uint32_t SpirvBuilder::specConstOp(uint32_t type, spv::Op op, const std::vector<uint32_t>& operands) {
    const uint32_t id = newId();
    std::vector<uint32_t> w{type, id, uint32_t(op)};
    w.insert(w.end(), operands.begin(), operands.end());
    emit(globals_, spv::OpSpecConstantOp, w);
    return id;
}

uint32_t SpirvBuilder::globalVariable(uint32_t ptrType, spv::StorageClass sc) {
    const uint32_t id = newId();
    emit(globals_, spv::OpVariable, {ptrType, id, uint32_t(sc)});
    return id;
}

uint32_t SpirvBuilder::beginFunction(uint32_t retType, uint32_t fnType) {
    const uint32_t fn = newId();
    fnHeader_.clear();
    fnVars_.clear();
    fnBody_.clear();
    emit(fnHeader_, spv::OpFunction, {retType, fn, uint32_t(spv::FunctionControlMaskNone), fnType});
    emit(fnHeader_, spv::OpLabel, {newId()});
    return fn;
}

uint32_t SpirvBuilder::localVariable(uint32_t valueType, uint32_t initializer) {
    const uint32_t ptr = typePointer(spv::StorageClassFunction, valueType);
    const uint32_t id = newId();
    if (initializer) emit(fnVars_, spv::OpVariable, {ptr, id, uint32_t(spv::StorageClassFunction), initializer});
    else emit(fnVars_, spv::OpVariable, {ptr, id, uint32_t(spv::StorageClassFunction)});
    return id;
}

void SpirvBuilder::endFunction() {
    functions_.insert(functions_.end(), fnHeader_.begin(), fnHeader_.end());
    functions_.insert(functions_.end(), fnVars_.begin(), fnVars_.end());
    functions_.insert(functions_.end(), fnBody_.begin(), fnBody_.end());
    emit(functions_, spv::OpFunctionEnd, {});
    fnHeader_.clear();
    fnVars_.clear();
    fnBody_.clear();
}

void SpirvBuilder::markExact(spv::Op op, uint32_t id) {
    if (!exactFloat_) return;
    switch (op) {
    case spv::OpFAdd:
    case spv::OpFSub:
    case spv::OpFMul:
    case spv::OpFDiv:
    case spv::OpFRem:
    case spv::OpFMod:
    case spv::OpFNegate:
    case spv::OpExtInst:
        decorate(id, spv::DecorationNoContraction);
        break;
    default:
        break;
    }
}

uint32_t SpirvBuilder::op(spv::Op o, uint32_t resultType, Ops operands) {
    const uint32_t id = newId();
    markExact(o, id);
    fnBody_.push_back(uint32_t(operands.size() + 3) << 16 | uint32_t(o));
    fnBody_.push_back(resultType);
    fnBody_.push_back(id);
    fnBody_.insert(fnBody_.end(), operands.begin(), operands.end());
    return id;
}

uint32_t SpirvBuilder::op(spv::Op o, uint32_t resultType, const std::vector<uint32_t>& operands) {
    const uint32_t id = newId();
    markExact(o, id);
    fnBody_.push_back(uint32_t(operands.size() + 3) << 16 | uint32_t(o));
    fnBody_.push_back(resultType);
    fnBody_.push_back(id);
    fnBody_.insert(fnBody_.end(), operands.begin(), operands.end());
    return id;
}

void SpirvBuilder::opVoid(spv::Op o, Ops operands) { emit(fnBody_, o, operands); }
void SpirvBuilder::opVoid(spv::Op o, const std::vector<uint32_t>& operands) { emit(fnBody_, o, operands); }

uint32_t SpirvBuilder::extInst(uint32_t set, uint32_t resultType, uint32_t instruction, Ops operands) {
    std::vector<uint32_t> w{set, instruction};
    w.insert(w.end(), operands.begin(), operands.end());
    return op(spv::OpExtInst, resultType, w);
}

void SpirvBuilder::label(uint32_t id) { emit(fnBody_, spv::OpLabel, {id}); }

std::vector<uint32_t> SpirvBuilder::finish(uint32_t version) {
    std::vector<uint32_t> out{spv::MagicNumber, version, 0 /* generator */, bound_, 0};
    for (auto* sec : {&capabilities_, &extensions_, &imports_, &memoryModel_, &entryPoints_, &executionModes_, &debug_,
                      &annotations_, &globals_, &functions_})
        out.insert(out.end(), sec->begin(), sec->end());
    return out;
}

}  // namespace vulkanizer::compiler
