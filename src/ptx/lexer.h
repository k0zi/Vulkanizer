#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vulkanizer::ptx {

enum class TokenKind {
    Identifier,  // vecAdd, ld, $L__BB0_2
    Directive,   // .version, .u32, .global (the leading dot is part of the text)
    Register,    // %r1, %tid (the leading percent is part of the text)
    Integer,     // 42, 0x1F, 0b101, 7U
    Float,       // 1.5, 7.8, 0f3F800000, 0d3FF0000000000000
    String,      // "file.cu" (quotes included)
    Punct,       // single character: { } ( ) [ ] , ; : < > + - ! @ = | & ^ ~ * / % ?
    End,
};

struct Token {
    TokenKind kind;
    std::string_view text;  // view into the source buffer
    uint32_t line;
};

struct ParseError {
    std::string message;
    uint32_t line = 0;
};

// Value of an Integer token (decimal, 0x hex, 0b binary, optional U suffix). False if out of range.
bool parseInteger(std::string_view text, uint64_t& out);

// Value of a Float token as raw IEEE bits. `width` is 32 for 0f..., 64 for 0d... and decimal literals.
bool parseFloat(std::string_view text, uint64_t& bits, uint32_t& width);

// Tokens reference `source`, which must outlive them. Always appends a final End token on success.
// `firstLine`: the line number of source[0] (bodies are tokenized on their own, see parser.h).
bool tokenize(std::string_view source, std::vector<Token>& out, ParseError& err, uint32_t firstLine = 1);

}  // namespace vulkanizer::ptx
