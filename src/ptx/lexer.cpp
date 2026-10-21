#include "ptx/lexer.h"

#include <bit>
#include <cctype>
#include <charconv>
#include <cstring>

namespace vulkanizer::ptx {

static bool isIdentStart(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$';
}
static bool isIdentChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$';
}
static bool isDigit(char c) { return std::isdigit(static_cast<unsigned char>(c)); }
static bool isHexDigit(char c) { return std::isxdigit(static_cast<unsigned char>(c)); }

bool parseInteger(std::string_view s, uint64_t& out) {
    if (!s.empty() && s.back() == 'U') s.remove_suffix(1);
    int base = 10;
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) base = 16, s.remove_prefix(2);
    else if (s.size() > 2 && s[0] == '0' && (s[1] == 'b' || s[1] == 'B')) base = 2, s.remove_prefix(2);
    auto res = std::from_chars(s.data(), s.data() + s.size(), out, base);
    return res.ec == std::errc() && res.ptr == s.data() + s.size();
}

bool parseFloat(std::string_view s, uint64_t& bits, uint32_t& width) {
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'f' || s[1] == 'F' || s[1] == 'd' || s[1] == 'D')) {
        width = (s[1] == 'f' || s[1] == 'F') ? 32 : 64;
        auto res = std::from_chars(s.data() + 2, s.data() + s.size(), bits, 16);
        return res.ec == std::errc() && res.ptr == s.data() + s.size() && s.size() - 2 == width / 4;
    }
    double v = 0;
    auto res = std::from_chars(s.data(), s.data() + s.size(), v);
    if (res.ec != std::errc() || res.ptr != s.data() + s.size()) return false;
    width = 64;
    bits = std::bit_cast<uint64_t>(v);
    return true;
}

bool tokenize(std::string_view s, std::vector<Token>& out, ParseError& err, uint32_t firstLine) {
    const size_t n = s.size();
    size_t i = 0;
    uint32_t line = firstLine;
    auto at = [&](size_t k) { return k < n ? s[k] : '\0'; };
    auto push = [&](TokenKind kind, size_t begin) { out.push_back({kind, s.substr(begin, i - begin), line}); };

    while (i < n) {
        const char c = s[i];
        if (c == '\n') {
            ++line;
            ++i;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
            continue;
        }
        if (c == '/' && at(i + 1) == '/') {
            while (i < n && s[i] != '\n') ++i;
            continue;
        }
        if (c == '/' && at(i + 1) == '*') {
            const uint32_t startLine = line;
            i += 2;
            while (i < n && !(s[i] == '*' && at(i + 1) == '/')) {
                if (s[i] == '\n') ++line;
                ++i;
            }
            if (i >= n) {
                err = {"unterminated block comment", startLine};
                return false;
            }
            i += 2;
            continue;
        }

        const size_t begin = i;
        if (c == '.' && isIdentStart(at(i + 1))) {
            ++i;
            while (i < n && isIdentChar(s[i])) ++i;
            push(TokenKind::Directive, begin);
            continue;
        }
        if (c == '%' && isIdentStart(at(i + 1))) {
            ++i;
            while (i < n && isIdentChar(s[i])) ++i;
            push(TokenKind::Register, begin);
            continue;
        }
        if (isIdentStart(c)) {
            while (i < n && isIdentChar(s[i])) ++i;
            push(TokenKind::Identifier, begin);
            continue;
        }
        if (isDigit(c)) {
            const char p = at(i + 1);
            if (c == '0' && (p == 'f' || p == 'F' || p == 'd' || p == 'D') && isHexDigit(at(i + 2))) {
                i += 2;  // hex-encoded IEEE float: 0f3F800000 / 0d...
                while (i < n && isHexDigit(s[i])) ++i;
                push(TokenKind::Float, begin);
                continue;
            }
            if (c == '0' && (p == 'x' || p == 'X') && isHexDigit(at(i + 2))) {
                i += 2;
                while (i < n && isHexDigit(s[i])) ++i;
                if (at(i) == 'U') ++i;
                push(TokenKind::Integer, begin);
                continue;
            }
            if (c == '0' && (p == 'b' || p == 'B') && (at(i + 2) == '0' || at(i + 2) == '1')) {
                i += 2;
                while (i < n && (s[i] == '0' || s[i] == '1')) ++i;
                if (at(i) == 'U') ++i;
                push(TokenKind::Integer, begin);
                continue;
            }
            bool isFloat = false;
            while (i < n && isDigit(s[i])) ++i;
            if (at(i) == '.' && isDigit(at(i + 1))) {
                isFloat = true;
                ++i;
                while (i < n && isDigit(s[i])) ++i;
            }
            if ((at(i) == 'e' || at(i) == 'E') &&
                (isDigit(at(i + 1)) || ((at(i + 1) == '+' || at(i + 1) == '-') && isDigit(at(i + 2))))) {
                isFloat = true;
                i += 2;
                while (i < n && isDigit(s[i])) ++i;
            }
            if (!isFloat && at(i) == 'U') ++i;
            push(isFloat ? TokenKind::Float : TokenKind::Integer, begin);
            continue;
        }
        if (c == '"') {
            ++i;
            while (i < n && s[i] != '"') {
                if (s[i] == '\\' && i + 1 < n) ++i;
                if (s[i] == '\n') ++line;
                ++i;
            }
            if (i >= n) {
                err = {"unterminated string literal", line};
                return false;
            }
            ++i;
            push(TokenKind::String, begin);
            continue;
        }
        if (c != '\0' && std::strchr("{}()[],;:<>+-!@=|&^~*/%?", c)) {
            ++i;
            push(TokenKind::Punct, begin);
            continue;
        }
        err = {std::string("unexpected character '") + c + "'", line};
        return false;
    }
    out.push_back({TokenKind::End, {}, line});
    return true;
}

}  // namespace vulkanizer::ptx
