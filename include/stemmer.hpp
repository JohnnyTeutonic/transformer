#pragma once
#include <string>

class Stemmer {
public:
    static std::string stem(std::string word);
private:
    static bool needs_e_after_ing(const std::string& stem);
    static bool needs_e_after_ed(const std::string& stem);
    static bool is_vowel(char c);
}; 