#include "../include/stemmer.hpp"
#include <algorithm>

std::string Stemmer::stem(std::string word) {
    // Convert to lowercase
    std::transform(word.begin(), word.end(), word.begin(), ::tolower);
    
    // Remove possessives
    if (word.length() > 2 && word.substr(word.length()-2) == "'s") {
        word = word.substr(0, word.length()-2);
    }

    // Handle plurals and common endings
    if (word.length() > 3) {
        if (word.length() > 5 && (word.substr(word.length()-4) == "tion" || 
            word.substr(word.length()-4) == "sion")) {
            return word.substr(0, word.length()-3) + "e";
        }
        if (word.length() > 4 && word.substr(word.length()-4) == "ment") {
            return word.substr(0, word.length()-4);
        }
        if (word.length() > 4 && word.substr(word.length()-3) == "ing") {
            std::string stem = word.substr(0, word.length()-3);
            if (stem.length() > 1) {
                if (stem.length() > 2 && stem[stem.length()-1] == stem[stem.length()-2]) {
                    return stem.substr(0, stem.length()-1);
                }
                if (needs_e_after_ing(stem)) {
                    return stem + "e";
                }
                return stem;
            }
        }
        if (word.back() == 's' && word[word.length()-2] != 's') {
            return word.substr(0, word.length()-1);
        }
    }
    return word;
}

bool Stemmer::needs_e_after_ing(const std::string& stem) {
    if (stem.empty()) return false;
    char last = stem.back();
    char second_last = stem.length() > 1 ? stem[stem.length()-2] : '\0';
    bool ends_in_consonant = !is_vowel(last);
    bool preceded_by_vowel = is_vowel(second_last);
    return (ends_in_consonant && preceded_by_vowel) || last == 'v' || last == 'z';
}

bool Stemmer::needs_e_after_ed(const std::string& stem) {
    return needs_e_after_ing(stem);
}

bool Stemmer::is_vowel(char c) {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
} 