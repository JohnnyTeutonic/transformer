class Transformer {
private:
    // ... existing code ...
    
    // Dynamic temperature scaling that respects base temperature from config
    float get_dynamic_temperature(PhraseType phrase_type, std::mt19937& gen);
    
    // ... existing code ...
}; 