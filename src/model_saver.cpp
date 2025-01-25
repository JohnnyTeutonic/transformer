#include "../include/model_saver.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

ModelSaver::ModelSaver() : logger(Logger::getInstance()) {}

bool ModelSaver::saveModel(Transformer& transformer, const std::string& directory,
                           const std::string& model_name) {
    try {
        std::string dir_path = createDirectory(directory);
        std::string model_path = dir_path + "/" + model_name + ".ckpt";

        logger.log("Saving model to: " + model_path);

        // Save model configuration
        if (!writeMetadata(directory, model_name, transformer.getConfig())) {
            logger.log("Failed to save model metadata", true);
            return false;
        }

        // Save model weights
        std::ofstream model_file(model_path, std::ios::binary);
        if (!model_file) {
            logger.log("Failed to open model file for writing", true);
            return false;
        }

        // Save each layer's weights
        const auto& layers = transformer.getLayers();
        for (const auto& layer : layers) {
            layer->save(model_file);
        }

        logger.log("Model saved successfully");
        return true;
    } catch (const std::exception& e) {
        logger.log("Error saving model: " + std::string(e.what()), true);
        return false;
    }
}

bool ModelSaver::loadModel(Transformer& transformer, const std::string& directory,
                           const std::string& model_name) {
    try {
        std::string model_path = directory + "/" + model_name;
        logger.log("Loading model from: " + model_path);

        // Load and verify configuration
        TransformerConfig config;
        if (!readMetadata(directory, model_name, config)) {
            logger.log("Failed to read model metadata", true);
            return false;
        }

        // Verify configuration matches
        if (config != transformer.getConfig()) {
            logger.log("Model configuration mismatch", true);
            return false;
        }

        // Load model weights
        std::ifstream model_file(model_path + ".bin", std::ios::binary);
        if (!model_file) {
            logger.log("Failed to open model file for reading", true);
            return false;
        }

        // Load each layer's weights
        auto& layers = transformer.getLayers();
        for (auto& layer : layers) {
            layer->load(model_file);
        }

        logger.log("Model loaded successfully");
        return true;
    } catch (const std::exception& e) {
        logger.log("Error loading model: " + std::string(e.what()), true);
        return false;
    }
}

bool ModelSaver::saveCheckpoint(Transformer& transformer, const std::string& directory,
                                const std::string& model_name, size_t step, float loss) {
    std::string filepath = directory + "/" + model_name + "_step_" + 
                          std::to_string(step) + ".pt";
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Save model parameters
    // Implementation details here
    
    file.close();
    return true;
}

bool ModelSaver::loadCheckpoint(Transformer& transformer, const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Load model parameters
    // Implementation details here
    
    file.close();
    return true;
}

bool ModelSaver::loadLatestCheckpoint(Transformer& transformer, const std::string& directory,
                                      const std::string& model_name, int& epoch, float& loss) {
    try {
        // Find latest checkpoint
        int latest_epoch = -1;
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.path().extension() == ".ckpt") {
                std::string filename = entry.path().stem().string();
                if (filename.find(model_name + "_checkpoint_") == 0) {
                    int checkpoint_epoch =
                        std::stoi(filename.substr(filename.find_last_of("_") + 1));
                    latest_epoch = std::max(latest_epoch, checkpoint_epoch);
                }
            }
        }

        if (latest_epoch == -1) {
            logger.log("No checkpoints found", true);
            return false;
        }

        // Load the latest checkpoint
        std::string checkpoint_file = getCheckpointFilename(directory, model_name, latest_epoch);
        if (!loadCheckpoint(transformer, checkpoint_file)) {
            return false;
        }

        // Update epoch and loss from the loaded checkpoint
        std::ifstream ckpt_file(checkpoint_file, std::ios::binary);
        size_t meta_size;
        ckpt_file.read(reinterpret_cast<char*>(&meta_size), sizeof(meta_size));

        std::string meta_str(meta_size, '\0');
        ckpt_file.read(&meta_str[0], meta_size);

        json checkpoint_meta = json::parse(meta_str);
        epoch = checkpoint_meta["epoch"];
        loss = checkpoint_meta["loss"];

        return true;
    } catch (const std::exception& e) {
        logger.log("Error loading latest checkpoint: " + std::string(e.what()), true);
        return false;
    }
}

std::string ModelSaver::createDirectory(const std::string& base_dir) const {
    fs::path dir_path(base_dir);
    fs::create_directories(dir_path);
    return dir_path.string();
}

std::string ModelSaver::getCheckpointFilename(const std::string& directory,
                                              const std::string& model_name, int epoch) const {
    return directory + "/" + model_name + "_checkpoint_" + std::to_string(epoch) + ".ckpt";
}

bool ModelSaver::writeMetadata(const std::string& directory, const std::string& model_name,
                               const TransformerConfig& config) const {
    json meta;
    meta["model_name"] = model_name;
    meta["vocab_size"] = config.vocab_size;
    meta["hidden_size"] = config.hidden_size;
    meta["num_heads"] = config.num_heads;
    meta["num_layers"] = config.num_layers;
    meta["use_flash_attention"] = config.use_flash_attention;
    meta["use_rope"] = config.use_rope;
    meta["use_sliding_window"] = config.use_sliding_window;
    meta["window_size"] = config.window_size;

    std::ofstream meta_file(directory + "/" + model_name + ".meta.json");
    meta_file << std::setw(4) << meta << std::endl;
    return true;
}

bool ModelSaver::readMetadata(const std::string& directory, const std::string& model_name,
                              TransformerConfig& config) const {
    std::ifstream meta_file(directory + "/" + model_name + ".meta.json");
    if (!meta_file) {
        return false;
    }

    json meta;
    meta_file >> meta;

    config.vocab_size = meta["vocab_size"];
    config.hidden_size = meta["hidden_size"];
    config.num_heads = meta["num_heads"];
    config.num_layers = meta["num_layers"];
    config.use_flash_attention = meta["use_flash_attention"];
    config.use_rope = meta["use_rope"];
    config.use_sliding_window = meta["use_sliding_window"];
    config.window_size = meta["window_size"];

    return true;
}

void ModelSaver::save_vocabulary(const std::string& path, const Vocabulary& vocab) {
    try {
        // Save special token mappings in consistent order
        special_tokens_json = {
            {"<pad>", 0},
            {"<unk>", 1},
            {"<bos>", 2},
            {"<eos>", 3},
            {"<mask>", 4}
        };

        // Write to file
        std::ofstream file(path);
        if (!file) {
            logger.log("Failed to open vocabulary file for writing: " + path, true);
            return;
        }
        file << std::setw(4) << special_tokens_json << std::endl;
    } catch (const std::exception& e) {
        logger.log("Error saving vocabulary: " + std::string(e.what()), true);
    }
}