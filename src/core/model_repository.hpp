#pragma once

#include "core/algorithm_types.hpp"

namespace markloc::algorithm {

struct RepositoryOptions {
    bool requireAllViews = false;
    bool loadParameters = true;
};

class ModelRepository {
public:
    explicit ModelRepository(RepositoryOptions options = {});

    ModelSnapshot load(const Path& root, std::string* error = nullptr) const;
    DatasetIndex index(const Path& imageRoot) const;
    static bool readGray(const Path& path, cv::Mat& gray, std::string* error = nullptr);

private:
    RepositoryOptions options_;
};

} // namespace markloc::algorithm
