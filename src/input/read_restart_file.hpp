#pragma once

// C/C++
#include <cstdio>
#include <cstring>
#include <string>

// torch
#include <torch/script.h>
#include <torch/torch.h>

// snap
#include <snap/layout/layout.hpp>

namespace snap {

using Variables = std::map<std::string, torch::Tensor>;

Variables load_restart(std::string const& path, int block_rank = get_rank());

//! \brief Resolve a restart path for a specific block rank.
//!
//! File-per-rank (uncombined) dumps embed a ".block<N>." token in the filename;
//! this rewrites <N> to `block_rank` so every rank loads its own file. Paths
//! without that token (combined bundles or single dumps) are returned
//! unchanged.
std::string restart_path_for_rank(std::string const& path, int block_rank);

}  // namespace snap
