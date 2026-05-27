// C/C++
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

// kintera
#include <kintera/utils/serialize.hpp>

// snap
#include <snap/mesh/meshblock.hpp>

#include "output_formats.hpp"
#include "output_utils.hpp"

namespace snap {

namespace {
//! \brief Write tensors to a temporary file and atomically rename it into
//! place, so a crash mid-write can never corrupt an existing restart file.
void save_tensors_atomic(Variables const& vars, std::string const& final_path) {
  std::string tmp_path = final_path + ".tmp";
  kintera::save_tensors(vars, tmp_path);
  std::error_code ec;
  std::filesystem::rename(tmp_path, final_path, ec);
  if (ec) {
    std::error_code rm_ec;
    std::filesystem::remove(tmp_path, rm_ec);
    throw std::runtime_error("Failed to finalize restart file '" + final_path +
                             "': " + ec.message());
  }
}
}  // namespace

RestartOutput::RestartOutput(OutputOptions const& options_)
    : OutputType(options_) {
  // Combined output (a single bundled ".restart" per dump) is the default.
  // Set `combine: false` in the output block to write one ".restart" per rank
  // with no cross-rank barrier or serial bundling (parallel-friendly at scale).
}

void RestartOutput::write_output_file(MeshBlockImpl* pmb, Variables const& vars,
                                      double current_time, bool final_write) {
  // make a cpu copy of variables
  Variables out_vars;
  bool skip_scalar_r = vars.count("scalar_s") && vars.count("hydro_u");
  for (auto const& [name, var] : vars) {
    if (skip_scalar_r && name == "scalar_r") {
      continue;
    }
    if (var.defined()) {
      out_vars[name] = var.to(torch::kCPU);
    }
  }

  // store last time and cycle
  out_vars["last_time"] = torch::tensor({current_time}, torch::kFloat64);
  out_vars["last_cycle"] = torch::tensor({(int64_t)pmb->cycle}, torch::kInt64);

  // save file number and next time for each output type
  std::vector<int> output_file_numbers;
  std::vector<double> output_next_times;

  for (auto out : pmb->output_types) {
    output_file_numbers.push_back(out->file_number);
    output_next_times.push_back(out->next_time);
  }

  out_vars["file_number"] = torch::tensor(output_file_numbers, torch::kInt64);
  out_vars["next_time"] = torch::tensor(output_next_times, torch::kFloat64);

  // shared stem: <output_dir>/<basename>.<blockid>.<file_number|final>
  char number[6];
  snprintf(number, sizeof(number), "%05d", file_number);
  char blockid[12];
  snprintf(blockid, sizeof(blockid), "block%d", pmb->options->layout()->rank());

  std::string stem;
  stem.assign(pmb->options->output_dir());
  stem.append("/");
  stem.append(pmb->options->basename());
  stem.append(".");
  stem.append(blockid);
  stem.append(".");
  stem.append(final_write ? "final" : number);

  // ensure the output directory exists
  std::error_code ec;
  std::filesystem::create_directories(pmb->options->output_dir(), ec);
  if (ec) {
    throw std::runtime_error("Failed to create output directory '" +
                             pmb->options->output_dir() + "': " + ec.message());
  }

  if (options->combine()) {
    // Each rank writes a per-block ".part"; the root rank bundles all parts
    // into one combined ".restart" file.
    save_tensors_atomic(out_vars, stem + ".part");
    combine_blocks(pmb, final_write);
  } else {
    // File-per-rank: each rank writes its own ".restart" directly. On restart
    // every rank loads its own file (the block id is rewritten to its rank).
    save_tensors_atomic(out_vars, stem + ".restart");
  }
}

}  // namespace snap
