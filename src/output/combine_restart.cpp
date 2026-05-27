// C/C++ headers
#include <glob.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <vector>

// snap
#include <snap/mesh/meshblock.hpp>

#include "output_formats.hpp"

namespace snap {
namespace {
constexpr char kRestartBundleMagic[] = "SNAPY_RESTART_BUNDLE_V1";

std::mutex combine_mutex;
std::map<std::string, int> combine_counts;

bool ready_to_combine(Layout const& layout, std::string const& key) {
  std::lock_guard<std::mutex> lock(combine_mutex);
  int& count = combine_counts[key];
  count += 1;
  if (count < layout->options->blocks_per_process()) {
    return false;
  }
  combine_counts.erase(key);
  return true;
}
}  // namespace

int make_restart_bundle(std::string const& bundle_name,
                        std::vector<std::string> const& file_list) {
  namespace fs = std::filesystem;

  std::vector<std::uintmax_t> sizes;
  sizes.reserve(file_list.size());
  for (auto const& file : file_list) {
    sizes.push_back(fs::file_size(file));
  }

  std::ofstream out(bundle_name, std::ios::binary | std::ios::trunc);
  if (!out) return -1;

  out << kRestartBundleMagic << "\n";
  out << file_list.size() << "\n";
  for (size_t i = 0; i < file_list.size(); ++i) {
    out << fs::path(file_list[i]).filename().string() << "\t" << sizes[i]
        << "\n";
  }
  out << "\n";
  if (!out) return -1;

  std::vector<char> buffer(1 << 20);
  for (auto const& file : file_list) {
    std::ifstream in(file, std::ios::binary);
    if (!in) return -1;

    while (in) {
      in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
      auto count = in.gcount();
      if (count > 0) {
        out.write(buffer.data(), count);
      }
    }
    if (!out) return -1;
  }

  out.flush();
  return out ? 0 : -1;
}

void RestartOutput::combine_blocks(MeshBlockImpl* pmb, bool final_write) {
  auto layout = pmb->get_layout();
  char number[64];
  snprintf(number, sizeof(number), "%05d", file_number);

  std::string key;
  key.assign(pmb->options->output_dir());
  key.append("|");
  key.append(pmb->options->basename());
  key.append("|restart|");
  key.append(final_write ? "final" : number);
  if (!ready_to_combine(layout, key)) {
    return;
  }

  /*c10d::BarrierOptions op;
  op.device_ids = {layout->options->local_rank()};
  layout->pg->barrier(op)->wait();*/
  if (layout->has_process_group()) {
    layout->comm->pg->barrier()->wait();
  }

  std::stringstream msg;

  if (layout->options->process_rank() == layout->options->process_root_rank()) {
    std::string infile;
    infile.assign(pmb->options->output_dir());
    infile.append("/");
    infile.append(pmb->options->basename());
    infile.append(".block*.");
    if (final_write) {
      infile.append("final");
    } else {
      infile.append(number);
    }
    infile.append(".part");

    glob_t glob_result;
    int err = glob(infile.c_str(), GLOB_TILDE, NULL, &glob_result);
    if (err != 0) {
      globfree(&glob_result);
      msg << "### FATAL ERROR in function [RestartOutput::combine_blocks]"
          << std::endl
          << "glob() failed with error " << err << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }

    std::string outfile;
    outfile.assign(pmb->options->output_dir());
    outfile.append("/");
    outfile.append(pmb->options->basename());
    outfile.append(".");
    if (final_write) {
      outfile.append("final");
    } else {
      outfile.append(number);
    }
    outfile.append(".restart");

    std::vector<std::string> file_list;

    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
      file_list.push_back(std::string(glob_result.gl_pathv[i]));
    }

    if (file_list.size() == 1) {
      // std::rename atomically replaces an existing destination on POSIX.
      err = std::rename(file_list.front().c_str(), outfile.c_str());
    } else {
      // Bundle into a temp file, then atomically rename over the destination so
      // a crash mid-bundle leaves the previous ".restart" intact.
      std::string tmp = outfile + ".tmp";
      err = make_restart_bundle(tmp, file_list);
      if (err == 0) {
        std::error_code rn_ec;
        std::filesystem::rename(tmp, outfile, rn_ec);
        if (rn_ec) {
          std::error_code rm_ec;
          std::filesystem::remove(tmp, rm_ec);
          err = -1;
        }
      }
    }

    if (err) {
      msg << "### FATAL ERROR in function [RestartOutput::combine_blocks]"
          << std::endl
          << "restart bundling failed with error " << err << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }

    globfree(&glob_result);

    // remove input part files
    if (file_list.size() != 1) {
      for (auto const& f : file_list) {
        remove(f.c_str());
      }
    }
  }
}

}  // namespace snap
