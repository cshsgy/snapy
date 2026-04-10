// C/C++ headers
#include <glob.h>

#include <cstdio>
#include <map>
#include <mutex>

// snap
#include <snap/mesh/meshblock.hpp>

#include "output_formats.hpp"

namespace snap {
namespace {
std::mutex combine_mutex;
std::map<std::string, int> combine_counts;

bool ready_to_combine(Layout const &layout, std::string const &key) {
  std::lock_guard<std::mutex> lock(combine_mutex);
  int &count = combine_counts[key];
  count += 1;
  if (count < layout->options->blocks_per_process()) {
    return false;
  }
  combine_counts.erase(key);
  return true;
}
}  // namespace

// execute tar command to create an archive
int make_tar_archive(std::string const &archive_name,
                     std::vector<std::string> const &file_list) {
  std::string command = "tar -cf " + archive_name;
  for (auto const &f : file_list) {
    command += " " + f;
  }
  return std::system(command.c_str());
}

void RestartOutput::combine_blocks(MeshBlockImpl *pmb, bool final_write) {
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

    remove(outfile.c_str());
    err = make_tar_archive(outfile, file_list);

    if (err) {
      msg << "### FATAL ERROR in function [RestartOutput::combine_blocks]"
          << std::endl
          << "make_tar_archive() failed with error " << err << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }

    globfree(&glob_result);

    // remove input part files
    for (auto const &f : file_list) {
      remove(f.c_str());
    }
  }
}

}  // namespace snap
