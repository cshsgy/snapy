// C/C++ headers
#include <glob.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>  // stringstream
#include <stdexcept>

// base
#include <configure.h>

// snap
#include <snap/mesh/meshblock.hpp>

#include "output_formats.hpp"

int mppnccombine(int argc, char *argv[]);

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

void NetcdfOutput::combine_blocks(MeshBlockImpl *pmb, bool) {
// Only proceed if NETCDF output enabled
#ifdef NETCDFOUTPUT
  auto layout = pmb->get_layout();
  char number[64];
  snprintf(number, sizeof(number), "%05d", file_number);

  std::string key;
  key.assign(pmb->options->output_dir());
  key.append("|");
  key.append(pmb->options->basename());
  key.append("|");
  key.append(options->file_id());
  key.append("|");
  key.append(number);
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
    infile.append(options->file_id());
    infile.append(".");
    infile.append(number);
    infile.append(".nc");

    glob_t glob_result;
    int err = glob(infile.c_str(), GLOB_TILDE, NULL, &glob_result);
    if (err != 0) {
      globfree(&glob_result);
      msg << "### FATAL ERROR in function [NetcdfOutput::combine_blocks]"
          << std::endl
          << "glob() failed with error " << err << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }

    std::string outfile;
    outfile.assign(pmb->options->output_dir());
    outfile.append("/");
    outfile.append(pmb->options->basename());
    outfile.append(".");
    outfile.append(options->file_id());
    outfile.append(".");
    outfile.append(number);
    outfile.append(".nc");

    int argc = 3 + glob_result.gl_pathc;
    // char argv[][2048] = {"CombineBlocks", "-r", outfile.c_str(),
    // infile.c_str()};
    char **argv = new char *[argc];
    for (int i = 0; i < argc; ++i) argv[i] = new char[2048];
    snprintf(argv[0], 2048, "%s", "CombineBlocks");
    snprintf(argv[1], 2048, "%s", "-r");
    snprintf(argv[2], 2048, "%s", outfile.c_str());
    for (int i = 3; i < argc; ++i)
      snprintf(argv[i], 2048, "%s", glob_result.gl_pathv[i - 3]);

    remove(outfile.c_str());
    err = mppnccombine(argc, argv);
    if (err) {
      msg << "### FATAL ERROR in function [NetcdfOutput::combine_blocks]"
          << std::endl
          << "mppnccombine failed with error " << err << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }

    globfree(&glob_result);
    for (int i = 0; i < argc; ++i) delete[] argv[i];
    delete[] argv;
  }

#endif  // NETCDFOUTPUT
}
}  // namespace snap
