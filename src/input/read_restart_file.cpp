// C/C++
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// kintera
#include <kintera/utils/serialize.hpp>

// snap
#include <snap/layout/layout.hpp>

namespace fs = std::filesystem;

namespace snap {
namespace {
constexpr char kRestartBundleMagic[] = "SNAPY_RESTART_BUNDLE_V1";
}

// -------------------------
// Small helpers
// -------------------------

struct RestartFields {
  std::string basename;
  std::string blockid;
  std::string filenumber;
};

static RestartFields parse_part_filename(const std::string& name) {
  constexpr std::string_view suffix = ".part";

  if (name.size() <= suffix.size() ||
      name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
    throw std::invalid_argument("filename does not end with .part");
  }

  // Strip ".part"
  const std::string_view core(name.data(), name.size() - suffix.size());

  // Find last two dots
  const size_t dot2 = core.rfind('.');
  if (dot2 == std::string::npos) {
    throw std::invalid_argument("filename missing filenumber field");
  }

  const size_t dot1 = core.rfind('.', dot2 - 1);
  if (dot1 == std::string::npos) {
    throw std::invalid_argument("filename missing block_id field");
  }

  RestartFields out;
  out.basename = std::string(core.substr(0, dot1));
  out.blockid = std::string(core.substr(dot1 + 1, dot2 - dot1 - 1));
  out.filenumber = std::string(core.substr(dot2 + 1));

  if (out.basename.empty() || out.blockid.empty() || out.filenumber.empty()) {
    throw std::invalid_argument("one or more filename fields are empty");
  }

  return out;
}

static bool ends_with(std::string const& s, std::string const& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool is_restart_bundle(std::string const& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;

  std::string magic;
  std::getline(in, magic);
  return magic == kRestartBundleMagic;
}

// Create a unique temp file path; not bulletproof, but good enough
static fs::path make_temp_path(std::string_view suffix) {
  fs::path dir = fs::temp_directory_path();

  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  for (int tries = 0; tries < 20; ++tries) {
    uint64_t r = dis(gen);
    std::ostringstream name;
    name << "tmp_" << std::hex << r << suffix;
    fs::path p = dir / name.str();
    if (!fs::exists(p)) return p;
  }

  // Fallback (very unlikely to collide)
  return dir / ("tmp_fallback" + std::string(suffix));
}

struct RestartBundleEntry {
  std::string name;
  std::uintmax_t size;
  std::streamoff offset;
};

static std::vector<RestartBundleEntry> read_restart_bundle_index(
    std::ifstream& in) {
  std::string line;
  std::getline(in, line);  // magic
  if (line != kRestartBundleMagic) {
    throw std::invalid_argument("restart bundle missing expected magic header");
  }

  std::getline(in, line);
  if (line.empty()) {
    throw std::invalid_argument("restart bundle missing entry count");
  }
  auto entry_count = static_cast<size_t>(std::stoull(line));

  std::vector<RestartBundleEntry> entries;
  entries.reserve(entry_count);
  for (size_t i = 0; i < entry_count; ++i) {
    std::getline(in, line);
    auto tab = line.find('\t');
    if (tab == std::string::npos) {
      throw std::invalid_argument("restart bundle index line missing tab");
    }

    RestartBundleEntry entry;
    entry.name = line.substr(0, tab);
    entry.size = static_cast<std::uintmax_t>(std::stoull(line.substr(tab + 1)));
    entries.push_back(std::move(entry));
  }

  std::getline(in, line);
  if (!line.empty()) {
    throw std::invalid_argument("restart bundle header missing terminator");
  }

  auto payload_offset = in.tellg();
  std::streamoff running = 0;
  for (auto& entry : entries) {
    entry.offset = payload_offset + running;
    running += static_cast<std::streamoff>(entry.size);
  }

  return entries;
}

static Variables load_pt_from_bundle(std::string const& path, int block_rank) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << path << ": failed to open restart bundle\n";
    return {};
  }

  std::vector<RestartBundleEntry> entries;
  try {
    entries = read_restart_bundle_index(in);
  } catch (std::exception const& exc) {
    std::cerr << path << ": failed to parse restart bundle: " << exc.what()
              << "\n";
    return {};
  }

  for (auto const& entry : entries) {
    if (!ends_with(entry.name, ".part")) continue;

    auto out = parse_part_filename(entry.name);
    int rank = std::stoi(out.blockid.substr(5, out.blockid.size() - 5));
    if (rank != block_rank) continue;

    fs::path tmp_path = make_temp_path(".part");
    std::ofstream out_file(tmp_path, std::ios::binary);
    if (!out_file) {
      std::cerr << path << ": failed to create temp file for restart bundle\n";
      return {};
    }

    in.clear();
    in.seekg(entry.offset);
    if (!in) {
      std::cerr << path << ": failed to seek to restart bundle payload\n";
      std::error_code ec;
      fs::remove(tmp_path, ec);
      return {};
    }

    std::vector<char> buffer(1 << 20);
    std::uintmax_t remaining = entry.size;
    while (remaining > 0) {
      auto chunk = static_cast<std::streamsize>(
          std::min<std::uintmax_t>(remaining, buffer.size()));
      in.read(buffer.data(), chunk);
      auto count = in.gcount();
      if (count <= 0) {
        std::cerr << path << ": truncated restart bundle payload\n";
        out_file.close();
        std::error_code ec;
        fs::remove(tmp_path, ec);
        return {};
      }
      out_file.write(buffer.data(), count);
      remaining -= static_cast<std::uintmax_t>(count);
    }
    out_file.flush();
    out_file.close();

    auto vars = kintera::load_tensors(tmp_path.string());
    std::error_code ec;
    fs::remove(tmp_path, ec);
    return vars;
  }

  std::cerr << path << ": no matching block found in restart bundle\n";
  return {};
}

std::string restart_path_for_rank(std::string const& path, int block_rank) {
  fs::path p(path);
  std::string fname = p.filename().string();

  static const std::string tok = ".block";
  size_t pos = fname.find(tok);
  if (pos == std::string::npos) return path;

  size_t dstart = pos + tok.size();
  size_t dend = dstart;
  while (dend < fname.size() &&
         std::isdigit(static_cast<unsigned char>(fname[dend]))) {
    ++dend;
  }

  // Require at least one digit followed by a '.', e.g. ".block3." — this avoids
  // rewriting an unrelated basename that merely contains the text "block".
  if (dend == dstart || dend >= fname.size() || fname[dend] != '.') {
    return path;
  }

  std::string rewritten =
      fname.substr(0, dstart) + std::to_string(block_rank) + fname.substr(dend);
  p.replace_filename(rewritten);
  return p.string();
}

Variables load_restart(std::string const& path, int block_rank) {
  // Dispatch based on whether `path` is a restart bundle or a single tensor
  // dump.
  if (is_restart_bundle(path)) {
    return load_pt_from_bundle(path, block_rank);
  } else {
    // Treat as a single .part TorchScript file
    return kintera::load_tensors(path);
  }

  return {};
}

}  // namespace snap
