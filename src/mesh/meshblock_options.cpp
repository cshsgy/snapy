// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/output/output_type.hpp>
#include <snap/utils/log.hpp>

#include "meshblock.hpp"

namespace snap {

MeshBlockOptions MeshBlockOptionsImpl::from_yaml(std::string input_file,
                                                 bool verbose) {
  auto op = MeshBlockOptionsImpl::create();

  // -------------- layout -------------- //
  op->layout() = LayoutOptionsImpl::from_yaml(input_file);

  // ------------- basename ------------- //
  // Extract filename from path using utility function, then remove extension
  std::string filename = get_filename(input_file);
  // Remove file extension
  size_t last_dot = filename.find_last_of('.');
  op->basename() =
      (last_dot == std::string::npos) ? filename : filename.substr(0, last_dot);
  if (verbose) {
    SINFO(MeshBlockOptions) << "basename = " << op->basename() << std::endl;
  }

  // -------------- hydro --------------- //
  op->hydro() = HydroOptionsImpl::from_yaml(input_file, verbose);
  if (verbose) op->hydro()->report(SINFO(MeshBlockOptions));

  // ------------- scalar --------------- //
  op->scalar() = ScalarOptionsImpl::from_yaml(input_file, verbose);
  if (op->scalar()->recon() == nullptr) {
    op->scalar()->recon() = op->hydro()->recon23();
  }
  if (verbose) op->scalar()->report(SINFO(MeshBlockOptions));

  // ------------- integrator ------------ //
  op->intg() = harp::IntegratorOptionsImpl::from_yaml(input_file);
  if (verbose) op->intg()->report(SINFO(MeshBlockOptions));

  auto config = YAML::LoadFile(input_file);
  op->verbose() = config["verbose"].as<bool>(verbose);

  // --------------- outputs ------------- //
  int fid = 0;
  if (config["outputs"]) {
    for (auto const &out_cfg : config["outputs"]) {
      op->outputs().push_back(OutputOptionsImpl::from_yaml(out_cfg, fid++));
    }

    if (verbose) {
      for (auto const &out_op : op->outputs()) {
        out_op->report(SINFO(MeshBlockOptions));
      }
    }
  }

  // ------------- coordinate ------------- //
  op->coord() = CoordinateOptionsImpl::from_yaml(input_file);
  for (auto const &out_op : op->outputs()) {
    auto check_slice = [](std::optional<double> const &slice, double lower,
                          double upper, char const *axis) {
      if (slice && (*slice < lower || *slice >= upper)) {
        throw std::invalid_argument(std::string("Slice at ") + axis + "=" +
                                    std::to_string(*slice) +
                                    " is out of range of Mesh");
      }
    };
    check_slice(out_op->x1_slice(), op->coord()->global_x1min(),
                op->coord()->global_x1max(), "x1");
    check_slice(out_op->x2_slice(), op->coord()->global_x2min(),
                op->coord()->global_x2max(), "x2");
    check_slice(out_op->x3_slice(), op->coord()->global_x3min(),
                op->coord()->global_x3max(), "x3");
  }
  if (verbose) op->coord()->report(SINFO(MeshBlockOptions));

  // --------- internal boundary ---------- //
  op->ib() = InternalBoundaryOptionsImpl::from_yaml(input_file);
  if (op->ib() && verbose) op->ib()->report(SINFO(MeshBlockOptions));

  // --------- external boundary ---------- //
  if (!config["boundary-condition"]) return op;
  if (!config["boundary-condition"]["external"]) return op;

  auto external_bc = config["boundary-condition"]["external"];

  if (op->coord()->nc1() > 1) {
    // x1-inner
    auto ix1 = external_bc["x1-inner"].as<std::string>("reflecting");
    if (ix1 == "periodic" && op->layout()->type() == "cubed") {
      op->layout()->periodic_z(true);
    }

    ix1 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix1) != get_bc_func().end(),
                "Boundary function '", ix1, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ix1]);

    if (verbose) {
      SINFO(MeshBlockOptions) << "x1-inner BC: " << ix1 << std::endl;
    }

    // x1-outer
    auto ox1 = external_bc["x1-outer"].as<std::string>("reflecting");
    if (ox1 == "periodic" && op->layout()->type() == "cubed") {
      op->layout()->periodic_z(true);
    }

    ox1 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox1) != get_bc_func().end(),
                "Boundary function '", ox1, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ox1]);

    if (verbose) {
      SINFO(MeshBlockOptions) << "x1-outer BC: " << ox1 << std::endl;
    }
  } else if (op->coord()->nc2() > 1 || op->coord()->nc3() > 1) {
    op->bfuncs().push_back(nullptr);
    op->bfuncs().push_back(nullptr);
  }

  if (op->coord()->nc2() > 1) {
    // x2-inner
    auto ix2 = external_bc["x2-inner"].as<std::string>("reflecting");
    if (ix2 == "periodic") op->layout()->periodic_x(true);

    ix2 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix2) != get_bc_func().end(),
                "Boundary function '", ix2, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ix2]);

    if (verbose) {
      SINFO(MeshBlockOptions) << "x2-inner BC: " << ix2 << std::endl;
    }

    // x2-outer
    auto ox2 = external_bc["x2-outer"].as<std::string>("reflecting");
    if (ox2 == "periodic") op->layout()->periodic_x(true);

    ox2 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox2) != get_bc_func().end(),
                "Boundary function '", ox2, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ox2]);

    if (verbose) {
      SINFO(MeshBlockOptions) << "x2-outer BC: " << ox2 << std::endl;
    }
  } else if (op->coord()->nc3() > 1) {
    op->bfuncs().push_back(nullptr);
    op->bfuncs().push_back(nullptr);
  }

  if (op->coord()->nc3() > 1) {
    // x3-inner
    auto ix3 = external_bc["x3-inner"].as<std::string>("reflecting");
    if (ix3 == "periodic") op->layout()->periodic_y(true);

    ix3 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix3) != get_bc_func().end(),
                "Boundary function '", ix3, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ix3]);

    if (verbose) {
      SINFO(MeshBlockOptions) << "x3-inner BC: " << ix3 << std::endl;
    }

    // x3-outer
    auto ox3 = external_bc["x3-outer"].as<std::string>("reflecting");
    if (ox3 == "periodic") op->layout()->periodic_y(true);

    ox3 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox3) != get_bc_func().end(),
                "Boundary function '", ox3, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ox3]);

    if (verbose) {
      SINFO(MeshBlockOptions) << "x3-outer BC: " << ox3 << std::endl;
    }
  }

  if (verbose) op->layout()->report(SINFO(MeshBlockOptions));

  return op;
}

bool MeshBlockOptionsImpl::is_physical_boundary(int dy, int dx, int dz) const {
  if (dy == -1) return bfuncs()[BoundaryFace::kInnerX3] != nullptr;
  if (dy == 1) return bfuncs()[BoundaryFace::kOuterX3] != nullptr;
  if (dx == -1) return bfuncs()[BoundaryFace::kInnerX2] != nullptr;
  if (dx == 1) return bfuncs()[BoundaryFace::kOuterX2] != nullptr;
  if (dz == -1) return bfuncs()[BoundaryFace::kInnerX1] != nullptr;
  if (dz == 1) return bfuncs()[BoundaryFace::kOuterX1] != nullptr;
  return false;
}

std::string MeshBlockOptionsImpl::device_str() const {
  if (!layout()) return "";

  if (layout()->device() == "cuda") {
    int device_index = layout()->device_id();
    if (device_index < 0) {
      device_index = layout()->local_rank();
    }
    return "cuda:" + std::to_string(device_index);
  } else if (layout()->device() == "cpu") {
    return "cpu";
  } else {
    return "";
  }
}

}  // namespace snap
