// snap
#include <snap/snap.h>

// snap
#include <snap/mesh/meshblock.hpp>

using namespace snap;

int main(int argc, char** argv) {
  std::string input_file = argc > 1 ? argv[1] : "shock.yaml";
  auto op = MeshBlockOptionsImpl::from_yaml(input_file);
  auto block = MeshBlock(op);

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available() && op->layout()->backend() == "nccl") {
    std::cout << "Running on CUDA" << std::endl;
    auto layout = block->get_layout();
    if (layout->has_process_group() &&
        layout->comm->pg->getBoundDeviceId().has_value()) {
      device = layout->comm->pg->getBoundDeviceId().value();
    } else {
      device = torch::Device(torch::kCUDA, op->layout()->local_rank());
    }
  }

  block->to(device);

  // initial conditions
  auto pcoord = block->pcoord;
  auto peos = block->phydro->peos;

  // coordinates
  auto grids = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto x1v = grids[2];
  auto x2v = grids[1];
  auto x3v = grids[0];

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = peos->nvar();

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  w[IDN] = torch::where(x1v < 0, 1.0, 0.125);
  w[IPR] = torch::where(x1v < 0, 1.0, 0.1);

  std::map<std::string, torch::Tensor> vars;

  // internal boundary
  auto r1 = torch::sqrt(x1v * x1v + x2v * x2v + x3v * x3v);
  auto solid = torch::where(r1 < 0.1, 1, 0).to(torch::kBool);

  vars["hydro_w"] = w;
  vars["solid"] = solid;

  block->initialize(vars);

  double current_time = 0.;
  block->make_outputs(vars, current_time);

  while (!block->pintg->stop(block->cycle++, current_time)) {
    auto dt = block->max_time_step(vars);
    block->print_cycle_info(vars, current_time, dt);

    // main loop
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(vars, dt, stage);
    }

    int err = block->check_redo(vars);
    if (err > 0) continue;  // redo this step with smaller dt
    if (err < 0) break;     // terminate simulation

    // make outputs
    current_time += dt;
    block->make_outputs(vars, current_time);
  }

  block->finalize(vars, current_time);
}
