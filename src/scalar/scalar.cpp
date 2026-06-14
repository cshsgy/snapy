// snap
#include "scalar.hpp"

#include <snap/hydro/hydro.hpp>
#include <snap/layout/layout.hpp>
#include <snap/mesh/meshblock.hpp>

namespace snap {
ScalarImpl::ScalarImpl(const ScalarOptions& options_, torch::nn::Module* p)
    : options(options_) {
  pmb = dynamic_cast<MeshBlockImpl const*>(p);
  reset();
}

void ScalarImpl::reset() {
  if (nvar() <= 0) return;

  TORCH_CHECK(pmb != nullptr, "[Scalar] Parent MeshBlock is null");
  TORCH_CHECK(options->riemann()->type() == "upwind",
              "[Scalar] Passive scalars currently require the upwind solver");

  pcoord = pmb->pcoord;
  precon = ReconstructImpl::create(options->recon(), this);
  priemann = RiemannSolverImpl::create(options->riemann(), this);

  if (options->thermo()) {
    pthermo = kintera::ThermoXImpl::create(options->thermo(), this);
  }
  if (options->kinetics()) {
    pkinetics = kintera::KineticsImpl::create(options->kinetics(), this);
  }

  auto nc1 = pcoord->options->nc1();
  auto nc2 = pcoord->options->nc2();
  auto nc3 = pcoord->options->nc3();

  _flux1 = register_buffer(
      "F1", nc1 > 1 ? torch::zeros({nvar(), nc3, nc2, nc1}, torch::kFloat64)
                    : torch::Tensor());
  _flux2 = register_buffer(
      "F2", nc2 > 1 ? torch::zeros({nvar(), nc3, nc2, nc1}, torch::kFloat64)
                    : torch::Tensor());
  _flux3 = register_buffer(
      "F3", nc3 > 1 ? torch::zeros({nvar(), nc3, nc2, nc1}, torch::kFloat64)
                    : torch::Tensor());
  _div = register_buffer(
      "D", torch::zeros({nvar(), nc3, nc2, nc1}, torch::kFloat64));
}

torch::Tensor ScalarImpl::forward(double dt, torch::Tensor u,
                                  Variables const& other) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  TORCH_CHECK(pmb != nullptr, "[Scalar] Parent MeshBlock is null");
  TORCH_CHECK(other.count("hydro_u"),
              "[Scalar] hydro_u is required for passive scalar transport");

  auto rho = other.at("hydro_u")[IDN].unsqueeze(0);
  auto r = u / rho;

  if (_flux1.defined()) {
    auto rlr = precon->forward(r, DIM1);
    _flux1.set_(
        priemann->forward(rlr[ILT], rlr[IRT], DIM1, pmb->phydro->flux1()[IDN]));
  }

  auto playout = pmb->get_layout();
  torch::Tensor rtmp2, rtmp3;
  Variables send_vars2, send_vars3;
  SyncOptions sync_opts;
  sync_opts.cross_panel_only(true).interpolate(false).type(kScalar);
  std::vector<CommWorkPtr> works;

  if (_flux2.defined()) {
    rtmp2 = precon->forward(r, DIM2);
    if (playout->options->type() == "cubed-sphere") {
      send_vars2["scalar_rlr"] =
          rtmp2.view({2 * nvar(), rtmp2.size(2), rtmp2.size(3), rtmp2.size(4)});
      pmb->begin_exchange(send_vars2, sync_opts.dim(DIM2));
    }
  }

  if (_flux3.defined()) {
    rtmp3 = precon->forward(r, DIM3);
    if (playout->options->type() == "cubed-sphere") {
      send_vars3["scalar_rlr"] =
          rtmp3.view({2 * nvar(), rtmp3.size(2), rtmp3.size(3), rtmp3.size(4)});
      pmb->begin_exchange(send_vars3, sync_opts.dim(DIM3));
    }
  }

  if (playout->options->type() == "cubed-sphere") {
    bool exchange_dim2 = _flux2.defined();
    bool exchange_dim3 = _flux3.defined();
    if (exchange_dim2) {
      pmb->launch_exchange(sync_opts.dim(DIM2), works);
    }
    if (exchange_dim3) {
      pmb->launch_exchange(sync_opts.dim(DIM3), works);
    }
    if (exchange_dim2)
      pmb->finalize_exchange(send_vars2, sync_opts.dim(DIM2), works);
    if (exchange_dim3)
      pmb->finalize_exchange(send_vars3, sync_opts.dim(DIM3), works);
  }

  if (_flux2.defined()) {
    _flux2.set_(priemann->forward(rtmp2[ILT], rtmp2[IRT], DIM2,
                                  pmb->phydro->flux2()[IDN]));
  }

  if (_flux3.defined()) {
    _flux3.set_(priemann->forward(rtmp3[ILT], rtmp3[IRT], DIM3,
                                  pmb->phydro->flux3()[IDN]));
  }

  _div.set_(pcoord->divergence(_flux1, _flux2, _flux3));
  auto ds = torch::zeros_like(_div);
  auto interior = pmb->part({0, 0, 0}, PartOptions().exterior(false));
  ds.index(interior) = -dt * _div.index(interior);
  return ds;
}

std::shared_ptr<ScalarImpl> ScalarImpl::create(ScalarOptions const& opts,
                                               torch::nn::Module* p,
                                               std::string const& name) {
  TORCH_CHECK(p, "[Scalar] Parent module is null");
  TORCH_CHECK(opts, "[Scalar] Options pointer is null");

  return p->register_module(name, Scalar(opts, p));
}

}  // namespace snap
