// fmt
#include <fmt/format.h>

// kintera
#include <kintera/thermo/relative_humidity.hpp>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/mesh/meshblock.hpp>

#include "output_type.hpp"
#include "output_utils.hpp"

namespace snap {

void OutputType::loadDiagOutputData(MeshBlockImpl* pmb, Variables const& vars) {
  OutputData* pod;
  auto peos = pmb->phydro->peos;
  auto pcoord = pmb->pcoord;

  if (ContainVariable("thermo") && pmb->phydro->options->eos()->thermo()) {
    auto const& w = vars.at("hydro_w");

    auto m = pmb->named_modules()["hydro.eos.thermo"];
    auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(m);
    kintera::ThermoX thermo_x(thermo_y->options);
    thermo_x->to(w.device());

    int ny = thermo_y->options->species().size() - 1;
    auto temp = peos->compute("W->T", {w});
    auto dens = w[IDN];
    auto pres = w[IPR];
    auto yfrac = w.narrow(0, ICY, ny);
    auto xfrac = thermo_y->compute("Y->X", {yfrac});

    // mole concentration [mol/m^3]
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});

    // volumetric entropy [J/(m^3 K)]
    auto entropy_vol = thermo_x->compute("TPV->S", {temp, pres, conc});

    // volumetric heat capacity [J/(m^3 K)]
    auto cp_vol = thermo_x->compute("TV->cp", {temp, conc});

    // molar entropy [J/(mol K)]
    auto entropy_mole = entropy_vol / conc.sum(-1);

    // molar heat capacity [J/(mol K)]
    auto cp_mole = cp_vol / conc.sum(-1);

    // mean molecular weight [kg/mol]
    auto mu = (thermo_x->mu * xfrac).sum(-1);

    // specific entropy [J/(kg K)]
    auto entropy = entropy_mole / mu;

    // potential temperature [K]
    auto theta = (entropy_vol / cp_vol).exp();

    // virtual potential temperature [K]
    auto Rgas = kintera::constants::Rgas / mu;
    auto feps = pres / (dens * Rgas * temp);
    auto theta_v = theta * feps;

    // relative humidity
    auto rh = kintera::relative_humidity(temp, conc, thermo_x->stoich,
                                         thermo_x->options->nucleation());

    // temperature
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "temp";
    pod->data.CopyFromTensor(temp);
    AppendOutputDataNode(pod);
    num_vars_ += 1;

    // potential temperature
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "theta";
    pod->data.CopyFromTensor(theta);
    AppendOutputDataNode(pod);
    num_vars_ += 1;

    // virtual potential temperature
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "theta_v";
    pod->data.CopyFromTensor(theta);
    AppendOutputDataNode(pod);
    num_vars_ += 1;

    // entropy
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "entropy";
    pod->data.CopyFromTensor(entropy);
    AppendOutputDataNode(pod);
    num_vars_ += 1;

    // relative humidity
    auto reactions = thermo_x->options->nucleation()->reactions();
    for (int i = 0; i < reactions.size(); ++i) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = fmt::format("rh_{}", reactions[i].products().begin()->first);
      // replace special characters '(' ')' and ',' with '_'
      for (char& c : pod->name) {
        if (c == '(' || c == ')' || c == ',') {
          c = '_';
        }
      }
      pod->data.CopyFromTensor(rh.select(-1, i));

      AppendOutputDataNode(pod);
      num_vars_ += 1;
    }
  }

  // implicit correction
  if (ContainVariable("implicit")) {
    auto du = pmb->phydro->named_buffers()["M"];

    // density
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "ic_dry";
    pod->data.InitFromTensor(du, 4, IDN, 1);
    AppendOutputDataNode(pod);
    num_vars_++;

    // momentum
    pod = new OutputData;
    pod->type = "VECTORS";
    pod->name = "ic_mom";
    pod->data.InitFromTensor(du, 4, IVX, 3);

    AppendOutputDataNode(pod);
    num_vars_ += 3;

    // total energy
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "ic_etot";
    pod->data.InitFromTensor(du, 4, IPR, 1);

    AppendOutputDataNode(pod);
    num_vars_++;

    // vapor + cloud
    auto ny = peos->nvar() - 5;
    if (ny > 0) {
      pod = new OutputData;
      pod->type = "VECTORS";
      pod->name = get_hydro_names(pmb, "ic_");
      pod->data.InitFromTensor(du, 4, ICY, ny);

      AppendOutputDataNode(pod);
      num_vars_ += ny;
    }
  }

  // vapor and cloud paths
  auto vol = pcoord->cell_volume();
  if (ContainVariable("path")) {
    auto const& u = vars.at("hydro_u");
    auto area = pcoord->face_area1();
    auto ny = peos->nvar() - 5;
    int il = pcoord->il();

    if (ny > 0) {
      pod = new OutputData;
      pod->type = "VECTORS";
      pod->name = get_hydro_names(pmb, "path_");
      auto u_sum = (u * vol).narrow(0, ICY, ny).sum(-1) / area.select(-1, il);

      pod->data.CopyFromTensor(u_sum);
      AppendOutputDataNode(pod);
      num_vars_ += ny;
    }
  }

  // zonal mean profiles
  if (ContainVariable("avg")) {
    auto layout = pmb->get_layout();
    c10d::ReduceOptions opsum;
    opsum.reduceOp = c10d::ReduceOp::SUM;
    opsum.rootRank = layout->options->process_root_rank();

    auto hydro_w_tol = vars.at("hydro_w") * vol;
    std::vector<at::Tensor> sum1 = {hydro_w_tol.sum({1, 2})};
    if (layout->has_process_group()) {
      layout->comm->pg->reduce(sum1, opsum)->wait();
    }

    std::vector<at::Tensor> sum2 = {vol.unsqueeze(0).sum({1, 2})};
    if (layout->has_process_group()) {
      layout->comm->pg->reduce(sum2, opsum)->wait();
    }
    auto avg_w = sum1[0] / sum2[0];

    // density
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "avg_rho";
    pod->data.InitFromTensor(avg_w, 2, IDN, 1);
    AppendOutputDataNode(pod);
    num_vars_++;

    // velocity vector
    pod = new OutputData;
    pod->type = "VECTORS";
    pod->name = "avg_vel";
    pod->data.InitFromTensor(avg_w, 2, IVX, 3);
    AppendOutputDataNode(pod);
    num_vars_ += 3;

    // pressure
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "avg_press";
    pod->data.InitFromTensor(avg_w, 2, IPR, 1);
    AppendOutputDataNode(pod);
    num_vars_++;

    auto ny = peos->nvar() - 5;
    if (ny > 0) {
      pod = new OutputData;
      pod->type = "VECTORS";
      pod->name = get_hydro_names(pmb, "avg_");
      pod->data.CopyFromTensor(avg_w.narrow(0, ICY, ny));
      AppendOutputDataNode(pod);
      num_vars_ += ny;
    }
  }
}
}  // namespace snap
