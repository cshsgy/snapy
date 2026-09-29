// C/C++
#include <algorithm>
#include <array>
#include <string>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/forcing/forcing.hpp>
#include <snap/utils/log.hpp>

#include "hydro.hpp"

namespace snap {
HydroOptions HydroOptionsImpl::from_yaml(std::string const& filename,
                                         bool verbose) {
  auto op = HydroOptionsImpl::create();

  // ------------ equation of state ------------ //
  op->eos() = EquationOfStateOptionsImpl::from_yaml(filename, verbose);
  if (verbose) op->eos()->report(SINFO(HydroOptions));

  // ------------- reconstruction ------------ //
  op->recon1() = ReconstructOptionsImpl::from_yaml(filename, "vertical");
  if (verbose) op->recon1()->report(SINFO(HydroOptions : vertical));

  op->recon23() = ReconstructOptionsImpl::from_yaml(filename, "horizontal");
  if (verbose) op->recon23()->report(SINFO(HydroOptions : horizontal));

  // ------------ riemann solver ------------ //
  op->riemann() = RiemannSolverOptionsImpl::from_yaml(filename, "dynamics");
  if (verbose) op->riemann()->report(SINFO(HydroOptions));

  // ---------- implicit correction --------- //
  op->icorr() = ImplicitOptionsImpl::from_yaml(filename);
  if (op->icorr() && verbose) op->icorr()->report(SINFO(HydroOptions));

  // ------------ sedimentation ------------- //
  op->sed() = SedHydroOptionsImpl::from_yaml(filename);
  if (op->sed() && verbose) op->sed()->report(SINFO(HydroOptions));

  // -------------- others ------------------ //
  auto config = YAML::LoadFile(filename);
  auto dyn = config["dynamics"];
  if (dyn) {
    // Every key here is read by a presence check, so an unknown key -- a
    // typo, or an option that no longer exists -- would be silently ignored
    // and the run would proceed as if it had been applied. Reject it instead.
    // Only the TOP level is checked here: the equation-of-state sub-block is
    // co-owned (kintera reads its own keys from it), so it is checked against
    // both libraries' keys in EquationOfStateOptionsImpl::from_yaml.
    static std::array<char const*, 8> const dynamics_keys = {
        "equation-of-state", "reconstruct",     "riemann-solver",
        "verbose",           "disable-flux-x1", "disable-flux-x2",
        "disable-flux-x3",   "wb-wall-clamp"};
    for (auto const& item : dyn) {
      auto key = item.first.as<std::string>();
      auto joined = [] {  // the message lists the checked keys
        std::string s;
        for (auto const* k : dynamics_keys)
          s += (s.empty() ? "" : ", ") + std::string(k);
        return s;
      };
      TORCH_CHECK(std::find(dynamics_keys.begin(), dynamics_keys.end(), key) !=
                      dynamics_keys.end(),
                  "HydroOptions: unknown key 'dynamics/", key,
                  "'. Valid keys: ", joined(), ".");
    }
    op->verbose() = dyn["verbose"].as<bool>(verbose);
    op->disable_flux_x1() = dyn["disable-flux-x1"].as<bool>(false);
    op->disable_flux_x2() = dyn["disable-flux-x2"].as<bool>(false);
    op->disable_flux_x3() = dyn["disable-flux-x3"].as<bool>(false);
    op->wb_wall_clamp() = dyn["wb-wall-clamp"].as<bool>(true);
  }

  // --------------- forcings --------------- //
  auto forcing = config["forcing"];
  if (!forcing) return op;

  // Same rationale as the dynamics sweep above.
  for (auto const& item : forcing) {
    auto key = item.first.as<std::string>();
    TORCH_CHECK(key != "fric-heat",
                "HydroOptions: 'forcing/fric-heat' has been removed. The x1 "
                "face gravity work already carries the sedimentation channel "
                "of the mass flux, so this key would apply the "
                "precipitation potential-energy release twice. Delete it.");
    TORCH_CHECK(
        key == "const-gravity" || key == "coriolis" || key == "diffusion" ||
            key == "body-heat" || key == "top-cool" || key == "bot-heat" ||
            key == "relax-bot-comp" || key == "relax-bot-temp" ||
            key == "relax-bot-velo" || key == "top-sponge-lyr" ||
            key == "bot-sponge-lyr" || key == "plume-forcing",
        "HydroOptions: unknown key 'forcing/", key,
        "'. Valid keys: const-gravity, coriolis, diffusion, body-heat, "
        "top-cool, bot-heat, relax-bot-comp, relax-bot-temp, relax-bot-velo, "
        "top-sponge-lyr, bot-sponge-lyr, plume-forcing.");
  }

  op->grav() = ConstGravityOptionsImpl::from_yaml(forcing);
  if (op->grav()) {
    if (op->disable_flux_x1()) op->grav()->grav1(0.);
    if (op->disable_flux_x2()) op->grav()->grav2(0.);
    if (op->disable_flux_x3()) op->grav()->grav3(0.);
    op->grav()->report(SINFO(HydroOptions));
  }

  op->coriolis() = CoriolisOptionsImpl::from_yaml(forcing);
  if (op->coriolis()) op->coriolis()->report(SINFO(HydroOptions));

  op->diffusion() = DiffusionOptionsImpl::from_yaml(forcing);
  if (op->diffusion()) op->diffusion()->report(SINFO(HydroOptions));

  op->bodyHeat() = BodyHeatOptionsImpl::from_yaml(forcing);
  if (op->bodyHeat()) op->bodyHeat()->report(SINFO(HydroOptions));

  op->topCool() = TopCoolOptionsImpl::from_yaml(forcing);
  if (op->topCool()) op->topCool()->report(SINFO(HydroOptions));

  op->botHeat() = BotHeatOptionsImpl::from_yaml(forcing);
  if (op->botHeat()) op->botHeat()->report(SINFO(HydroOptions));

  op->relaxBotComp() = RelaxBotCompOptionsImpl::from_yaml(forcing);
  if (op->relaxBotComp()) op->relaxBotComp()->report(SINFO(HydroOptions));

  op->relaxBotTemp() = RelaxBotTempOptionsImpl::from_yaml(forcing);
  if (op->relaxBotTemp()) op->relaxBotTemp()->report(SINFO(HydroOptions));

  op->relaxBotVelo() = RelaxBotVeloOptionsImpl::from_yaml(forcing);
  if (op->relaxBotVelo()) op->relaxBotVelo()->report(SINFO(HydroOptions));

  op->topSpongeLyr() = TopSpongeLyrOptionsImpl::from_yaml(forcing);
  if (op->topSpongeLyr()) op->topSpongeLyr()->report(SINFO(HydroOptions));

  op->botSpongeLyr() = BotSpongeLyrOptionsImpl::from_yaml(forcing);
  if (op->botSpongeLyr()) op->botSpongeLyr()->report(SINFO(HydroOptions));

  if (op->eos()->type() == "plume-eos") {
    op->plumeForcing() = PlumeForcingOptionsImpl::from_yaml(forcing);
    if (op->plumeForcing()) op->plumeForcing()->report(SINFO(HydroOptions));
  }

  return op;
}

HydroOptions HydroOptionsImpl::clone() const {
  auto op = HydroOptionsImpl::create();

  op->verbose() = verbose();
  op->disable_flux_x1() = disable_flux_x1();
  op->disable_flux_x2() = disable_flux_x2();
  op->disable_flux_x3() = disable_flux_x3();
  op->rho_ref() = rho_ref();
  if (grav()) op->grav() = grav()->clone();
  if (coriolis()) op->coriolis() = coriolis()->clone();
  if (diffusion()) op->diffusion() = diffusion()->clone();
  if (bodyHeat()) op->bodyHeat() = bodyHeat()->clone();
  if (topCool()) op->topCool() = topCool()->clone();
  if (botHeat()) op->botHeat() = botHeat()->clone();
  if (relaxBotComp()) op->relaxBotComp() = relaxBotComp()->clone();
  if (relaxBotTemp()) op->relaxBotTemp() = relaxBotTemp()->clone();
  if (relaxBotVelo()) op->relaxBotVelo() = relaxBotVelo()->clone();
  if (topSpongeLyr()) op->topSpongeLyr() = topSpongeLyr()->clone();
  if (botSpongeLyr()) op->botSpongeLyr() = botSpongeLyr()->clone();
  if (plumeForcing()) op->plumeForcing() = plumeForcing()->clone();

  // TODO(cli)
  /*if (eos()) op->eos() = eos()->clone();
  if (recon1()) op->recon1() = recon1()->clone();
  if (recon23()) op->recon23() = recon23()->clone();
  if (riemann()) op->riemann() = riemann()->clone();
  if (icorr()) op->icorr() = icorr()->clone();
  if (sed()) op->sed() = sed()->clone();*/

  return op;
}

}  // namespace snap
