// C/C++
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// base
#include <configure.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/vectorize.hpp>

#include "netcdf_utils.hpp"
#include "output_formats.hpp"
#include "output_utils.hpp"

// Only proceed if NETCDF output enabled
#ifdef NETCDFOUTPUT

// External library headers
#include <netcdf.h>

#endif  // NETCDFOUTPUT

namespace snap {
NetcdfOutput::NetcdfOutput(OutputOptions const &options_)
    : OutputType(options_) {}

void NetcdfOutput::write_output_file(MeshBlockImpl *pmb_in,
                                     Variables const &vars, double current_time,
                                     bool final_write) {
  // skip final write if specified
  if (final_write) return;

#ifdef NETCDFOUTPUT
#define SNAP_NETCDF_CHECK(call)                                      \
  do {                                                               \
    int status__ = (call);                                           \
    if (status__ != NC_NOERR) {                                      \
      throw std::runtime_error(std::string(#call) +                  \
                               " failed: " + nc_strerror(status__)); \
    }                                                                \
  } while (false)

  auto pmb = LoadOutputData(pmb_in, vars);
  int rank = pmb->options->layout()->rank();

  auto pmeta = MetadataTable::GetInstance();
  auto phydro = pmb->phydro;

  int nc1 = pmb->options->coord()->nc1();
  int nc2 = pmb->options->coord()->nc2();
  int nc3 = pmb->options->coord()->nc3();
  int nghost = pmb->options->coord()->nghost();

  // set start/end array indices depending on whether ghost zones are included
  out_is = nc1 > 1 ? nghost : 0;
  out_ie = nc1 > 1 ? nc1 - nghost - 1 : 0;
  out_js = nc2 > 1 ? nghost : 0;
  out_je = nc2 > 1 ? nc2 - nghost - 1 : 0;
  out_ks = nc3 > 1 ? nghost : 0;
  out_ke = nc3 > 1 ? nc3 - nghost - 1 : 0;

  // FIXME: include_ghost zones probably doesn't work with grids other than
  // CCC
  if (options->include_ghost_zones()) {
    if (out_is != out_ie) {
      out_is -= nghost;
      out_ie += nghost;
    }

    if (out_js != out_je) {
      out_js -= nghost;
      out_je += nghost;
    }

    if (out_ks != out_ke) {
      out_ks -= nghost;
      out_ke += nghost;
    }
  }

  if (!TransformOutputData(pmb)) {
    ClearOutputData();
    if (options->combine()) combine_blocks(pmb, final_write);
    if (pmb != pmb_in) delete pmb;
    return;
  }

  // create filename: <basename>.<blockid>.<fileid>.<XXXXX>.nc
  // file_number
  std::string fname;
  char number[6];
  snprintf(number, sizeof(number), "%05d", file_number);
  char blockid[12];
  snprintf(blockid, sizeof(blockid), "block%d", rank);

  fname.assign(pmb->options->output_dir());
  fname.append("/");
  fname.append(pmb->options->basename());
  fname.append(".");
  fname.append(blockid);
  fname.append(".");
  fname.append(options->file_id());
  fname.append(".");
  fname.append(number);
  fname.append(".nc");

  // 1. open file for output
  std::stringstream msg;
  int ifile;

  std::error_code ec;
  std::filesystem::create_directories(pmb->options->output_dir(), ec);
  if (ec) {
    throw std::runtime_error("Failed to create output directory '" +
                             pmb->options->output_dir() + "': " + ec.message());
  }
  SNAP_NETCDF_CHECK(nc_create(fname.c_str(), NC_NETCDF4, &ifile));

  // 2. coordinate structure
  int ncells1 = out_ie - out_is + 1;
  int ncells2 = out_je - out_js + 1;
  int ncells3 = out_ke - out_ks + 1;

  int nfaces1 = ncells1;
  if (ncells1 > 1) nfaces1++;
  int nfaces2 = ncells2;
  if (ncells2 > 1) nfaces2++;
  int nfaces3 = ncells3;
  if (ncells3 > 1) nfaces3++;

  // 2. define coordinate
  int idt, idx1, idx2, idx3, idx1f, idx2f, idx3f, iray;
  // time
  nc_def_dim(ifile, "time", NC_UNLIMITED, &idt);

  nc_def_dim(ifile, "x1", ncells1, &idx1);
  if (ncells1 > 1) nc_def_dim(ifile, "x1f", nfaces1, &idx1f);

  nc_def_dim(ifile, "x2", ncells2, &idx2);
  if (ncells2 > 1) nc_def_dim(ifile, "x2f", nfaces2, &idx2f);

  nc_def_dim(ifile, "x3", ncells3, &idx3);
  if (ncells3 > 1) nc_def_dim(ifile, "x3f", nfaces3, &idx3f);

  // 3. define variables
  auto layout = pmb->get_layout();

  int level = 0;
  auto [lx2, lx3, lx1] = layout->loc_of(rank);

  int nb1 = layout->options->pz();
  int nb2 = layout->options->px();
  int nb3 = layout->options->py();

  int face = 0;
  if (layout->options->type() == "cubed-sphere") {
    face = lx1;
    lx1 = 0;
    lx2 += (face % 3) * nb2;
    lx3 += (face / 3) * nb3;
    nb2 *= 3;
    nb3 *= 2;
  }
  if (options->x1_slice()) {
    lx1 = 0;
    nb1 = 1;
  }
  if (options->x2_slice()) {
    lx2 = 0;
    nb2 = 1;
  }
  if (options->x3_slice()) {
    lx3 = 0;
    nb3 = 1;
  }

  int ivt, ivx1, ivx2, ivx3, ivx1f, ivx2f, ivx3f, imu, iphi;
  int loc[4] = {lx1, lx3, lx2, level};
  int pos[4];

  // NC_FLOAT keeps ~7 significant digits: a relative change below ~1e-7 in a
  // field, or a time step below ~1e-7 * t, does not show in the file.
  // `double_precision: true` on the output block writes NC_DOUBLE.
  const bool nc_dbl = options->double_precision();
  const nc_type nctype = nc_dbl ? NC_DOUBLE : NC_FLOAT;

  nc_def_var(ifile, "time", nctype, 1, &idt, &ivt);
  nc_put_att_text(ifile, ivt, "axis", 1, "T");
  nc_put_att_text(ifile, ivt, "units", 1, "s");
  nc_put_att_text(ifile, ivt, "long_name", 4, "time");

  nc_def_var(ifile, "x1", nctype, 1, &idx1, &ivx1);
  nc_put_att_text(ifile, ivx1, "axis", 1, "Z");
  nc_put_att_text(ifile, ivx1, "units", 1, "m");
  nc_put_att_text(ifile, ivx1, "long_name", 27, "Z-coordinate at cell center");

  pos[0] = 1;
  pos[1] = ncells1 * nb1;
  pos[2] = ncells1 * loc[0] + 1;
  pos[3] = ncells1 * (loc[0] + 1);
  nc_put_att_int(ifile, ivx1, "domain_decomposition", NC_INT, 4, pos);

  if (ncells1 > 1) {
    nc_def_var(ifile, "x1f", nctype, 1, &idx1f, &ivx1f);
    nc_put_att_text(ifile, ivx1f, "units", 1, "m");
    nc_put_att_text(ifile, ivx1f, "long_name", 25, "Z-coordinate at cell face");
    pos[0]--;
    pos[2]--;
    nc_put_att_int(ifile, ivx1f, "domain_decomposition", NC_INT, 4, pos);
  }

  nc_def_var(ifile, "x2", nctype, 1, &idx2, &ivx2);
  nc_put_att_text(ifile, ivx2, "axis", 1, "X");
  nc_put_att_text(ifile, ivx2, "units", 1, "m");
  nc_put_att_text(ifile, ivx2, "long_name", 27, "X-coordinate at cell center");

  pos[0] = 1;
  pos[1] = ncells2 * nb2;
  pos[2] = ncells2 * loc[2] + 1;
  pos[3] = ncells2 * (loc[2] + 1);
  nc_put_att_int(ifile, ivx2, "domain_decomposition", NC_INT, 4, pos);

  if (ncells2 > 1) {
    nc_def_var(ifile, "x2f", nctype, 1, &idx2f, &ivx2f);
    nc_put_att_text(ifile, ivx2f, "units", 1, "m");
    nc_put_att_text(ifile, ivx2f, "long_name", 25, "Y-coordinate at cell face");
    pos[0]--;
    pos[2]--;
    nc_put_att_int(ifile, ivx2f, "domain_decomposition", NC_INT, 4, pos);
  }

  nc_def_var(ifile, "x3", nctype, 1, &idx3, &ivx3);
  nc_put_att_text(ifile, ivx3, "axis", 1, "Y");
  nc_put_att_text(ifile, ivx3, "units", 1, "m");
  nc_put_att_text(ifile, ivx3, "long_name", 27, "Y-coordinate at cell center");

  pos[0] = 1;
  pos[1] = ncells3 * nb3;
  pos[2] = ncells3 * loc[1] + 1;
  pos[3] = ncells3 * (loc[1] + 1);
  nc_put_att_int(ifile, ivx3, "domain_decomposition", NC_INT, 4, pos);

  if (ncells3 > 1) {
    nc_def_var(ifile, "x3f", nctype, 1, &idx3f, &ivx3f);
    nc_put_att_text(ifile, ivx3f, "units", 1, "m");
    nc_put_att_text(ifile, ivx3f, "long_name", 25, "X-coordinate at cell face");
    pos[0]--;
    pos[2]--;
    nc_put_att_int(ifile, ivx3f, "domain_decomposition", NC_INT, 4, pos);
  }

  int nbtotal = nb1 * nb2 * nb3;
  nc_put_att_int(ifile, NC_GLOBAL, "NumFilesInSet", NC_INT, 1, &nbtotal);

  OutputData *pdata = pfirst_data_;

  // count total variables (vector variables are expanded into flat scalars)
  int total_vars = 0;
  while (pdata != nullptr) {
    auto names = Vectorize<std::string>(pdata->name.c_str(), ";");
    std::string grid = pmeta->GetGridType(names[0]);
    int nvar = get_num_variables(grid, pdata->data);

    total_vars += nvar;
    pdata = pdata->pnext;
  }

  int iaxis[4] = {idt, idx1, idx3, idx2};
  int iaxis1[4] = {idt, idx1f, idx3, idx2};
  int iaxis2[4] = {idt, idx1, idx3, idx2f};
  int iaxis3[4] = {idt, idx1, idx3f, idx2};
  int iaxisr[4] = {idt, iray, idx3, idx2};
  int iaxis_23[3] = {idt, idx3, idx2};
  int *var_ids = new int[total_vars];
  int *ivar = var_ids;

  pdata = pfirst_data_;
  while (pdata != nullptr) {
    auto names = Vectorize<std::string>(pdata->name.c_str(), ";");
    std::string grid = pmeta->GetGridType(names[0]);
    int nvar = get_num_variables(grid, pdata->data);

    std::vector<std::string> varnames;
    if (names.size() >= nvar) {
      for (int n = 0; n < nvar; ++n) {
        varnames.push_back(names[n]);
      }
    } else {
      for (int n = 0; n < nvar; ++n) {
        size_t pos = pdata->name.find('?');
        if (nvar == 1) {                     // SCALARS
          if (pos < pdata->name.length()) {  // find '?'
            varnames.push_back(pdata->name.substr(0, pos) +
                               pdata->name.substr(pos + 1));
          } else {
            varnames.push_back(pdata->name);
          }
        } else {  // VECTORS
          char c[16];
          snprintf(c, sizeof(c), "%d", n + 1);
          if (pos < pdata->name.length()) {  // find '?'
            varnames.push_back(pdata->name.substr(0, pos) + c +
                               pdata->name.substr(pos + 1));
          } else {
            varnames.push_back(pdata->name + c);
          }
        }
      }
    }

    for (int n = 0; n < nvar; ++n) {
      auto const &raw_name = varnames[n];
      auto name = sanitize_netcdf_name(raw_name);

      if (grid == "CCF" && ncells1 > 1)
        SNAP_NETCDF_CHECK(
            nc_def_var(ifile, name.c_str(), nctype, 4, iaxis1, ivar));
      else if ((grid == "CFC") && (ncells2 > 1))
        SNAP_NETCDF_CHECK(
            nc_def_var(ifile, name.c_str(), nctype, 4, iaxis2, ivar));
      else if ((grid == "FCC") && (ncells3 > 1))
        SNAP_NETCDF_CHECK(
            nc_def_var(ifile, name.c_str(), nctype, 4, iaxis3, ivar));
      else if (grid == "--C")
        SNAP_NETCDF_CHECK(
            nc_def_var(ifile, name.c_str(), nctype, 2, iaxis, ivar));
      else if (grid == "-CC")
        SNAP_NETCDF_CHECK(
            nc_def_var(ifile, name.c_str(), nctype, 3, iaxis_23, ivar));
      else if (grid == "--F")
        SNAP_NETCDF_CHECK(
            nc_def_var(ifile, name.c_str(), nctype, 2, iaxis1, ivar));
      else if (grid == "---")
        SNAP_NETCDF_CHECK(
            nc_def_var(ifile, name.c_str(), nctype, 1, iaxis, ivar));
      else
        SNAP_NETCDF_CHECK(
            nc_def_var(ifile, name.c_str(), nctype, 4, iaxis, ivar));

      // set units
      auto attr = pmeta->GetUnits(raw_name);
      if (attr != "") {
        nc_put_att_text(ifile, *ivar, "units", attr.length(), attr.c_str());
      }

      // set long_name
      attr = pmeta->GetLongName(raw_name);
      if (attr != "") {
        nc_put_att_text(ifile, *ivar, "long_name", attr.length(), attr.c_str());
      }

      ivar++;
    }
    pdata = pdata->pnext;
  }

  SNAP_NETCDF_CHECK(nc_enddef(ifile));

  // 4. write variables
  const size_t nbuf = (size_t)nfaces1 * nfaces3 * nfaces2;
  // One buffer of the selected type. Each computed double is assigned once.
  // The default path casts to float here, so an overflow stays +inf. NetCDF
  // is not asked to narrow, because that stores the float fill value.
  struct SelBuf {
    bool dbl;
    double *d = nullptr;
    float *f = nullptr;
    size_t n = 0;
    SelBuf(bool dbl, size_t nbuf) : dbl(dbl) {
      if (dbl)
        d = new double[nbuf]();
      else
        f = new float[nbuf]();
    }
    ~SelBuf() {
      delete[] d;
      delete[] f;
    }
    SelBuf(const SelBuf &) = delete;
    SelBuf &operator=(const SelBuf &) = delete;
    void reset() { n = 0; }
    void set(size_t i, double v) {
      if (dbl)
        d[i] = v;
      else
        f[i] = static_cast<float>(v);
    }
    void push(double v) { set(n++, v); }
    void put_var(int ncid, int varid) const {
      if (dbl)
        nc_put_var_double(ncid, varid, d);
      else
        nc_put_var_float(ncid, varid, f);
    }
    void put_vara(int ncid, int varid, const size_t *start,
                  const size_t *count) const {
      if (dbl)
        nc_put_vara_double(ncid, varid, start, count, d);
      else
        nc_put_vara_float(ncid, varid, start, count, f);
    }
  } buf(nc_dbl, nbuf);
  size_t start[4] = {0, 0, 0, 0};
  size_t count[4] = {1, (size_t)ncells1, (size_t)ncells3, (size_t)ncells2};
  size_t count1[4] = {1, (size_t)nfaces1, (size_t)ncells3, (size_t)ncells2};
  size_t count2[4] = {1, (size_t)ncells1, (size_t)nfaces3, (size_t)ncells2};
  size_t count3[4] = {1, (size_t)ncells1, (size_t)ncells3, (size_t)nfaces2};
  size_t count_23[3] = {1, (size_t)ncells3, (size_t)ncells2};

  double timed = current_time;
  float timef = (float)current_time;
  if (nc_dbl)
    nc_put_vara_double(ifile, ivt, start, count, &timed);
  else
    nc_put_vara_float(ifile, ivt, start, count, &timef);

  int coord_is = options->x1_slice() ? islice : out_is;
  int coord_ie = options->x1_slice() ? islice : out_ie;
  int coord_js = options->x2_slice() ? jslice : out_js;
  int coord_je = options->x2_slice() ? jslice : out_je;
  int coord_ks = options->x3_slice() ? kslice : out_ks;
  int coord_ke = options->x3_slice() ? kslice : out_ke;

  for (int i = coord_is; i <= coord_ie; ++i)
    buf.set(i - coord_is, pmb->pcoord->x1v[i].item<double>());
  buf.put_var(ifile, ivx1);

  if (ncells1 > 1) {
    for (int i = coord_is; i <= coord_ie + 1; ++i)
      buf.set(i - coord_is, pmb->pcoord->x1f[i].item<double>());
    buf.put_var(ifile, ivx1f);
  }

  for (int j = coord_js; j <= coord_je; ++j) {
    buf.set(j - coord_js,
            pmb->pcoord->x2v[j].item<double>() + (face % 3) * M_PI / 2.);
  }
  buf.put_var(ifile, ivx2);

  if (ncells2 > 1) {
    for (int j = coord_js; j <= coord_je + 1; ++j) {
      buf.set(j - coord_js,
              pmb->pcoord->x2f[j].item<double>() + (face % 3) * M_PI / 2.);
    }
    buf.put_var(ifile, ivx2f);
  }

  for (int k = coord_ks; k <= coord_ke; ++k) {
    buf.set(k - coord_ks,
            pmb->pcoord->x3v[k].item<double>() + (face / 3) * M_PI / 2.);
  }
  buf.put_var(ifile, ivx3);

  if (ncells3 > 1) {
    for (int k = coord_ks; k <= coord_ke + 1; ++k) {
      buf.set(k - coord_ks,
              pmb->pcoord->x3f[k].item<double>() + (face / 3) * M_PI / 2.);
    }
    buf.put_var(ifile, ivx3f);
  }

  ivar = var_ids;
  pdata = pfirst_data_;
  while (pdata != nullptr) {
    auto names = Vectorize<std::string>(pdata->name.c_str(), ",");
    std::string grid = pmeta->GetGridType(names[0]);
    int nvar = get_num_variables(grid, pdata->data);

    if (grid == "CCF" && ncells1 > 1) {
      for (int n = 0; n < nvar; n++) {
        buf.reset();
        for (int i = out_is; i <= out_ie + 1; ++i)
          for (int k = out_ks; k <= out_ke; ++k)
            for (int j = out_js; j <= out_je; ++j)
              buf.push(pdata->data(n, k, j, i));
        buf.put_vara(ifile, *ivar, start, count1);
        ++ivar;
      }
    } else if ((grid == "CFC") && (ncells2 > 1)) {
      for (int n = 0; n < nvar; n++) {
        buf.reset();
        for (int i = out_is; i <= out_ie; ++i)
          for (int k = out_ks; k <= out_ke; ++k)
            for (int j = out_js; j <= out_je + 1; ++j)
              buf.push(pdata->data(n, k, j, i));
        buf.put_vara(ifile, *ivar, start, count2);
        ++ivar;
      }
    } else if ((grid == "FCC") && (ncells3 > 1)) {
      for (int n = 0; n < nvar; n++) {
        buf.reset();
        for (int i = out_is; i <= out_ie; ++i)
          for (int k = out_ks; k <= out_ke + 1; ++k)
            for (int j = out_js; j <= out_je; ++j)
              buf.push(pdata->data(n, k, j, i));
        buf.put_vara(ifile, *ivar, start, count3);
        ++ivar;
      }
    } else if (grid == "--C") {
      for (int n = 0; n < nvar; n++) {
        buf.reset();
        for (int i = out_is; i <= out_ie; ++i) buf.push(pdata->data(n, i));
        buf.put_vara(ifile, *ivar, start, count);
        ++ivar;
      }
    } else if (grid == "-CC") {
      for (int n = 0; n < nvar; n++) {
        buf.reset();
        for (int k = out_ks; k <= out_ke; ++k)
          for (int j = out_js; j <= out_je; ++j) buf.push(pdata->data(n, k, j));
        buf.put_vara(ifile, *ivar, start, count_23);
        ++ivar;
      }
    } else if (grid == "--F") {
      for (int n = 0; n < nvar; n++) {
        buf.reset();
        for (int i = out_is; i <= out_ie + 1; ++i) buf.push(pdata->data(n, i));
        buf.put_vara(ifile, *ivar, start, count1);
        ++ivar;
      }
    } else if (grid == "---") {
      for (int n = 0; n < nvar; n++) {
        buf.reset();
        buf.push(pdata->data(n));
        buf.put_vara(ifile, *ivar, start, count);
        ++ivar;
      }
    } else {
      for (int n = 0; n < nvar; n++) {
        buf.reset();
        for (int i = out_is; i <= out_ie; ++i)
          for (int k = out_ks; k <= out_ke; ++k)
            for (int j = out_js; j <= out_je; ++j)
              buf.push(pdata->data(n, k, j, i));
        buf.put_vara(ifile, *ivar, start, count);
        ++ivar;
      }
    }

    // doesn't work
    // nc_put_att_text(ifile, *(ivar-1), "output",
    //  output_params.variable.length(), output_params.variable.c_str());
    pdata = pdata->pnext;
  }

  // 5. close nc file
  SNAP_NETCDF_CHECK(nc_close(ifile));

  ClearOutputData();  // required when LoadOutputData() is used.
  delete[] var_ids;

  if (options->combine()) {
    combine_blocks(pmb, final_write);
  }

  if (pmb != pmb_in) delete pmb;
#endif  // NETCDFOUTPUT
}
}  // namespace snap
