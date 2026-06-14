//! \brief Cubed-sphere layout implementation
/*!
 * Notes on connectivity:
 *  - We model only the surface (six 2D faces).
 *  - Each face holds a px-by-py processor grid (px==py required by the cubed
 * grid; however code allows px,py).
 *  - Global rank = face_number * (py*px) + zorder_rank_within_face.
 *
 * Orientation model across an edge:
 *   From (face f, side s \in {L,R,B,T}) we land on (nface, nside),
 *   and the along-edge index is either preserved or reversed.
 *   No "transpose" is required if nside is defined correctly:
 *     - neighbor side L/R varies along neighbor Y (rows)
 *     - neighbor side B/T varies along neighbor X (cols)
 *
 * Face naming:
 *   Faces are named according to the global cartesian +X,+Y,+Z,-X,-Y,-Z
 * directions Face names should be consistent with the numbering of faces in the
 * global cartesian coordinates.
 *
 *   Table `CS_FACE_EDGES[6][4]` and `CS_FACE_NAMES[6][3]` should be
 *   edited together to ensure consistency.
 *
 * Ghost zone communication:
 *   The main function to perform ghost zone communication is
 *    MeshBlock-owned begin/launch/finalize exchange path.
 *   The loc_of() function return a 3-item tuple (rx,ry,face) for a given rank.
 *   Position of a serialization/deserialization buffer is identified by
 *   get_buffer_id(offset), where offset is a 3-item tuple (dx,dy,0)
 *
 *   CubedSphereLayoutImpl::serialize() saves the ghost zone data into send
 * buffers. CubedSphereLayoutImpl::deserialize() loads the received data into
 * ghost zones.
 *
 *   For the cubed-sphere layout, only face-adjacent neighbors are considered.
 *   Within serialization, _covariant_to_cartesian() is called to convert
 *   covariant vector components to Cartesian components.
 *
 *   Within deserialization, _cartesian_to_covariant() is called to convert
 *   Cartesian vector components back to covariant components.
 *
 *   The cell-centered angular coordinates are in `pcoord->x2v` and
 * `pcoord->x3v`. Consult src/coord/coordinate.hpp,
 * src/coord/gnomonic_equiangular.cpp, for more details.
 */

// C/C++
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

// fmt
#include <fmt/format.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "connectivity.hpp"
#include "cubed_sphere_layout.hpp"

namespace snap {

namespace {

std::mutex g_cubed_sphere_comm_mutex;

std::tuple<int, int, int> find_peer_offset(CubedSphereLayoutImpl const& layout,
                                           int peer_rank, int target_rank) {
  auto peer_iloc = layout.loc_of(peer_rank);
  for (auto offset :
       {std::tuple<int, int, int>{-1, 0, 0}, std::tuple<int, int, int>{1, 0, 0},
        std::tuple<int, int, int>{0, -1, 0},
        std::tuple<int, int, int>{0, 1, 0}}) {
    if (layout.neighbor_rank(peer_iloc, offset) == target_rank) {
      return offset;
    }
  }

  TORCH_CHECK(false, "failed to find reverse cubed-sphere offset from rank ",
              peer_rank, " to rank ", target_rank);
}

}  // namespace

/*!
 * ----------------------------
 * Global cartesian coordinates
 * ----------------------------
 *
 *       +Z
 *       ^
 *       |
 *       |----> +Y
 *      /
 *  +X /
 *
 * ---------------------------
 * Local cartesian coordinates
 * ---------------------------
 *
 * (0) +X
 * (4) -Y
 * (3) +Z
 *
 *                                            z  y
 *                 ___________                | /
 *                 |\        .\               |/---x
 *                 | \   3   . \             (3)
 *                 |  \_________\
 *      y          | 4 |     .  |
 *      |          \. .|......  |
 *  z___|(4)        \  |    0 . |
 *     /             \ |       .|             y
 *   x/               \|________|             |
 *                                        (0) |___x
 *                                           /
 *                                         z/
 *
 *
 *  (1) +Y
 *  (2) -X
 *  (5) -Z
 *                                         y  x
 *                 __________              | /
 *                 |\       .\             |/___z
 *                 | \      . \           (1)
 *      y  z       |  \________\
 *      | /        |  |  2  .  |
 *  x___|/         |..|......  |
 *       (2)       \  |     . 1|        (5)  ___ x
 *                  \ |  5   . |           /|
 *                   \|_______.|         y/ |
 *                                          z
 *
 * --------------------------
 * Cubed-sphere connectivity
 * --------------------------
 * Face numbering (editable):
 *
 *           -------
 *           |  3  |
 *     |-----|-----|-----|-----|
 *     |  4  |  0  |  1  |  2  |
 *     |-----|-----|-----|-----|
 *           |  5  |
 *           |-----|
 *
 *
 * --------------
 * Side numbering
 * --------------
 *   Side_L = 0
 *   Side_R = 1
 *   Side_B = 2
 *   Side_T = 3
 *
 * -------------------------------
 * Local face orientation and sides
 * -------------------------------
 *
 *         (T,3)          beta (y,3)
 *        |-----|         ^
 *  (L,0) |  X  | (R,1)   |
 *        |-----|         |----> alpha (x,2)
 *         (B,2)
 *
 * IMPORTANT: Different codes choose different local face axes.
 * If your tests show flipped corner order, toggle `rev` for that edge.
 */

// face 0: +X
// face 1: +Y
// face 2: -X
// face 3: +Z
// face 4: -Y
// face 5: -Z
const char CS_FACE_NAMES[6][3] = {"+X", "+Y", "-X", "+Z", "-Y", "-Z"};

/*!
 * Each entry says: on face F, the global velocity component VEL{1,2,3}
 * corresponds to local component idx with sign sgn.
 */
const CSVel CS_CART_TO_LOCAL_VEL[6][3] = {
    /* face 0: */
    [0] = {/* VEL1 */ {VEL3, +1},
           /* VEL2 */ {VEL1, +1},
           /* VEL3 */ {VEL2, +1}},
    /* face 1: */
    [1] = {/* VEL1 */ {VEL3, +1},
           /* VEL2 */ {VEL2, -1},
           /* VEL3 */ {VEL1, +1}},
    /* face 2: */
    [2] = {/* VEL1 */ {VEL3, +1},
           /* VEL2 */ {VEL1, -1},
           /* VEL3 */ {VEL2, -1}},
    /* face 3: */
    [3] = {/* VEL1 */ {VEL1, +1},
           /* VEL2 */ {VEL3, -1},
           /* VEL3 */ {VEL2, +1}},
    /* face 4: */
    [4] = {/* VEL1 */ {VEL3, +1},
           /* VEL2 */ {VEL2, +1},
           /* VEL3 */ {VEL1, -1}},
    /* face 5: */
    [5] = {/* VEL1 */ {VEL1, -1},
           /* VEL2 */ {VEL3, +1},
           /* VEL3 */ {VEL2, +1}}};

/*!
 * Each entry says: on face F, the local velocity component VEL_{Z,X,Y}
 * corresponds to global component idx with sign sgn.
 */
const CSVel CS_LOCAL_TO_CART_VEL[6][3] = {
    /* face 0: */
    [0] = {/* VEL1 */ {VEL2, +1},
           /* VEL2 */ {VEL3, +1},
           /* VEL3 */ {VEL1, +1}},
    /* face 1: */
    [1] = {/* VEL1 */ {VEL3, +1},
           /* VEL2 */ {VEL2, -1},
           /* VEL3 */ {VEL1, +1}},
    /* face 2: */
    [2] = {/* VEL1 */ {VEL2, -1},
           /* VEL2 */ {VEL3, -1},
           /* VEL3 */ {VEL1, +1}},
    /* face 3: */
    [3] = {/* VEL1 */ {VEL1, +1},
           /* VEL2 */ {VEL3, +1},
           /* VEL3 */ {VEL2, -1}},
    /* face 4: */
    [4] = {/* VEL1 */ {VEL3, -1},
           /* VEL2 */ {VEL2, +1},
           /* VEL3 */ {VEL1, +1}},
    /* face 5: */
    [5] = {/* VEL1 */ {VEL1, -1},
           /* VEL2 */ {VEL3, +1},
           /* VEL3 */ {VEL2, +1}}};

const CSVel CS_G2L_VEL[6][3] = {
    /* face 0: */
    [0] = {/* VEL1 */ {VEL3, +1},
           /* VEL2 */ {VEL1, +1},
           /* VEL3 */ {VEL2, +1}},
    /* face 1: */
    [1] = {/* VEL1 */ {VEL3, +1},
           /* VEL2 */ {VEL2, -1},
           /* VEL3 */ {VEL1, +1}},
    /* face 2: */
    [2] = {/* VEL1 */ {VEL3, +1},
           /* VEL2 */ {VEL1, -1},
           /* VEL3 */ {VEL2, -1}},
    /* face 3: */
    [3] = {/* VEL1 */ {VEL1, +1},
           /* VEL2 */ {VEL3, -1},
           /* VEL3 */ {VEL2, +1}},
    /* face 4: */
    [4] = {/* VEL1 */ {VEL3, +1},
           /* VEL2 */ {VEL2, +1},
           /* VEL3 */ {VEL1, -1}},
    /* face 5: */
    [5] = {/* VEL1 */ {VEL1, -1},
           /* VEL2 */ {VEL3, +1},
           /* VEL3 */ {VEL2, +1}}};

const CSVel CS_L2G_VEL[6][3] = {
    /* face 0: */
    [0] = {/* VEL1 */ {VEL2, +1},
           /* VEL2 */ {VEL3, +1},
           /* VEL3 */ {VEL1, +1}},
    /* face 1: */
    [1] = {/* VEL1 */ {VEL3, +1},
           /* VEL2 */ {VEL2, -1},
           /* VEL3 */ {VEL1, +1}},
    /* face 2: */
    [2] = {/* VEL1 */ {VEL2, -1},
           /* VEL2 */ {VEL3, -1},
           /* VEL3 */ {VEL1, +1}},
    /* face 3: */
    [3] = {/* VEL1 */ {VEL1, +1},
           /* VEL2 */ {VEL3, +1},
           /* VEL3 */ {VEL2, -1}},
    /* face 4: */
    [4] = {/* VEL1 */ {VEL3, -1},
           /* VEL2 */ {VEL2, +1},
           /* VEL3 */ {VEL1, +1}},
    /* face 5: */
    [5] = {/* VEL1 */ {VEL1, -1},
           /* VEL2 */ {VEL3, +1},
           /* VEL3 */ {VEL2, +1}}};

/*!
 * Sides: 0=L, 1=R, 2=B, 3=T  (left, right, bottom, top)
 * Each entry says: leaving face F via side S,
 * you arrive at (nface, nside) and the along-edge index is reversed? (0/1)
 */
const CSEdge CS_FACE_EDGES[6][4] = {
    /* face 0: neighbors 4(L),1(R),5(B),3(T) */
    [0] = {/* L */ {4, SIDE_R, 0},
           /* R */ {1, SIDE_L, 0},
           /* B */ {5, SIDE_T, 0},
           /* T */ {3, SIDE_B, 0}},
    /* face 1: neighbors 0(L),2(R),5(B),3(T) */
    [1] = {/* L */ {0, SIDE_R, 0},
           /* R */ {2, SIDE_L, 0},
           /* B */ {5, SIDE_R, 1},
           /* T */ {3, SIDE_R, 0}},
    /* face 2: neighbors 1(L),4(R),5(B),3(T) */
    [2] = {/* L */ {1, SIDE_R, 0},
           /* R */ {4, SIDE_L, 0},
           /* B */ {5, SIDE_B, 1},
           /* T */ {3, SIDE_T, 1}},
    /* face 3: neighbors 4(L),1(R),0(B),2(T) */
    [3] = {/* L */ {4, SIDE_T, 1},
           /* R */ {1, SIDE_T, 0},
           /* B */ {0, SIDE_T, 0},
           /* T */ {2, SIDE_T, 1}},
    /* face 4: neighbors 2(L),0(R),5(B),3(T) */
    [4] = {/* L */ {2, SIDE_R, 0},
           /* R */ {0, SIDE_L, 0},
           /* B */ {5, SIDE_L, 0},
           /* T */ {3, SIDE_L, 1}},
    /* face 5: neighbors 4(L),1(R),2(B),0(T) */
    [5] = {/* L */ {4, SIDE_B, 0},
           /* R */ {1, SIDE_B, 1},
           /* B */ {2, SIDE_B, 1},
           /* T */ {0, SIDE_B, 0}}};

void populate_cs_l2g_vel(CSVel l2g[6][3]) {
  for (int f = 0; f < 6; ++f) {
    for (int c = 0; c < 3; ++c) {
      for (int gc = 0; gc < 3; ++gc) {
        if (CS_G2L_VEL[f][gc].idx == c) {
          l2g[f][c].idx = gc;
          l2g[f][c].sgn = CS_G2L_VEL[f][gc].sgn;
        }
      }
    }
  }
}

static inline int get_side(std::tuple<int, int, int> const& offset) {
  auto [dy, dx, _] = offset;
  if (dx == -1 && dy == 0)
    return SIDE_L;
  else if (dx == 1 && dy == 0)
    return SIDE_R;
  else if (dx == 0 && dy == -1)
    return SIDE_B;
  else if (dx == 0 && dy == 1)
    return SIDE_T;
  else
    return -1;  // invalid
}

static inline void cs_clamp_inside(int pxy, int* nx, int* ny) {
  if (*nx < 0)
    *nx = 0;
  else if (*nx >= pxy)
    *nx = pxy - 1;
  if (*ny < 0)
    *ny = 0;
  else if (*ny >= pxy)
    *ny = pxy - 1;
}

static inline void cs_edge_map_into_neighbor(int pxy, int leaving_side,
                                             int pos /*0..k-1*/,
                                             const CSEdge* emap, int* out_rx,
                                             int* out_ry) {
  // Map along-edge index into neighbor face border
  int pos2 = pos;
  if (emap->rev) {
    pos2 = pxy - 1 - pos;
  }

  switch (emap->nside) {
    case SIDE_L:
      *out_rx = 0;
      *out_ry = pos2;
      break; /* varies in y */
    case SIDE_R:
      *out_rx = pxy - 1;
      *out_ry = pos2;
      break;
    case SIDE_B:
      *out_rx = pos2;
      *out_ry = 0;
      break; /* varies in x */
    case SIDE_T:
      *out_rx = pos2;
      *out_ry = pxy - 1;
      break;
    default:
      *out_rx = 0;
      *out_ry = 0;
      break;
  }
}

void CubedSphereLayoutImpl::_initialize() {
  // build the ranks
  TORCH_CHECK(options->pz() == 1,
              "CubedSphereLayoutImpl: pz must be 1 for cubed-sphere layout");
  TORCH_CHECK(
      options->px() == options->py(),
      "CubedSphereLayoutImpl: px must equal py for cubed-sphere layout");

  int px = options->px();
  int py = options->py();

  _coords2.resize(px * py);
  build_zorder_coords2(pxy(), pxy(), _coords2.data());
  build_rank_of2(pxy(), pxy(), _coords2.data(), _rankof.data());

  // build backend
  _init_process_group();
}

void CubedSphereLayoutImpl::pretty_print(std::ostream& os) const {
  options->report(os);
  for (int f = 0; f < 6; ++f) {
    os << " Rank | (rx,ry;f)\n";
    os << "----------------\n";
    for (int r = 0; r < pxy() * pxy(); ++r) {
      int gr = _global_rank_from_face_local(f, r);
      os << fmt::format(" {:>3} | ({:>2},{:>2};{:>2})\n", gr, _coords2[r].x,
                        _coords2[r].y, f);
    }
  }
}

void CubedSphereLayoutImpl::_step_one(int face, int rx, int ry, int dx, int dy,
                                      int* out_face, int* out_rx,
                                      int* out_ry) const {
  /* Try to stay on-face */
  int nx = rx + dx;
  int ny = ry + dy;
  if (0 <= nx && nx < pxy() && 0 <= ny && ny < pxy()) {
    *out_face = face;
    *out_rx = nx;
    *out_ry = ny;
    return;
  }

  /* Identify which single edge is crossed */
  int side = -1;
  if (nx < 0)
    side = SIDE_L;
  else if (nx >= pxy())
    side = SIDE_R;
  else if (ny < 0)
    side = SIDE_B;
  else if (ny >= pxy())
    side = SIDE_T;

  const CSEdge emap = CS_FACE_EDGES[face][side];
  *out_face = emap.nface;

  /* Along-edge position on current face */
  int pos = (side == SIDE_L || side == SIDE_R) ? ry : rx;

  /* Map to neighbor border */
  cs_edge_map_into_neighbor(pxy(), side, pos, &emap, out_rx, out_ry);

  // cs_clamp_inside(pxy, out_rx, out_ry);
}

int CubedSphereLayoutImpl::rank_of(std::tuple<int, int, int> iloc) const {
  auto [rx, ry, face] = iloc;
  if (face < 0 || face >= 6) return -1;
  if (rx < 0 || rx >= pxy() || ry < 0 || ry >= pxy()) return -1;
  return _global_rank_from_face_local(face, _rankof[ry * pxy() + rx]);
}

std::tuple<int, int, int> CubedSphereLayoutImpl::loc_of(int global_rank) const {
  if (global_rank < 0 || global_rank >= 6 * pxy() * pxy()) return {-1, -1, -1};
  int face, r_local;
  _global_rank_to_face_local(global_rank, &face, &r_local);
  int rx = _coords2[r_local].x;
  int ry = _coords2[r_local].y;
  return {rx, ry, face};
}

/* get neighbor GLOBAL rank for (dx,dy) in {-1,0,1}^2 (incl. corners) */
int CubedSphereLayoutImpl::neighbor_rank(
    std::tuple<int, int, int> iloc, std::tuple<int, int, int> offset) const {
  auto [rx, ry, face] = iloc;
  auto [dy, dx, _] = offset;

  if (dx == 0 && dy == 0) {
    /* self */
    int rloc = _face_local_rank(rx, ry);
    return _global_rank_from_face_local(face, rloc);
  }

  /* 1-step edge move */
  if ((dx == 0) ^ (dy == 0)) {
    int f1, x1, y1;
    _step_one(face, rx, ry, dx, dy, &f1, &x1, &y1);
    int rloc = _face_local_rank(x1, y1);
    return _global_rank_from_face_local(f1, rloc);
  }

  /* corners: at least crossing one edge, maybe two */
  // find the current block's logical location
  int lx, ly;
  logical_loc2(rx, ry, pxy(), pxy(), &lx, &ly);

  if ((dx + lx <= 1) && (dx + lx >= -1)) {
    // do (dx,0) and then (0,dy)
    // printf("lx = %d, ly = %d, dx = %d, dy = %d\n", lx, ly, dx, dy);
    int f1, x1, y1;
    _step_one(face, rx, ry, dx, 0, &f1, &x1, &y1);

    int f2, x2, y2;
    _step_one(f1, x1, y1, 0, dy, &f2, &x2, &y2);
    int rloc = _face_local_rank(x2, y2);
    return _global_rank_from_face_local(f2, rloc);
  } else if ((dy + ly <= 1) && (dy + ly >= -1)) {
    // do (0, dy) and then (dx, 0)
    int f1, x1, y1;
    _step_one(face, rx, ry, 0, dy, &f1, &x1, &y1);

    int f2, x2, y2;
    _step_one(f1, x1, y1, dx, 0, &f2, &x2, &y2);
    int rloc = _face_local_rank(x2, y2);
    return _global_rank_from_face_local(f2, rloc);
  } else {  // crossing two edges
    int f1, x1, y1;
    _step_one(face, rx, ry, dx, 0, &f1, &x1, &y1);
    int rloc = _face_local_rank(x1, y1);
    return _global_rank_from_face_local(f1, rloc);
  }
}

void CubedSphereLayoutImpl::serialize(MeshBlockImpl const* pmb, Variables& vars,
                                      SyncOptions const& opts) {
  if (options->verbose()) {
    SINFO(CubedSphereLayout) << "serializing data into send buffers\n";
  }

  auto pcoord = pmb->pcoord;

  // Get my logical location
  auto iloc = loc_of(options->rank());

  // Iterate over all face-adjacent neighbor directions
  int dy_min = opts.dy_min();
  int dy_max = opts.dy_max();
  int dx_min = opts.dx_min();
  int dx_max = opts.dx_max();

  // Serialize over all intra-panel neighbors first
  if (!opts.cross_panel_only()) {
    for (int dy = dy_min; dy <= dy_max; ++dy)
      for (int dx = dx_min; dx <= dx_max; ++dx) {
        // skip the center (self)
        if (dy == 0 && dx == 0) continue;
        if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;

        std::tuple<int, int, int> offset(dy, dx, 0);
        int nb = neighbor_rank(iloc, offset);
        if (nb < 0) continue;  // no neighbor

        // skip inter-panel neighbors
        if (std::get<2>(iloc) != std::get<2>(loc_of(nb))) continue;

        // Get the interior part for this direction
        auto sub = pmb->part(offset, PartOptions().exterior(false));

        // Copy data from mesh to send buffer
        int bid = get_buffer_id(offset);

        pmb->send_bufs[bid].clear();
        pmb->recv_bufs[bid].clear();
        pmb->send_bufs[bid].reserve(vars.size());
        pmb->recv_bufs[bid].reserve(vars.size());

        for (auto& [name, var] : vars) {
          // do partial send if name string contains ':'
          auto pos = name.find(":");
          if (pos != std::string::npos) {
            auto suffix = name.substr(pos + 1);
            if ((suffix == "+" && (dy < 0 || dx < 0)) ||
                (suffix == "-" && (dy > 0 || dx > 0)))
              continue;
          }
          pmb->send_bufs[bid].push_back(var.index(sub).clone());
          pmb->recv_bufs[bid].push_back(
              torch::empty_like(pmb->send_bufs[bid].back()));
        }
      }
  }

  // get mesh
  torch::Tensor x2_coord, x3_coord, cosine_cell;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();

  if (opts.dim() == SyncOptions::DIM3) {
    x3_coord = pcoord->x3f;
    x2_coord = pcoord->x2v;
    cosine_cell = pcoord->cosine_face3_kj.expand({nc3 + 1, nc2, nc1});
  } else if (opts.dim() == SyncOptions::DIM2) {
    x3_coord = pcoord->x3v;
    x2_coord = pcoord->x2f;
    cosine_cell = pcoord->cosine_face2_kj.expand({nc3, nc2 + 1, nc1});
  } else {
    x3_coord = pcoord->x3v;
    x2_coord = pcoord->x2v;
    cosine_cell = pcoord->cosine_cell_kj.expand({nc3, nc2, nc1});
  }

  auto mesh = torch::meshgrid({x3_coord, x2_coord, pcoord->x1v}, "ij");

  // Serialize over all inter-panel neighbors
  for (int dy = dy_min; dy <= dy_max; ++dy)
    for (int dx = dx_min; dx <= dx_max; ++dx) {
      // skip the center (self)
      if (dy == 0 && dx == 0) continue;
      if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;

      std::tuple<int, int, int> offset(dy, dx, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      // skip intra-panel neighbors
      if (std::get<2>(iloc) == std::get<2>(loc_of(nb))) continue;

      // Get the interior part for this direction
      auto part_opts = PartOptions().exterior(false);
      if (opts.dim() == SyncOptions::DIM2) {
        part_opts.depth(1).exterior(dx > 0);
      } else if (opts.dim() == SyncOptions::DIM3) {
        part_opts.depth(1).exterior(dy > 0);
      }

      auto sub = pmb->part(offset, part_opts);
      auto sub3 = pmb->part(offset, part_opts.ndim(3));

      // Copy data from mesh to send buffer
      int bid = get_buffer_id(offset);

      pmb->send_bufs[bid].clear();
      pmb->recv_bufs[bid].clear();
      pmb->send_bufs[bid].reserve(vars.size());
      pmb->recv_bufs[bid].reserve(vars.size());

      int my_side = get_side(offset);
      int nb_side = CS_FACE_EDGES[std::get<2>(iloc)][my_side].nside;

      bool rev_flag = CS_FACE_EDGES[std::get<2>(iloc)][my_side].rev;
      bool trans_flag = (my_side - 1.5) * (nb_side - 1.5) < 0;
      bool flip_flag = (my_side % 2) == (nb_side % 2);

      auto alpha = mesh[1].index(sub3);
      auto beta = mesh[0].index(sub3);

      for (auto& [name, var] : vars) {
        // do partial send if name string contains ':'
        auto pos = name.find(":");
        if (pos != std::string::npos) {
          auto suffix = name.substr(pos + 1);
          if ((suffix == "+" && (dy < 0 || dx < 0)) ||
              (suffix == "-" && (dy > 0 || dx > 0)))
            continue;
        }

        auto var_send = var.index(sub).clone();

        switch (opts.type()) {
          case kConserved: {
            auto vel = var_send.narrow(0, IVX, 3);
            coord_vec_raise_(vel, cosine_cell.index(sub3));
            cs_contra_to_sph_(vel, alpha, beta, std::get<2>(iloc));
          } break;
          case kPrimitive: {
            auto vel = var_send.narrow(0, IVX, 3);
            cs_contra_to_sph_(vel, alpha, beta, std::get<2>(iloc));
          } break;
          case kScalar:
            break;
        }

        // check reverse flag
        if (rev_flag) {
          if (dy != 0) {
            var_send = var_send.flip(-2);
          } else if (dx != 0) {
            var_send = var_send.flip(-3);
          }
        }

        // check flip flag
        if (flip_flag) {
          if (dy != 0) {
            var_send = var_send.flip(-3);
          } else if (dx != 0) {
            var_send = var_send.flip(-2);
          }
        }

        // check transpose flag
        if (trans_flag) {
          auto sizes = var_send.sizes();
          var_send = var_send.transpose(-2, -3).reshape(sizes);
        }

        pmb->send_bufs[bid].push_back(var_send);
        pmb->recv_bufs[bid].push_back(
            torch::empty_like(pmb->send_bufs[bid].back()));
      }
    }
}

void CubedSphereLayoutImpl::deserialize(MeshBlockImpl const* pmb,
                                        Variables& vars,
                                        SyncOptions const& opts) const {
  if (options->verbose()) {
    SINFO(CubedSphereLayout) << "deserializing data from receive buffers\n";
  }

  auto pcoord = pmb->pcoord;

  // Get my logical location
  auto iloc = loc_of(options->rank());

  int dy_min = opts.dy_min();
  int dy_max = opts.dy_max();
  int dx_min = opts.dx_min();
  int dx_max = opts.dx_max();

  // Deserialize over all intra-panel neighbors first
  if (!opts.cross_panel_only()) {
    for (int dy = dy_min; dy <= dy_max; ++dy)
      for (int dx = dx_min; dx <= dx_max; ++dx) {
        // skip the center (self)
        if (dy == 0 && dx == 0) continue;
        if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;

        std::tuple<int, int, int> offset(dy, dx, 0);
        int nb = neighbor_rank(iloc, offset);
        if (nb < 0) continue;  // no neighbor

        // skip inter-panel neighbors
        if (std::get<2>(iloc) != std::get<2>(loc_of(nb))) continue;

        // Get the exterior (ghost zone) part for this direction
        auto sub = pmb->part(offset, PartOptions().exterior(true));

        // Copy data from receive buffer to mesh ghost zones
        int bid = get_buffer_id(offset);
        int count = 0;
        for (auto& [name, var] : vars) {
          // do partial recv if name string contains ':'
          auto pos = name.find(":");
          if (pos != std::string::npos) {
            auto suffix = name.substr(pos + 1);
            if ((suffix == "+" && (dy > 0 || dx > 0)) ||
                (suffix == "-" && (dy < 0 || dx < 0)))
              continue;
          }
          var.index_put_(sub, pmb->recv_bufs[bid][count++]);
        }
      }
  }

  // get mesh
  torch::Tensor x2_coord, x3_coord, cosine_cell;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();

  if (opts.dim() == SyncOptions::DIM3) {
    x3_coord = pcoord->x3f;
    x2_coord = pcoord->x2v;
    cosine_cell = pcoord->cosine_face3_kj.expand({nc3 + 1, nc2, nc1});
  } else if (opts.dim() == SyncOptions::DIM2) {
    x3_coord = pcoord->x3v;
    x2_coord = pcoord->x2f;
    cosine_cell = pcoord->cosine_face2_kj.expand({nc3, nc2 + 1, nc1});
  } else {
    x3_coord = pcoord->x3v;
    x2_coord = pcoord->x2v;
    cosine_cell = pcoord->cosine_cell_kj.expand({nc3, nc2, nc1});
  }

  auto mesh = torch::meshgrid({x3_coord, x2_coord, pcoord->x1v}, "ij");

  // Deserialize over all inter-panel neighbors
  for (int dy = dy_min; dy <= dy_max; ++dy)
    for (int dx = dx_min; dx <= dx_max; ++dx) {
      // skip the center (self)
      if (dy == 0 && dx == 0) continue;
      if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;

      std::tuple<int, int, int> offset(dy, dx, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      // skip intra-panel neighbors
      if (std::get<2>(iloc) == std::get<2>(loc_of(nb))) continue;

      // Get the exterior (ghost zone) part for this direction
      auto part_opts = PartOptions().exterior(true);
      if (opts.dim() == SyncOptions::DIM2) {
        part_opts.depth(1).exterior(dx > 0);
      } else if (opts.dim() == SyncOptions::DIM3) {
        part_opts.depth(1).exterior(dy > 0);
      }

      auto sub = pmb->part(offset, part_opts);
      auto sub3 = pmb->part(offset, part_opts.ndim(3));

      auto alpha = mesh[1].index(sub3);
      auto beta = mesh[0].index(sub3);

      // Copy data from receive buffer to mesh ghost zones
      int bid = get_buffer_id(offset);
      int count = 0;
      for (auto& [name, var] : vars) {
        // do partial recv if name string contains ':'
        auto pos = name.find(":");
        if (pos != std::string::npos) {
          auto suffix = name.substr(pos + 1);
          if ((suffix == "+" && (dy > 0 || dx > 0)) ||
              (suffix == "-" && (dy < 0 || dx < 0)))
            continue;
        }

        var.index_put_(sub, pmb->recv_bufs[bid][count++]);
        if (opts.interpolate()) {
          pcoord->interp_ghost(var, offset);
        }

        switch (opts.type()) {
          case kConserved: {
            auto vel = var.index(sub).narrow(0, IVX, 3);
            cs_sph_to_contra_(vel, alpha, beta, std::get<2>(iloc));
            coord_vec_lower_(vel, cosine_cell.index(sub3));
          } break;
          case kPrimitive: {
            auto vel = var.index(sub).narrow(0, IVX, 3);
            cs_sph_to_contra_(vel, alpha, beta, std::get<2>(iloc));
          } break;
          case kScalar:
            break;
        }
      }
    }
}

void CubedSphereLayoutImpl::exchange_remote(MeshBlockImpl const* pmb,
                                            SyncOptions const& opts,
                                            std::vector<CommWorkPtr>& works) {
  TORCH_CHECK(
      owner() != nullptr,
      "[CubedSphereLayout:exchange_remote] layout has no owning MeshBlock");
  TORCH_CHECK(pmb != nullptr,
              "[CubedSphereLayout:exchange_remote] MeshBlock pointer is null");

  if (options->verbose()) {
    SINFO(CubedSphereLayout) << "performing communication\n";
  }

  // Get my rank
  auto rank = options->rank();

  // Get my logical location
  auto iloc = loc_of(rank);

  int dy_min = opts.dy_min();
  int dy_max = opts.dy_max();
  int dx_min = opts.dx_min();
  int dx_max = opts.dx_max();

  struct RemoteExchangeOp {
    int remote_process;
    int buffer_id;
    int send_tag;
    int recv_tag;
    int local_block;
    int remote_local_block;
    std::tuple<int, int, int> offset;
    std::tuple<int, int, int> remote_offset;
  };
  std::vector<RemoteExchangeOp> remote_ops;

  for (int dy = dy_min; dy <= dy_max; ++dy)
    for (int dx = dx_min; dx <= dx_max; ++dx) {
      // skip the center (self)
      if (dy == 0 && dx == 0) continue;
      if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;

      std::tuple<int, int, int> offset(dy, dx, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      int r = get_buffer_id(offset);
      int remote_process = options->owner_process_rank(nb);
      bool is_remote = remote_process != options->process_rank();

      if (is_remote) {
        int remote_local_block = options->local_block_index(nb);
        int local_block = options->local_block_index(rank);
        auto remote_offset = find_peer_offset(*this, nb, rank);
        int send_tag =
            make_comm_tag(remote_local_block, remote_offset, opts.phyid());
        int recv_tag = make_comm_tag(local_block, offset, opts.phyid());
        remote_ops.push_back({remote_process, r, send_tag, recv_tag,
                              local_block, remote_local_block, offset,
                              remote_offset});
      } else if (nb == rank) {  // self-send
        TORCH_CHECK(false, "I should not be here");
      }
    }

  if (remote_ops.empty()) return;
  TORCH_CHECK(has_process_group(),
              "[CubedSphereLayout:exchange_remote] remote communication "
              "requires an initialized process group");

  std::lock_guard<std::mutex> lock(g_cubed_sphere_comm_mutex);
  for (auto const& op : remote_ops) {
    auto send_work = comm->send(pmb->send_bufs[op.buffer_id], op.remote_process,
                                op.send_tag);
    if (send_work) {
      works.push_back(send_work);
    }
    auto recv_work = comm->recv(pmb->recv_bufs[op.buffer_id], op.remote_process,
                                op.recv_tag);
    if (recv_work) {
      works.push_back(recv_work);
    }
  }
}

std::tuple<int, int, int> CubedSphereLayoutImpl::_remap_exchange_offset(
    std::tuple<int, int, int> iloc, int dy, int dx, int dz) const {
  (void)iloc;
  (void)dz;
  return {dy, dx, 0};
}

std::tuple<int, int, int> CubedSphereLayoutImpl::_peer_exchange_offset(
    int peer_rank, int target_rank, SyncOptions const& opts,
    std::tuple<int, int, int> offset) const {
  (void)opts;
  (void)offset;
  return find_peer_offset(*this, peer_rank, target_rank);
}

void CubedSphereLayoutImpl::_interpolate_to_local(
    MeshBlockImpl const* pmb, std::tuple<int, int, int> offset,
    torch::Tensor var) const {
  // my coordinates
  auto pcoord = pmb->pcoord;
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v},
                              /*indexing=*/"ij");

  auto sub = pmb->part(offset, PartOptions().exterior(true));

  auto x2v = mesh[1].unsqueeze(0).index(sub).squeeze(0);
  auto x3v = mesh[0].unsqueeze(0).index(sub).squeeze(0);

  auto var_neighbor = var.clone();

  if (options->verbose() && is_root()) {
    std::cout << "offset = (" << std::get<0>(offset) << ", "
              << std::get<1>(offset) << ", " << std::get<2>(offset) << ")\n";
    std::cout << "var from neighbor = \n"
              << var_neighbor[IDN].squeeze().transpose(0, 1).flip(0) << "\n";
  }

  //\TODO calculate neighbor coordinates and perform interpolation
}

}  // namespace snap
