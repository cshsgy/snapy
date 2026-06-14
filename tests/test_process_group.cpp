// C/C++
#include <unistd.h>

// external
#include <gtest/gtest.h>

// snap
#include <snap/layout/distributed.hpp>
#include <snap/layout/layout.hpp>
#include <snap/layout/process_group.hpp>

using namespace snap;

namespace {

int unique_test_port() { return 29501 + (getpid() % 1000); }

LayoutOptions make_test_options() {
  auto opts = LayoutOptionsImpl::create();
  opts->backend("gloo");
  opts->master_addr("127.0.0.1");
  opts->master_port(unique_test_port());
  opts->process_rank(0);
  opts->rank(0);
  opts->root_rank(0);
  opts->local_rank(0);
  opts->process_world_size(1);
  opts->world_size(1);
  opts->blocks_per_process(1);
  return opts;
}

}  // namespace

TEST(LayoutOptions, DefaultsToAvailableCommunicationBackend) {
  auto opts = LayoutOptionsImpl::create();
#ifdef USE_UCX
  EXPECT_EQ(opts->backend(), "ucx");
#else
  EXPECT_EQ(opts->backend(), "gloo");
#endif
}

TEST(ProcessGroupContext, SkipsSingleProcessCommunication) {
  auto opts = make_test_options();
  auto ctx = ProcessGroupContext::create(opts);

  EXPECT_FALSE(ctx->owns_process_group());
  EXPECT_FALSE(ctx->pg.defined());

  set_process_group(nullptr);
}
