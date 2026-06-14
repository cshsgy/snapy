#include <configure.h>

#ifdef USE_UCX

#include <ucp/api/ucp.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include "layout.hpp"
#include "process_group.hpp"

namespace snap {
namespace {

void check_ucs(ucs_status_t status, char const* operation) {
  TORCH_CHECK(status == UCS_OK, operation,
              " failed: ", ucs_status_string(status));
}

struct UcxBatch {
  std::mutex mutex;
  std::condition_variable cv;
  int remaining = 0;
  ucs_status_t status = UCS_OK;

  void complete(ucs_status_t request_status) {
    std::lock_guard<std::mutex> lock(mutex);
    if (status == UCS_OK && request_status != UCS_OK) status = request_status;
    remaining -= 1;
    if (remaining == 0) cv.notify_all();
  }
};

class UcxWork final : public CommWork {
 public:
  UcxWork(std::shared_ptr<UcxBatch> batch, std::vector<torch::Tensor> tensors)
      : batch_(std::move(batch)), tensors_(std::move(tensors)) {}

  void wait() override {
    std::unique_lock<std::mutex> lock(batch_->mutex);
    batch_->cv.wait(lock, [&]() { return batch_->remaining == 0; });
    check_ucs(batch_->status, "UCX request");
    lock.unlock();
    sync_tensor_streams(tensors_);
  }

 private:
  std::shared_ptr<UcxBatch> batch_;
  std::vector<torch::Tensor> tensors_;
};

void send_callback(void* request, ucs_status_t status, void* user_data) {
  static_cast<UcxBatch*>(user_data)->complete(status);
  ucp_request_free(request);
}

void recv_callback(void* request, ucs_status_t status,
                   ucp_tag_recv_info_t const*, void* user_data) {
  static_cast<UcxBatch*>(user_data)->complete(status);
  ucp_request_free(request);
}

uint64_t tensor_tag(int source_rank, int tag, int tensor_index) {
  TORCH_CHECK(source_rank >= 0 && source_rank <= 0xffff,
              "UCX source rank out of range: ", source_rank);
  TORCH_CHECK(tag >= 0 && tag <= 0x7fffff,
              "UCX message tag out of range: ", tag);
  TORCH_CHECK(tensor_index >= 0 && tensor_index <= 0xff,
              "UCX tensor-list index out of range: ", tensor_index);
  return (static_cast<uint64_t>(source_rank) << 32) |
         (static_cast<uint64_t>(tag) << 8) |
         static_cast<uint64_t>(tensor_index);
}

void apply_reduce(torch::Tensor& target, torch::Tensor const& source,
                  c10d::ReduceOp op) {
  if (op == c10d::ReduceOp::SUM) {
    target.add_(source);
  } else if (op == c10d::ReduceOp::MIN) {
    target.copy_(torch::minimum(target, source));
  } else if (op == c10d::ReduceOp::MAX) {
    target.copy_(torch::maximum(target, source));
  } else {
    TORCH_CHECK(false, "UCX reduction does not support this reduction op");
  }
}

class NativeUcxTransport final : public ProcessGroupContext::UcxTransport {
 public:
  NativeUcxTransport(LayoutOptions options,
                     at::intrusive_ptr<c10d::Store> store)
      : options_(std::move(options)), store_(std::move(store)) {
    ucp_params_t context_params{};
    context_params.field_mask =
        UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_MT_WORKERS_SHARED;
    context_params.features = UCP_FEATURE_TAG;
    context_params.mt_workers_shared = 1;

    ucp_config_t* config = nullptr;
    check_ucs(ucp_config_read(nullptr, nullptr, &config), "ucp_config_read");
    auto status = ucp_init(&context_params, config, &context_);
    ucp_config_release(config);
    check_ucs(status, "ucp_init");

    ucp_worker_params_t worker_params{};
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    check_ucs(ucp_worker_create(context_, &worker_params, &worker_),
              "ucp_worker_create");

    ucp_address_t* address = nullptr;
    size_t address_length = 0;
    check_ucs(ucp_worker_get_address(worker_, &address, &address_length),
              "ucp_worker_get_address");
    std::vector<uint8_t> local_address(
        reinterpret_cast<uint8_t*>(address),
        reinterpret_cast<uint8_t*>(address) + address_length);
    store_->set(address_key(options_->process_rank()), local_address);
    ucp_worker_release_address(worker_, address);

    endpoints_.resize(options_->process_world_size(), nullptr);
    for (int peer = 0; peer < options_->process_world_size(); ++peer) {
      if (peer == options_->process_rank()) continue;
      auto remote_address = store_->get(address_key(peer));
      ucp_ep_params_t endpoint_params{};
      endpoint_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
      endpoint_params.address =
          reinterpret_cast<ucp_address_t const*>(remote_address.data());
      check_ucs(ucp_ep_create(worker_, &endpoint_params, &endpoints_[peer]),
                "ucp_ep_create");
    }

    progress_thread_ = std::thread([this]() {
      while (!stopping_.load(std::memory_order_acquire)) {
        if (ucp_worker_progress(worker_) == 0) std::this_thread::yield();
      }
      while (ucp_worker_progress(worker_) != 0) {
      }
    });
  }

  ~NativeUcxTransport() override { shutdown(); }

  CommWorkPtr send(std::vector<torch::Tensor>& tensors, int peer,
                   int tag) override {
    TORCH_CHECK(peer >= 0 && peer < endpoints_.size() && endpoints_[peer],
                "invalid UCX send peer ", peer);
    return post(tensors, peer, tag, true);
  }

  CommWorkPtr recv(std::vector<torch::Tensor>& tensors, int peer,
                   int tag) override {
    TORCH_CHECK(peer >= 0 && peer < endpoints_.size() && endpoints_[peer],
                "invalid UCX receive peer ", peer);
    return post(tensors, peer, tag, false);
  }

  void allreduce(std::vector<torch::Tensor>& tensors,
                 c10d::ReduceOp op) override {
    std::lock_guard<std::mutex> lock(collective_mutex_);
    int tag = next_collective_tag();
    reduce_impl(tensors, op, 0, tag);
    for (int peer = 1; peer < options_->process_world_size(); ++peer) {
      if (options_->process_rank() == 0) {
        send(tensors, peer, tag + 1)->wait();
      } else if (options_->process_rank() == peer) {
        recv(tensors, 0, tag + 1)->wait();
      }
    }
  }

  void reduce(std::vector<torch::Tensor>& tensors, c10d::ReduceOp op,
              int root) override {
    std::lock_guard<std::mutex> lock(collective_mutex_);
    reduce_impl(tensors, op, root, next_collective_tag());
  }

  void barrier() override {
    std::vector<torch::Tensor> token = {
        torch::zeros({1}, torch::TensorOptions().dtype(torch::kUInt8))};
    allreduce(token, c10d::ReduceOp::SUM);
  }

  void shutdown() override {
    bool expected = false;
    if (!shutdown_started_.compare_exchange_strong(expected, true)) return;

    for (auto& endpoint : endpoints_) {
      if (endpoint == nullptr) continue;
      ucp_request_param_t params{};
      params.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
      params.flags = UCP_EP_CLOSE_FLAG_FORCE;
      void* request = ucp_ep_close_nbx(endpoint, &params);
      if (request != nullptr && !UCS_PTR_IS_ERR(request)) {
        while (ucp_request_check_status(request) == UCS_INPROGRESS) {
          std::this_thread::yield();
        }
        ucp_request_free(request);
      }
      endpoint = nullptr;
    }
    stopping_.store(true, std::memory_order_release);
    if (progress_thread_.joinable()) progress_thread_.join();
    if (worker_) ucp_worker_destroy(worker_);
    if (context_) ucp_cleanup(context_);
    worker_ = nullptr;
    context_ = nullptr;
  }

 private:
  static std::string address_key(int rank) {
    return "snapy/ucx/address/" + std::to_string(rank);
  }

  CommWorkPtr post(std::vector<torch::Tensor>& tensors, int peer, int tag,
                   bool is_send) {
    auto batch = std::make_shared<UcxBatch>();
    batch->remaining = static_cast<int>(tensors.size());
    if (tensors.empty()) return std::make_shared<UcxWork>(batch, tensors);

    for (int i = 0; i < tensors.size(); ++i) {
      TORCH_CHECK(tensors[i].is_contiguous(),
                  "UCX communication requires contiguous tensors");
      ucp_request_param_t params{};
      params.op_attr_mask =
          UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
      params.user_data = batch.get();

      void* request = nullptr;
      if (is_send) {
        params.cb.send = send_callback;
        request = ucp_tag_send_nbx(
            endpoints_[peer], tensors[i].data_ptr(), tensors[i].nbytes(),
            tensor_tag(options_->process_rank(), tag, i), &params);
      } else {
        params.cb.recv = recv_callback;
        request = ucp_tag_recv_nbx(
            worker_, tensors[i].data_ptr(), tensors[i].nbytes(),
            tensor_tag(peer, tag, i), UINT64_MAX, &params);
      }

      if (request == nullptr) {
        batch->complete(UCS_OK);
      } else if (UCS_PTR_IS_ERR(request)) {
        batch->complete(UCS_PTR_STATUS(request));
      }
    }
    return std::make_shared<UcxWork>(batch, tensors);
  }

  void reduce_impl(std::vector<torch::Tensor>& tensors, c10d::ReduceOp op,
                   int root, int tag) {
    if (options_->process_rank() == root) {
      for (int peer = 0; peer < options_->process_world_size(); ++peer) {
        if (peer == root) continue;
        std::vector<torch::Tensor> incoming;
        incoming.reserve(tensors.size());
        for (auto const& tensor : tensors) {
          incoming.push_back(torch::empty_like(tensor));
        }
        recv(incoming, peer, tag)->wait();
        for (int i = 0; i < tensors.size(); ++i) {
          apply_reduce(tensors[i], incoming[i], op);
        }
      }
    } else {
      send(tensors, root, tag)->wait();
    }
  }

  int next_collective_tag() {
    int sequence = collective_sequence_++;
    TORCH_CHECK(sequence < 0x100000, "UCX collective tag space exhausted");
    return 0x600000 + sequence * 2;
  }

  LayoutOptions options_;
  at::intrusive_ptr<c10d::Store> store_;
  ucp_context_h context_ = nullptr;
  ucp_worker_h worker_ = nullptr;
  std::vector<ucp_ep_h> endpoints_;
  std::atomic<bool> shutdown_started_{false};
  std::atomic<bool> stopping_{false};
  std::thread progress_thread_;
  std::mutex collective_mutex_;
  int collective_sequence_ = 0;
};

}  // namespace

void ProcessGroupContext::_init_ucx() {
  ucx_ = std::make_shared<NativeUcxTransport>(options_, store);
  owns_process_group_ = true;
}

}  // namespace snap

#endif
