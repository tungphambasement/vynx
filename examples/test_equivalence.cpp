#include <sys/stat.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "equivalence_utils.hpp"
#include "nn/engines/cuda_engine.hpp"
#include "nn/engines/cudnn_engine.hpp"
#include "nn/example_graphs.hpp"
#include "nn/graph.hpp"
#include "nn/graph_executor.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"

using namespace tunx;

int main(int argc, char** argv) {
  std::string model_name = "resnet50";
  std::string pt_dir = "";
  std::string tunx_dir = "";
  size_t batch_size = 1;
  std::string executor_mode = "optimized";
  std::string engine_mode = "default";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_name = argv[++i];
    } else if (arg == "--pt-dir" && i + 1 < argc) {
      pt_dir = argv[++i];
    } else if (arg == "--tunx-dir" && i + 1 < argc) {
      tunx_dir = argv[++i];
    } else if (arg == "--batch-size" && i + 1 < argc) {
      batch_size = std::stoi(argv[++i]);
    } else if (arg == "--executor-mode" && i + 1 < argc) {
      executor_mode = argv[++i];
    } else if (arg == "--engine" && i + 1 < argc) {
      engine_mode = argv[++i];
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (pt_dir.empty() || tunx_dir.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " --model <name> --pt-dir <dir> --tunx-dir <dir> [--batch-size <N>]"
              << " [--executor-mode optimized|naive|linear|branching|joining]"
              << " [--engine default|cuda|cudnn]" << std::endl;
    return 1;
  }

  // Ensure output directory exists
  mkdir(tunx_dir.c_str(), 0777);

  DeviceManager& manager = DeviceManager::instance();
  auto device_ids = manager.get_all();
  if (device_ids.empty()) {
    std::cerr << "No devices found." << std::endl;
    return 1;
  }

  // Find CUDA device if available
  tunx::DeviceID target_device_id = device_ids[0];
  for (auto id : device_ids) {
    if (manager.get(id).device_type() == DeviceType::CUDA) {
      target_device_id = id;
      break;
    }
  }

  Device& device = manager.get(target_device_id);
  IAllocator& allocator = DeviceAllocator::instance(device);

  std::cout << "Creating model " << model_name << " on " << device.get_name() << std::endl;
  ExampleGraphs::register_defaults();

  GraphOpts opts;
  if (engine_mode == "cuda") {
    opts.engine = make_engine<CUDAEngine>();
  } else if (engine_mode == "cudnn") {
    opts.engine = make_engine<CuDNNEngine>();
  } else if (engine_mode != "default") {
    std::cerr << "Unknown engine: " << engine_mode << std::endl;
    return 1;
  }

  std::string actual_model_name = model_name;
  if (model_name == "resnet50") {
    actual_model_name = "imagenet100_resnet50";
  } else if (model_name == "gpt2") {
    actual_model_name = "gpt2_small";
  }

  Graph graph = ExampleGraphs::create(actual_model_name, allocator, opts);

  std::cout << "Loading initial parameters from " << pt_dir << std::endl;
  for (auto& edge : graph.edges()) {
    auto layer = edge->layer();
    auto params = layer->params();
    if (params.empty()) continue;

    std::string name = layer->name();

    if (params.size() >= 1) {
      std::string path = pt_dir + "/" + name + ".weight.bin";
      if (file_exists(path)) {
        load_tensor_bin(params[0].data(), path);
      } else {
        std::cerr << "Warning: Missing " << path << std::endl;
      }
    }
    if (params.size() >= 2) {
      std::string path = pt_dir + "/" + name + ".bias.bin";
      if (file_exists(path)) {
        load_tensor_bin(params[1].data(), path);
      } else {
        std::cerr << "Warning: Missing " << path << std::endl;
      }
    }
  }

  std::cout << "Loading inputs and labels..." << std::endl;
  bool is_lm = (model_name.find("gpt2") != std::string::npos);
  Vec<size_t> input_shape = {batch_size, 224, 224, 3};
  Vec<size_t> label_shape = {batch_size};
  if (is_lm) {
    input_shape = {batch_size, 1024};
    label_shape = {batch_size, 1024};
  }

  Tensor inputs;
  if (is_lm) {
    inputs = Tensor(input_shape, DType_t::INT32, allocator);
  } else {
    inputs = Tensor(input_shape, DType_t::FP32, allocator);
  }
  load_tensor_bin(inputs, pt_dir + "/inputs.bin");

  Tensor labels = Tensor(label_shape, DType_t::INT32, allocator);
  load_tensor_bin(labels, pt_dir + "/labels.bin");

  // Build a uid -> layer_name map from all consumer nodes so we can name the dumped files.
  std::map<std::string, std::string> uid_to_layer_name;  // node_uid -> layer_name
  for (auto& edge : graph.edges()) {
    auto layer = edge->layer();
    if (!layer) continue;
    std::string name = layer->name();
    for (auto& consumer : edge->consumers()) {
      uid_to_layer_name[consumer->uid()] = name;
    }
  }

  std::cout << "Running forward pass..." << std::endl;
  GraphExecutor executor(graph);

  SolverOptions solver_options;
  if (executor_mode == "naive") {
    solver_options = SolverOptions{true, false, false, false};
  } else if (executor_mode == "linear") {
    solver_options = SolverOptions{false, true, false, false};
  } else if (executor_mode == "branching") {
    solver_options = SolverOptions{false, true, true, false};
  } else if (executor_mode == "joining") {
    solver_options = SolverOptions{false, true, false, true};
  } else if (executor_mode != "optimized") {
    std::cerr << "Unknown executor mode: " << executor_mode << std::endl;
    return 1;
  }
  std::map<std::string, Tensor> captured_acts;
  executor.set_forward_hook(
      [&](const Edge& edge, const std::map<std::string, Tensor>& consumers_data) {
        for (const auto& consumer : edge->consumers()) {
          const std::string& uid = consumer->uid();
          auto it = consumers_data.find(uid);
          if (it != consumers_data.end()) {
            Tensor t_copy(it->second.shape(), it->second.dtype(), allocator);
            tunx::copy(it->second, t_copy, executor.graph().handle().get_stream());
            captured_acts[uid] = std::move(t_copy);
          }
        }
      });

  TensorBundle input_tensors{{"input", inputs}};
  executor.build_plans(input_tensors, solver_options);

  TensorBundle outputs = executor.forward(input_tensors);
  Tensor predictions = outputs.get("output");

  std::cout << "Dumping outputs..." << std::endl;
  save_tensor_bin(predictions, tunx_dir + "/outputs.bin");

  // Capture intermediate activations right after each layer via the hook to prevent packed
  // allocator from overwriting them
  std::cout << "Dumping intermediate activations..." << std::endl;
  for (auto& [uid, tensor] : captured_acts) {
    auto it = uid_to_layer_name.find(uid);
    if (it == uid_to_layer_name.end()) continue;
    save_tensor_bin(tensor, tunx_dir + "/" + it->second + ".act.bin");
  }

  // Dump BatchNorm statistics captured by the training forward pass.
  for (auto& edge : graph.edges()) {
    auto layer = edge->layer();
    if (!layer) continue;
    auto residual_it = executor.residuals().find(edge);
    if (residual_it == executor.residuals().end()) continue;

    const std::string& name = layer->name();
    const auto residual_tensors = residual_it->second.tensors();
    auto batch_mean = residual_tensors.find("batch_mean.");
    auto batch_invar = residual_tensors.find("batch_invar.");
    if (batch_mean != residual_tensors.end()) {
      save_tensor_bin(batch_mean->second, tunx_dir + "/" + name + ".batch_mean.bin");
    }
    if (batch_invar != residual_tensors.end()) {
      save_tensor_bin(batch_invar->second, tunx_dir + "/" + name + ".batch_invar.bin");
    }

    auto params = layer->params();
    if (params.size() >= 6 && batch_mean != residual_tensors.end()) {
      save_tensor_bin(params[4].data(), tunx_dir + "/" + name + ".running_mean.bin");
      save_tensor_bin(params[5].data(), tunx_dir + "/" + name + ".running_var.bin");
    }
  }

  std::cout << "Running backward pass..." << std::endl;

  std::shared_ptr<Loss> criterion = std::make_shared<CrossEntropyLoss>();
  float loss;
  criterion->compute_loss(predictions, labels, loss);
  Tensor loss_gradient = Tensor(predictions.shape(), predictions.dtype(), allocator);
  criterion->compute_gradient(predictions, labels, loss_gradient);
  save_tensor_bin(loss_gradient, tunx_dir + "/grad_output.bin");

  TensorBundle output_grads{{"output", loss_gradient}};

  std::shared_ptr<Optimizer> optimizer = std::make_shared<Adam>(1e-3, 0.9, 0.999, 1e-8, 3e-4);
  optimizer->attach(graph);
  optimizer->zero_grads();

  std::map<std::string, Tensor> captured_grads;
  executor.set_backward_grad_hook([&](const Edge& edge,
                                      const std::map<std::string, Tensor>& consumers_grads,
                                      const std::map<std::string, Tensor>& /*producers_grads*/) {
    for (const auto& consumer : edge->consumers()) {
      const std::string& uid = consumer->uid();
      auto it = consumers_grads.find(uid);
      if (it != consumers_grads.end()) {
        Tensor t_copy(it->second.shape(), it->second.dtype(), allocator);
        tunx::copy(it->second, t_copy, executor.graph().handle().get_stream());
        captured_grads[uid] = std::move(t_copy);
        // use copy since packed allocator do not respect ownership that much. Will need fix later
      }
    }
  });

  executor.backward(output_grads);
  cudaDeviceSynchronize();

  // Dump intermediate gradients — collected during backward via the hook.
  std::cout << "Dumping intermediate gradients..." << std::endl;
  for (auto& [uid, tensor] : captured_grads) {
    auto it = uid_to_layer_name.find(uid);
    if (it == uid_to_layer_name.end()) continue;
    save_tensor_bin(tensor, tunx_dir + "/" + it->second + ".act.grad.bin");
  }

  std::cout << "Dumping gradients..." << std::endl;

  for (auto& edge : graph.edges()) {
    auto layer = edge->layer();
    auto params = layer->params();
    if (params.empty()) continue;

    std::string name = layer->name();

    if (params.size() >= 1) {
      save_tensor_bin(params[0].grad(), tunx_dir + "/" + name + ".weight.grad.bin");
    }
    if (params.size() >= 2) {
      save_tensor_bin(params[1].grad(), tunx_dir + "/" + name + ".bias.grad.bin");
    }
  }

  std::cout << "Running optimizer step..." << std::endl;
  optimizer->update();

  std::cout << "Dumping updated parameters..." << std::endl;
  for (auto& edge : graph.edges()) {
    auto layer = edge->layer();
    auto params = layer->params();
    if (params.empty()) continue;

    std::string name = layer->name();

    if (params.size() >= 1) {
      save_tensor_bin(params[0].data(), tunx_dir + "/" + name + ".weight.updated.bin");
    }
    if (params.size() >= 2) {
      save_tensor_bin(params[1].data(), tunx_dir + "/" + name + ".bias.updated.bin");
    }
  }

  std::cout << "Dump complete for " << model_name << " in " << tunx_dir << std::endl;

  return 0;
}
