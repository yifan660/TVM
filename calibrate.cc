/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *
 * \file calibrate.cc
 *
 * \brief Create profile graph and calibrate on dataset
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>

#include <numeric>

#include "./quantize.h"

namespace tvm {
namespace relay {
namespace quantize {

// KL divergence minimization code is adapted from MXNet.
// The original one is in incubator-mxnet/src/operator/quantization/calibrate.cc
static std::vector<float> SmoothDistribution(const std::vector<float>& p,
                                             const float eps = 0.0001) {
  std::vector<size_t> is_zeros(p.size());
  std::vector<size_t> is_nonzeros(p.size());
  {
    auto it = p.begin();
    std::generate(is_zeros.begin(), is_zeros.end(),
                  [&it]() { return static_cast<size_t>(*(it++) == 0.f); });
  }
  {
    auto it = p.begin();
    std::generate(is_nonzeros.begin(), is_nonzeros.end(),
                  [&it]() { return static_cast<size_t>(*(it++) != 0.f); });
  }
  size_t n_zeros = std::accumulate(is_zeros.begin(), is_zeros.end(), 0);
  size_t n_nonzeros = p.size() - n_zeros;
  if (!n_nonzeros) {
    // The discrete probability distribution is malformed. All entries are 0.
    return std::vector<float>();
  }
  float eps1 = eps * static_cast<float>(n_zeros) / static_cast<float>(n_nonzeros);
  if (eps1 >= 1.0) return std::vector<float>();
  auto ret = p;
  for (size_t i = 0; i < p.size(); i++) {
    ret[i] += eps * is_zeros[i] - eps1 * is_nonzeros[i];
  }
  return ret;
}

static float ComputeEntropy(float* p, float* q, size_t size) {
  float p_sum = std::accumulate(p, p + size, 0.f);
  float q_sum = std::accumulate(q, q + size, 0.f);
  float ret = 0;
  for (size_t i = 0; i < size; i++) {
    ICHECK(p[i] > 0 && q[i] > 0);
    p[i] /= p_sum;
    q[i] /= q_sum;
    if (p[i] && q[i]) ret += p[i] * std::log(p[i] / q[i]);
  }
  return ret;
}

float MinimizeKL(const std::vector<int>& hist, const std::vector<float>& hist_edges, int num_bins,
                 int num_quantized_bins) {
  const int zero_bin_idx = num_bins / 2;
  const int num_half_quantized_bins = num_quantized_bins / 2;
  std::vector<float> thresholds(num_bins / 2 + 1 - num_quantized_bins / 2, 0.f);
  std::vector<float> divergence(thresholds.size(), 0.f);
  std::vector<float> quantized_bins(num_quantized_bins, 0);
  for (int i = num_quantized_bins / 2; i < zero_bin_idx + 1; ++i) {
    const int p_bin_idx_start = zero_bin_idx - i;
    const int p_bin_idx_stop = zero_bin_idx + i + 1;
    thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop];

    std::vector<int> sliced_nd_hist(p_bin_idx_stop - p_bin_idx_start);
    std::vector<float> p(sliced_nd_hist.size());
    p[0] = 0;
    p.back() = 0;
    for (int j = 0; j < num_bins; j++) {
      if (j <= p_bin_idx_start) {
        p[0] += hist[j];
      } else if (j >= p_bin_idx_stop) {
        p.back() += hist[j];
      } else {
        sliced_nd_hist[j - p_bin_idx_start] = hist[j];
        p[j - p_bin_idx_start] = hist[j];
      }
    }
    // calculate how many bins should be merged to generate quantized distribution q
    const auto num_merged_bins = sliced_nd_hist.size() / num_quantized_bins;
    for (int j = 0; j < num_quantized_bins; j++) {
      const int start = j * num_merged_bins;
      const int stop = (j + 1) * num_merged_bins;
      quantized_bins[j] =
          std::accumulate(sliced_nd_hist.begin() + start, sliced_nd_hist.begin() + stop, 0);
    }
    quantized_bins.back() += std::accumulate(
        sliced_nd_hist.begin() + static_cast<int>(num_quantized_bins * num_merged_bins),
        sliced_nd_hist.end(), 0);
    // expand quantized_bins into p.size bins
    std::vector<float> q(sliced_nd_hist.size(), 0);
    for (int j = 0; j < num_quantized_bins; j++) {
      const int start = j * num_merged_bins;
      const int stop = (j == num_quantized_bins - 1) ? q.size() : ((j + 1) * num_merged_bins);
      int norm = std::count_if(sliced_nd_hist.begin() + start, sliced_nd_hist.begin() + stop,
                               [](size_t i) { return i != 0; });
      if (norm) {
        for (int k = start; k < stop; k++) {
          if (p[k]) q[k] = quantized_bins[j] / norm;
        }
      }
    }
    p = SmoothDistribution(p);
    q = SmoothDistribution(q);

    if (!q.size()) {
      divergence[i - num_half_quantized_bins] = std::numeric_limits<float>::infinity();
    } else {
      divergence[i - num_half_quantized_bins] = ComputeEntropy(p.data(), q.data(), p.size());
    }
  }
  auto min_divergence_idx =
      std::distance(divergence.begin(), std::min_element(divergence.begin(), divergence.end()));
  return thresholds[min_divergence_idx];
}

class StatsCollector : private ExprMutator {
 public:
  StatsCollector() : simulated_quantize_op_(Op::Get("relay.op.annotation.simulated_quantize")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return std::move(ret_func);
  }

 private:
  Array<Expr> profile_data_;
  const Op& simulated_quantize_op_;

  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == simulated_quantize_op_) {
      auto attrs = new_call->attrs.as<SimulatedQuantizeAttrs>();
      // rewrite the annotation
      auto new_attrs = make_object<SimulatedQuantizeAttrs>();
      const Expr& quantize_input = new_call->args[0];                  // expression being quantized
      auto placeholder = MakeConstantScalar(DataType::Float(32), 0.);  // unused argument
      Array<Expr> new_args{quantize_input, placeholder, placeholder, placeholder};
      new_attrs->kind = QAnnotateKind::kQIdentity;
      new_attrs->sign = attrs->sign;
      new_attrs->rounding = attrs->rounding;
      Expr identity_quantize = Call(new_call->op, new_args, Attrs{new_attrs}, {});

      // add non-const expressions to profile data
      if (attrs->kind != QAnnotateKind::kQWeight) {
        ICHECK(!quantize_input.as<ConstantNode>());
        profile_data_.push_back(identity_quantize);
      }
      return identity_quantize;
    } else {
      return new_e;
    }
  }
};

/*
 * \brief Given an annotated graph, create a profile graph to collect profile data from the
 * calibration dataset.
 *
 * This pass collects simulated_quantize op into a tuple. Simulated_quantize ops are rewritten to
 * identity mode. The tuple is the output of the profile graph. Both input and output of this pass
 * are relay::Function.
 *
 * \param expr The simulation graph after annotation.
 * \return The profile graph.
 */
Expr CreateStatsCollector(const Expr& expr) { return StatsCollector().Collect(expr); }

TVM_REGISTER_GLOBAL("relay._quantize.CreateStatsCollector").set_body_typed(CreateStatsCollector);

TVM_REGISTER_GLOBAL("relay._quantize.FindScaleByKLMinimization")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int* hist_ptr = static_cast<int*>(static_cast<void*>(args[0]));
      float* hist_edges_ptr = static_cast<float*>(static_cast<void*>(args[1]));
      int num_bins = args[2];
      int num_quantized_bins = args[3];
      std::vector<int> hist(hist_ptr, hist_ptr + num_bins);
      std::vector<float> hist_edges(hist_edges_ptr, hist_edges_ptr + num_bins + 1);
      ret[0] = MinimizeKL(hist, hist_edges, num_bins, num_quantized_bins);
    });

}  // namespace quantize
}  // namespace relay
}  // namespace tvm












void alpha2DeltaOffset()  {
  if(force_positive || half_range)  {
    delta = maximum();
  }
}



def mid_tread_quantization(self, tensor, id, target, clip=False, sym=True):
        std = tensor.std(-1)
        omega = self.get_omega(std, target_bins=(2**target)).round()

        if clip:
            alpha_mult = tensor.new_tensor(self.get_alpha_mult(omega, sym=sym))
            mu = tensor.mean(dim=-1)
            b = torch.mean(torch.abs(tensor - mu.unsqueeze(-1)), dim=-1)

            rng = (2 * alpha_mult * b) if sym else (torch.max(mu, mu.new_tensor([0.])) + alpha_mult * b)
        else:
            rng = (tensor.max(-1)[0] - tensor.min(-1)[0]) if sym else tensor.max(-1)[0]

        Delta = torch.where(omega > 0, rng / omega,
                            tensor.new_tensor([np.finfo(np.float32).max]))

        # quantize
        out = tensor / Delta.unsqueeze(-1)
        out.round_()

        # clamp
        if clip:
            # Centralize quantization range around mean and make it non-negative for asymetric case
            mu_q = mu / Delta if sym else torch.max(mu, mu.new_tensor([0.])) / Delta
            c_max = mu_q + (omega / 2 if sym else omega)
            c_min = ((mu_q - omega / 2) if sym else tensor.new_tensor([0]))

            # In practice all the parameters Delta, omega, c_max, c_min can be pre-calculated based on statistics
            out = torch.min(out, c_max.unsqueeze(-1))
            out = torch.max(out, c_min.unsqueeze(-1))

        if self.measure_entropy:
            entropy = shannon_entropy(out, handle_negative=True)
            # workaround for out of memory issue
            torch.cuda.empty_cache()
        else:
            entropy = None

        # dequantize
        out.mul_(Delta.unsqueeze(-1))
        return out, entropy


def get_alpha_laplace(self, tensor, stat_id=None, kind='mean', per_channel=False):
  if stat_id is not None:
    b = self.sm().get_tensor_stat(stat_id, 'b', kind=kind)
  else:
    if per_channel:
      b = self.__act_stats_perchannel__(tensor, ['b'], avg_over_batch=False)['b']
    else:
      b = self.__act_stats__(tensor, ['b'], avg_over_batch=False)['b']

    if self.bit_alloc_act and per_channel and self.num_bits <= 4:
      prior = 'std' if self.bit_alloc_prior == 'gaus' else 'b'
      if stat_id is not None:
                std = self.sm().get_tensor_stat(stat_id, prior, kind='mean')
                std = to_cuda(std, tensor.device)
            else:
                if per_channel:
                    std = self.__act_stats_perchannel__(tensor, [prior], avg_over_batch=False)[prior]
                else:
                    std = self.__act_stats__(tensor, [prior], avg_over_batch=False)[prior]

            bit_alloc = self.get_bits_alloc_fixed_target(std, self.bit_alloc_target_act, self.bit_alloc_round)
            aciq_factor = np.array([(self.alpha_laplace_positive[nbit.item()] if (self.force_positive or self.half_range) else self.alpha_laplace[nbit.item()]) for nbit in bit_alloc])
            aciq_factor = to_cuda(aciq_factor, tensor.device)
        else:
            aciq_factor = (self.alpha_laplace_positive[self.num_bits] if (self.force_positive or self.half_range) else self.alpha_laplace[self.num_bits])

        return to_cuda(b, tensor.device) * aciq_factor

    def get_alpha_gaus(self, tensor, tag, stat_id=None, per_channel=False):
        if stat_id is not None:
            std = self.sm().get_tensor_stat(stat_id, 'std', 'mean')
        else:
            if per_channel:
                std = self.__act_stats_perchannel__(tensor, ['std'], avg_over_batch=False)['std']
            else:
                std = self.__act_stats__(tensor, ['std'], avg_over_batch=False)['std']

        return std * (self.alpha_gaus_positive[self.num_bits] if (self.force_positive or self.half_range) else self.alpha_gaus[self.num_bits])

def alpha2DeltaOffset(self, alpha, max_value, min_value, mean, clip2max=False):
  alpha = to_numpy(alpha)
  max_value = to_numpy(max_value)
  min_value = to_numpy(min_value)
  mean = to_numpy(mean)
  if self.force_positive or self.half_range:
    delta = np.maximum(np.array(mean), 0) + alpha
    if clip2max:
      delta = np.minimum(delta, max_value)
    offset = 0
  else:
    delta = 2 * alpha
            if clip2max:
                delta = np.minimum(delta, max_value - min_value)
            offset = np.maximum(min_value, mean - alpha)

        return delta, offset

    def get_alpha(self, tensor, tag="", stat_id=None, clip_type='laplace', per_channel=False):
        if clip_type == 'laplace':
            alpha = self.get_alpha_laplace(tensor, stat_id, per_channel=per_channel)  # laplace clipping
        elif clip_type == 'gaus':
            alpha = self.get_alpha_gaus(tensor, tag, stat_id, per_channel=per_channel)  # gaussian clipping
        elif 'std' in clip_type:
            p = float(clip_type.replace('std', ''))
            alpha = self.get_alpha_pstd(tensor, p, tag, stat_id, per_channel=per_channel)  # 2std clipping
        elif clip_type == 'mix':
            mse_laplace = self.sm().get_tensor_stat(stat_id, 'mse_laplace', 'mean')
            mse_gaus = self.sm().get_tensor_stat(stat_id, 'mse_gaus', 'mean')
            mse_lowp = self.sm().get_tensor_stat(stat_id, 'mse_lowp', 'mean')

            alpha_laplace = self.get_alpha_laplace(tensor, stat_id, per_channel=per_channel)  # laplace clipping
            alpha_gaus = self.get_alpha_gaus(tensor, tag, stat_id, per_channel=per_channel)  # gaussian clipping
            # simulate alpha range for gemm_lowp
            min_ = self.sm().get_tensor_stat(stat_id, 'min', 'mean')
            max_ = self.sm().get_tensor_stat(stat_id, 'max', 'mean')
            alpha_lowp = (max_ - min_)/2

            alpha = np.where(mse_gaus < mse_laplace, alpha_gaus, alpha_laplace)
            alpha = np.where(mse_lowp < mse_gaus, alpha_lowp, alpha)

        return alpha

def gemmlowpClippingQuantize(self, tensor, id, tag="", stat_id=None, clip_type='laplace'):
  if stat_id is not None:
    min_value = self.sm().get_tensor_stat(stat_id, 'min', 'mean')
    max_value = self.sm().get_tensor_stat(stat_id, 'max', 'mean')
    mean = self.sm().get_tensor_stat(stat_id, 'mean', 'mean')
  else:
    if self.pcq_a and len(tensor.shape) > 3 and (tensor.shape[2] > 1 or tensor.shape[3] > 1):
      stats = self.__act_stats_perchannel__(tensor, ['min', 'max'], avg_over_batch=False)
      mean = self.__act_stats_perchannel__(tensor, ['mean'], avg_over_batch=True)['mean']
    else:
      stats = self.__act_stats__(tensor, ['min', 'max', 'mean'], avg_over_batch=False)
      mean = stats['mean']
            min_value = stats['min']
            max_value = stats['max']
            # mean = stats['mean']

        if self.pcq_a and len(tensor.shape) > 3 and (tensor.shape[2] > 1 or tensor.shape[3] > 1) \
                and len(min_value) > 1 and len(max_value) > 1:
            # min_value = self.sm().get_tensor_stat(stat_id, 'min', 'min')
            # max_value = self.sm().get_tensor_stat(stat_id, 'max', 'max')
            alpha = self.get_alpha(tensor, tag, stat_id, clip_type, per_channel=True)
            range, min_value = self.alpha2DeltaOffset(alpha, max_value, min_value, mean)
            min_value = to_cuda(min_value, tensor.device)
            range = to_cuda(range, tensor.device)
            max_ = min_value + range
            res = self.gemmlowpQuantizeActivationPerChannel(tensor.contiguous(), id, tag, stat_id, min_=min_value, max_=max_)
        else:
            alpha = self.get_alpha(tensor, tag, stat_id, clip_type, per_channel=False)
            max_value = float(max_value); min_value = float(min_value); mean = float(mean); alpha = float(alpha)
            range, min_value = self.alpha2DeltaOffset(alpha, max_value, min_value, mean)
            res = self.__gemmlowpQuantize1__(tensor.contiguous(), to_cuda(range, tensor.device), to_cuda(min_value, tensor.device))

        return res
