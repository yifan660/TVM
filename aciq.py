class IntQuantizer(Function):
    def __init__(self, size, params):
        self.num_bits = size
        # TODO: expose as cmd line parameters
        self.stochastic = False
        self.int_exp = False
        self.enforce_true_zero = True #params['true_zero']
        self.clipping = params['clipping'] if 'clipping' in params else 'no'
        self.stats_kind = params['stats_kind'] if 'stats_kind' in params else 'mean'
        self.kld = params['kld'] if 'kld' in params else False
        self.pcq_w = params['pcq_weights']
        self.pcq_a = params['pcq_act']
        self.bit_alloc_act = params['bit_alloc_act']
        self.bit_alloc_weight = params['bit_alloc_weight']
        self.bcorr_act = params['bcorr_act']
        self.bcorr_weight = params['bcorr_weight']
        self.vcorr_weight = params['vcorr_weight']
        self.bit_alloc_round = params['bit_alloc_rmode'] == 'round'
        self.bit_alloc_prior = params['bit_alloc_prior']
        self.bit_alloc_target_act = params['bit_alloc_target_act'] if params['bit_alloc_target_act'] is not None else self.num_bits
        self.bit_alloc_target_weight = params['bit_alloc_target_weight'] if params['bit_alloc_target_weight'] is not None else self.num_bits
        self.measure_entropy = params['measure_entropy']
        self.logger = params['logger']
        self.mtd_quant = params['mtd_quant']

        self.alpha_gaus = {1: 1.24, 2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92}
        self.alpha_gaus_positive = {1: 1.71, 2: 2.15, 3: 2.55, 4: 2.93, 5: 3.28, 6: 3.61, 7: 3.92, 8: 4.2}

        self.alpha_laplace = {0: 1.05, 1: 1.86, 2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}
        self.alpha_laplace_positive = {0: 1.86, 1: 2.83, 2: 3.89, 3: 5.02, 4: 6.2, 5: 7.41, 6: 8.64, 7: 9.89, 8: 11.16}



    def __call__(self, tensor, id, tag="", stat_id=None, override_att=None):
        if override_att is not None:
            orig_att = getattr(self, override_att[0])
            setattr(self, override_att[0], override_att[1])
        if self.kld:
            res = self.gemmlowpKldQuantize(tensor, tag, stat_id=stat_id)
        elif self.clipping != 'no':
            # print("clipping %s: %d" % (tag, self.num_bits))
            if self.mtd_quant:
                res = self.mid_tread_quantize_activation(tensor, id)
            else:
                res = self.gemmlowpClippingQuantize(tensor, id, tag, stat_id=stat_id, clip_type=self.clipping)
        elif self.pcq_w:
            # print("pcq_w %s: %d" % (tag, self.num_bits))
            if self.mtd_quant:
                res = self.mid_tread_quantize_weights_per_channel(tensor, id)
            else:
                res = self.gemmlowpQuantizeWeightsPerChannel(tensor, id)
        elif self.pcq_a and len(tensor.shape) > 3 and (tensor.shape[2] > 1 or tensor.shape[3] > 1):
            # print("pcq_a %s: %d" % (tag, self.num_bits))
            if self.mtd_quant:
                res = self.mid_tread_quantize_activation_per_channel(tensor, id)
            else:
                res = self.gemmlowpQuantizeActivationPerChannel(tensor, id, tag, stat_id=stat_id)
        else:
            # print("no clipping %s: %d" % (tag, self.num_bits))
            res = self.gemmlowpMinMaxQuantize(tensor, tag, stat_id=stat_id)

        if override_att is not None:
            setattr(self, override_att[0], orig_att)
        return res


def __gemmlowpQuantize1__(self, tensor, delta, offset, bit_alloc=None, measure_entropy=False):
    qmin = 0.
    if bit_alloc is None:
        qmax = 2.**self.num_bits - 1.
        scale = (delta) / (qmax - qmin)
    else:
        qmax = 2.**bit_alloc - 1.
        scale = torch.where(qmax > 0, (delta) / (qmax - qmin), torch.tensor([0.]).to(tensor.device))

    scale = torch.max(scale, torch.tensor([1e-8]).to(tensor.device))

  output = tensor.detach()
  if self.enforce_true_zero:
    initial_zero_point = qmin - offset / scale
    # make zero exactly represented
    zero_point = torch.round(initial_zero_point)
    output = torch.div(output, scale.unsqueeze(-1))
    output = torch.add(output, zero_point.unsqueeze(-1))
  else:
    output = torch.add(output, -offset.unsqueeze(-1))
    output = torch.div(output, scale.unsqueeze(-1))

  if bit_alloc is None:
    output.clamp_(qmin, qmax).round_()  # quantize
  else:
    qmax = qmax.view(qmax.numel(), 1)
    output = torch.where(output.gt(qmax), qmax, output)
    output.clamp_(qmin).round_()

  if measure_entropy:
    entropy = shannon_entropy(output.int())
    # entropy = most_requent_value_compression(output.int())

  if self.enforce_true_zero:
    output = torch.add(output, -zero_point.unsqueeze(-1))
    output = torch.mul(output, scale.unsqueeze(-1))  # dequantize
  else:
    output = torch.mul(output, scale.unsqueeze(-1))
    output = torch.add(output, offset.unsqueeze(-1))  # dequantize

  # workaround for out of memory issue
  torch.cuda.empty_cache()

  if measure_entropy:
    return output.view(tensor.shape), entropy
  else:
    return output.view(tensor.shape)



# Below is code for ACIQ
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

  if self.pcq_a and len(tensor.shape) > 3 and (tensor.shape[2] > 1 or tensor.shape[3] > 1) and len(min_value) > 1 and len(max_value) > 1:
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



def gemmlowpQuantizeActivationPerChannel(self, tensor, id, tag="", stat_id=None, min_=None, max_=None):
  if min_ is None:
    if self.force_positive or self.half_range:
      min_ = 0  # np.zeros(min_.shape)
    elif stat_id is not None:
      min_ = self.sm().get_tensor_stat(stat_id, 'min', kind=self.stats_kind)
    else:
      min_ = self.__act_stats_perchannel__(tensor, ['min'], avg_over_batch=False)['min']
  min_ = to_cuda(min_, tensor.device)

  if max_ is None:
    if stat_id is not None:
      max_ = self.sm().get_tensor_stat(stat_id, 'max', kind=self.stats_kind)
    else:
      max_ = self.__act_stats_perchannel__(tensor, ['max'], avg_over_batch=False)['max']
  max_ = to_cuda(max_, tensor.device)

  N, C, H, W = tensor.shape  # N x C x H x W
  t = tensor.detach().transpose(0, 1).contiguous()  # C x N x H x W
  t = t.view(t.shape[0], -1)

  if self.bit_alloc_act and self.num_bits <= 4:
    prior = 'std' if self.bit_alloc_prior == 'gaus' else 'b'
    if stat_id is not None:
      alpha = self.sm().get_tensor_stat(stat_id, prior, kind='mean')
      alpha = to_cuda(alpha, tensor.device)
    else:
      alpha = self.__act_stats_perchannel__(tensor, [prior], avg_over_batch=False)[prior]

    bit_alloc = self.get_bits_alloc_fixed_target(alpha, self.bit_alloc_target_act, self.bit_alloc_round)
  else:
    bit_alloc = None

  if self.measure_entropy:
    output, entropy = self.__gemmlowpQuantize1__(t, max_ - min_, min_, bit_alloc=bit_alloc, measure_entropy=True)
    if self.logger is not None:
      self.logger.log_metric(id + '.entropy', entropy.item(), step='auto', meterId='avg.entropy.act', weight=output.numel())
  else:
    output = self.__gemmlowpQuantize1__(t, max_ - min_, min_, bit_alloc=bit_alloc, measure_entropy=self.measure_entropy)

  output = output.view(C, N, H, W).transpose(0, 1).contiguous()  # N x C x H x W
  return output.view(tensor.shape)



def gemmlowpQuantizeWeightsPerChannel(self, tensor, id, min_=None, max_=None):
  # Assume weights with dimensions [OFM,IFM,K1,K2]
  t = tensor.view(tensor.shape[0], -1)

  # per output channel min, max
  if min_ is None:
    min_ = t.min(-1)[0]
  if max_ is None:
    max_ = t.max(-1)[0]

  if self.bit_alloc_weight and self.num_bits <= 4:
    alpha = t.std(-1)
    bit_alloc = self.get_bits_alloc_fixed_target(alpha, self.bit_alloc_target_weight, self.bit_alloc_round)
  else:
    bit_alloc = None

  if self.measure_entropy:
    output, entropy = self.__gemmlowpQuantize1__(t, max_ - min_, min_, bit_alloc=bit_alloc, measure_entropy=True)
    if self.logger is not None:
      self.logger.log_metric(id + '.entropy', entropy.item(), step='auto', meterId='avg.entropy.weight', weight=output.numel())
  else:
    output = self.__gemmlowpQuantize1__(t, max_ - min_, min_, bit_alloc=bit_alloc)

  return output.view(tensor.shape)



