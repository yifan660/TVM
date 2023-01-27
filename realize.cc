/*
def get_omega(sigma, target_bins):
        B = len(sigma) * target_bins

        # Calculate bit allocation
        p = sigma ** (2./3)
        omega = (B * p) / p.sum()
        return omega

    @staticmethod
    def get_alpha_mult(omega, sym=True):
        omega = omega.cpu().numpy()
        if not sym:
            omega *= 2
        i = omega_table.searchsorted(omega)
        inc = (alpha_table[i] - alpha_table[i - 1]) / (omega_table[i] - omega_table[i - 1])
        alpha = alpha_table[i] - inc * (omega_table[i] - omega)
        return alpha
*/
// sigma: standard deviation of tensor value


/*

class StatisticalClipper(Function):
    def __init__(self, rho):
        self.rho = rho

    def __call__(self, tensor, tag="", stat_id=None, inplace=False):
        cls_layer = (tag == 'activation_linear' and tensor.shape[1] == 1000)
        if stat_id is not None and not cls_layer:
            kind_max = {'min': 'min', 'max': 'max', 'mean': 'mean', 'std': 'mean', 'range': 'mean', 'mean_abs': 'mean',
                    'b': 'mean'}
            kind_avg = {'min': 'mean', 'max': 'mean', 'mean': 'mean', 'std': 'mean', 'range': 'mean', 'mean_abs': 'mean',
                    'b': 'mean'}
            min_min, max_max, _, _, _, _, _ = SM().get_tensor_stats(stat_id, kind_max)
            avg_min, avg_max, _, _, _, _, _ = SM().get_tensor_stats(stat_id, kind_avg)

            if avg_min == 0 or avg_max == 0:
                # Do not clip not symetrical activations
                return tensor

            k1 = max_max / avg_max if avg_max != 0 else self.rho * max_max
            k2 = min_min / avg_min if avg_min != 0 else self.rho * min_min
            max_ = self.rho * max_max / k1 if k1 != 0 else max_max
            min_ = self.rho * min_min / k2 if k2 != 0 else min_min
            if (max_ > 0 and min_ < 0):
                # clip symetrical range only
                maxabs = torch.max(torch.abs(tensor.max()), torch.abs(tensor.min()))
                upper_bound = maxabs * self.rho
                lower_bound = -maxabs * self.rho
                if inplace:
                    tensor.clamp_(lower_bound, upper_bound)
                else:
                    tensor = torch.clamp(tensor, lower_bound, upper_bound)

        return tensor


class RatioClipper(Function):
    def __init__(self, rho):
        self.rho = rho

    def __call__(self, tensor, tag="", inplace=False):
        max_ = tensor.max()
        min_ = tensor.min()
        if (max_ > 0 and min_ < 0):
            # clip symetrical range only
            maxabs = torch.max(torch.abs(tensor.max()), torch.abs(tensor.min()))
            upper_bound = maxabs * self.rho
            lower_bound = -maxabs * self.rho
            if inplace:
                tensor.clamp_(lower_bound, upper_bound)
            else:
                tensor = torch.clamp(tensor, lower_bound, upper_bound)

        return tensor

*/
StatisticalClipper()
{
    rho = <ClipAttrs>
    max - min
    k1 = max_max / avg_max if avg_max != 0 else self.rho * max_max
    k2 = min_min / avg_min if avg_min != 0 else self.rho * min_min
    max_ = self.rho * max_max / k1 if k1 != 0 else max_max
    min_ = self.rho * min_min / k2 if k2 != 0 else min_min

}
Expr ACIQRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx)   {
    // ClipAttrs - include/tvm/relay/attrs/transform.h
    // new_args is tensor
    ICHECK_EQ();
    if(const auto* n = new_args[0].as<QRealizeIntExprNode>())   {
        const auto ref_attrs = ref_call->attrs.as<ClipAttrs>();
        auto attrs = make_object<ClipAttrs>();
        double dom_scale = GetScalarFromConstant<float>(n->dom_scale);
        // sigma is array of value
        // value of all tensor
        attrs->a_min = (ref_attrs->a_min/dom_scale)^2/3;
        attrs->a_max = (ref_attrs->a_max/dom_scale)^2/3;
        // logic
        // get standard deviation from each tensor
        // find omega using omega = B*p/p.sum()
        for(auto new_args)   {
            calculate standard deviation 
            get from attrs.
            uint32_t sum += new_args[1].value_index;
            uint32_t count++;
        }
        target_bins = n->target_bins;
        B = count*target_bins;
        sigma = new_args->


        }
        Expr ret = Call(ref_call->op, {n->data}, Attrs(attrs), ref_call->type_args);
        return QRealizeIntExpr(ret, n->dom_scale, n->dtype);
    }
    ICHECK();
    return Expr();
    
}

RELAY_REGISTER_OP("ACIQClip").set_attr<FForwardRewrite>("FQRealizeRewrite", ACIQClip);
