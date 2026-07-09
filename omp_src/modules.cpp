#ifdef _OPENMP
#include <omp.h>

#include "../include/modules.h"

#include "../include/datatypes.h"
#include "../include/vit_nvtx.h"

#include <utility>
#include <assert.h>
#include <math.h>



// Linear class

Linear::Linear(vit_size _in_features, vit_size _out_features, vit_bool _use_bias) : A(), b() {
    in_features = _in_features;
    out_features = _out_features;
    use_bias = _use_bias;
}

Linear::Linear(Linear&& lin) : A(std::move(lin.A)), b(std::move(lin.b)) {
    in_features = lin.in_features;
    out_features = lin.out_features;
    use_bias = lin.use_bias;
}

Linear::~Linear() {}

Linear& Linear::operator= (Linear&& lin) {
    in_features = lin.in_features;
    out_features = lin.out_features;
    use_bias = lin.use_bias;

    A = std::move(lin.A);
    b = std::move(lin.b);

    return *this;
}

void Linear::operator()(const Tensor& x_in, Tensor& x_out) const {
    VIT_NVTX_RANGE("linear");
    assert(A.get_ROWS() == out_features);
    assert(A.get_COLS() == in_features);
    assert(x_in.get_C() == in_features);
    if (use_bias == true) {
        assert(b.get_DIM() == out_features);
    }

    Tensor y(x_in.get_B(), x_in.get_N(), out_features);

    vit_float cumulate;
    #pragma omp parallel for collapse(3) private(cumulate) shared(y,x_in) schedule(static)
    for (int i=0;i<y.get_B();++i) {
        for (int j=0;j<y.get_N();++j) {
            for (int k=0;k<y.get_C();++k) {
                cumulate = use_bias==true ? b.at(k) : 0;

                for (int l=0;l<x_in.get_C();++l) {
                    cumulate += x_in.at(i,j,l) * A.at(k,l);
                }

                y.set(i,j,k,cumulate);
            }
        }
    }

    x_out = std::move(y);
}

vit_size Linear::get_in_features() const {
    return in_features;
}

vit_size Linear::get_out_features() const {
    return out_features;
}

vit_bool Linear::get_use_bias() const {
    return use_bias;
}

void Linear::move_A(Matrix& _A) {
    A = std::move(_A);
}

void Linear::move_b(RowVector& _b) {
    b = std::move(_b);
}

void Linear::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );

    os.write( (char*) &in_features, sizeof(vit_size));
    os.write( (char*) &out_features, sizeof(vit_size));
    os.write( (char*) &use_bias, sizeof(vit_bool));

    A.to_ofstream(os);
    if (use_bias == true) {
        b.to_ofstream(os);
    }
}

void Linear::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );

    is.read( (char*) &in_features, sizeof(vit_size));
    is.read( (char*) &out_features, sizeof(vit_size));
    is.read( (char*) &use_bias, sizeof(vit_bool));

    A.from_ifstream(is);
    if (use_bias == true) {
        b.from_ifstream(is);
    }
}



// LayerNorm class

LayerNorm::LayerNorm(
    vit_size _normalized_shape,
    vit_float _eps,
    vit_bool _use_bias
) : g(), b() {
    normalized_shape = _normalized_shape;
    eps = _eps;
    use_bias = _use_bias;
}

LayerNorm::LayerNorm(LayerNorm&& ln) : g(std::move(ln.g)), b(std::move(ln.b)) {
    normalized_shape = ln.normalized_shape;
    eps = ln.eps;
    use_bias = ln.use_bias;
}

LayerNorm::~LayerNorm() {}

LayerNorm& LayerNorm::operator= (LayerNorm&& ln) {
    normalized_shape = ln.normalized_shape;
    eps = ln.eps;
    use_bias = ln.use_bias;

    g = std::move(ln.g);
    b = std::move(ln.b);

    return *this;
}

void LayerNorm::operator()(Tensor& x) const {
    this->operator()(x,1,x.get_C());
}

void LayerNorm::operator()(Tensor& x, vit_size num_heads, vit_size head_dim) const {
    VIT_NVTX_RANGE("layernorm");
    assert(x.get_C() == head_dim*num_heads);
    assert(g.get_DIM() == head_dim);
    if (use_bias == true) {
        assert(b.get_DIM() == head_dim);
    }

    vit_float mean;
    vit_float var;
    vit_float st_dev;
    vit_float new_val;
    #pragma omp parallel for collapse(3) private(mean, var, st_dev, new_val) shared(x,num_heads,head_dim) schedule(dynamic)
    for (int i=0;i<x.get_B();++i) {
        for (int j=0;j<x.get_N();++j) {
            for (int k=0;k<num_heads;++k) {
                mean = 0.0;
                for (int l=0;l<head_dim;++l) {
                    mean += x.at(i,j, (k*head_dim) + l);
                }
                mean /= (float) head_dim;

                var = 0.0;
                for (int l=0;l<head_dim;++l) {
                    var += std::pow( x.at(i,j, (k*head_dim) + l) - mean, 2);
                }
                var /= (float) head_dim;
                st_dev = 1.0 / std::sqrt( var + eps);

                for (int l=0;l<head_dim;++l) {
                    new_val = x.at(i,j, (k*head_dim) + l);
                    new_val = (new_val - mean) * st_dev * g.at(l);
                    new_val += use_bias==true ? b.at(l) : 0;
                    x.set(i,j, (k*head_dim) + l, new_val);
                }
            }
        }
    }

}

vit_size LayerNorm::get_normalized_shape() const {
    return normalized_shape;
}

vit_float LayerNorm::get_eps() const {
    return eps;
}

vit_bool LayerNorm::get_use_bias() const {
    return use_bias;
}

void LayerNorm::move_g(RowVector& _g) {
    g = std::move(_g);
}

void LayerNorm::move_b(RowVector& _b) {
    b = std::move(_b);
}

void LayerNorm::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );

    os.write( (char*) &normalized_shape, sizeof(vit_size));
    os.write( (char*) &eps, sizeof(vit_float));
    os.write( (char*) &use_bias, sizeof(vit_bool));

    g.to_ofstream(os);
    if (use_bias == true) {
        b.to_ofstream(os);
    }
}

void LayerNorm::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );

    is.read( (char*) &normalized_shape, sizeof(vit_size));
    is.read( (char*) &eps, sizeof(vit_float));
    is.read( (char*) &use_bias, sizeof(vit_bool));

    g.from_ifstream(is);
    if (use_bias == true) {
        b.from_ifstream(is);
    }
}



// LayerScale class

LayerScale::LayerScale(vit_size _dim, vit_float _val) {
    dim = _dim;
    val = _val;
}

LayerScale::LayerScale(const LayerScale& ln) {
    dim = ln.dim;
    val = ln.val;
}

LayerScale::~LayerScale() {}

LayerScale& LayerScale::operator= (const LayerScale& ln) {
    dim = ln.dim;
    val = ln.val;

    return *this;
}

void LayerScale::operator()(Tensor& x) const {
    #pragma omp parallel for collapse(3) shared(x) schedule(static)
    for (int i=0;i<x.get_B();++i) {
        for (int j=0;j<x.get_N();++j) {
            for (int k=0;k<x.get_C();++k) {
                x.set(i,j,k, x.at(i,j,k) * val );
            }
        }
    }
}

vit_size LayerScale::get_dim() const {
    return dim;
}

vit_float LayerScale::get_val() const {
    return val;
}

void LayerScale::set_val(vit_float _val) {
    val = _val;
}

void LayerScale::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );

    os.write( (char*) &dim, sizeof(vit_size));
    os.write( (char*) &val, sizeof(vit_float));
}

void LayerScale::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );

    is.read( (char*) &dim, sizeof(vit_size));
    is.read( (char*) &val, sizeof(vit_float));
}



// Activation class

Activation::Activation(vit_float (*_act)(vit_float val)) {
    act = _act;
}

Activation::Activation(const Activation& a) {
    act = a.act;
}

Activation::~Activation() {}

Activation& Activation::operator= (const Activation& a) {
    act = a.act;

    return *this;
}

void Activation::operator()(Tensor& x) const {
    VIT_NVTX_RANGE("gelu");
    vit_float val;
    #pragma omp parallel for collapse(3) private(val) shared(x) schedule(static)
    for (int i=0;i<x.get_B();++i) {
        for (int j=0;j<x.get_N();++j) {
            for (int k=0;k<x.get_C();++k) {
                val = x.at(i,j,k);
                val = act(val);
                x.set(i,j,k,val);
            }
        }
    }
}

void Activation::set_act(vit_float (*_act)(vit_float val)) {
    act = _act;
}

void Activation::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );

    vit_size act_type;
    if (act == ReLU) {
        act_type = 1;
    } else if (act == GELU) {
        act_type = 2;
    } else {
        act_type = 0;
    }

    os.write( (char*) &act_type, sizeof(vit_size));
}

void Activation::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );

    vit_size act_type;
    is.read( (char*) &act_type, sizeof(vit_size));
    assert(act_type != 0);

    switch (act_type) {
        case 1: act = ReLU;
        break;
        case 2: act = GELU;
        break;
    }
}



// Other functions

vit_float ReLU(vit_float val) {
    return val>0 ? val : 0;
}

vit_float GELU(vit_float val) {
    return 0.5 * val * (
        1 + std::tanh(
            std::sqrt(2.0/M_PI) * (
                val + ( 0.044715 * std::pow(val,3) )
            )
        )
    );
}

void global_pool_nlc (
    const Tensor& x_in,
    Tensor& x_out,
    pool_type pt,
    vit_size num_prefix_tokens,
    vit_bool reduce_include_prefix
) {
    VIT_NVTX_RANGE("pool");
    vit_float val;
    vit_float max_val;
    vit_float avg_val;

    Tensor y(x_in.get_B(), 1, x_in.get_C() );

    vit_size start_N = reduce_include_prefix==true ? 0 : num_prefix_tokens;
    assert(start_N<x_in.get_N() || pt==pool_token);
    vit_size N_length = x_in.get_N() - start_N;

    switch (pt) {
    case pool_token:
        #pragma omp parallel for collapse(2) private(val) shared(x_in,y) schedule(static)
        for (int i=0;i<x_in.get_B();++i) {
            for (int k=0;k<x_in.get_C();++k) {
                val = x_in.at(i, 0, k);
                y.set(i, 0, k, val);
            }
        }
        break;
    case pool_avg:
        #pragma omp parallel for collapse(2) private(avg_val) shared(x_in,start_N,N_length,y) schedule(static)
        for (int i=0;i<x_in.get_B();++i) {
            for (int k=0;k<x_in.get_C();++k) {
                avg_val = x_in.at(i, start_N, k);
                for (int j=start_N+1;j<x_in.get_N();++j) {
                    avg_val += x_in.at(i, j, k);
                }
                avg_val /= N_length;
                y.set(i, 0, k, avg_val);
            }
        }
        break;
    case pool_max:
        #pragma omp parallel for collapse(2) private(max_val,val) shared(x_in,start_N,y) schedule(static)
        for (int i=0;i<x_in.get_B();++i) {
            for (int k=0;k<x_in.get_C();++k) {
                max_val = x_in.at(i, start_N, k);
                for (int j=start_N+1;j<x_in.get_N();++j) {
                    val = x_in.at(i, j, k);
                    max_val = max_val>val ? max_val : val;
                }
                y.set(i, 0, k, max_val);
            }
        }
        break;
    case pool_avgmax:
        #pragma omp parallel for collapse(2) private(avg_val,max_val,val) shared(x_in,start_N,N_length,y) schedule(static)
        for (int i=0;i<x_in.get_B();++i) {
            for (int k=0;k<x_in.get_C();++k) {
                avg_val = x_in.at(i, start_N, k);
                max_val = x_in.at(i, start_N, k);
                for (int j=start_N+1;j<x_in.get_N();++j) {
                    val = x_in.at(i, j, k);
                    max_val = max_val>val ? max_val : val;
                    avg_val += val;
                }
                avg_val /= N_length;
                val = (max_val + avg_val) * 0.5;
                y.set(i, 0, k, val);
            }
        }
        break;
    }

    x_out = std::move(y);
}

#else

#error "Error: omp sources must be compiled with -fopenmp compiler flag!"

#endif
