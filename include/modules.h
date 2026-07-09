#ifndef __MODULES_H__
#define __MODULES_H__

#include <fstream>

#include "datatypes.h"



class Linear {
private:
    vit_size in_features;
    vit_size out_features;
    vit_bool use_bias;

    Matrix A;
    RowVector b; // subject to use_bias

    // Device-resident copies of the weights, uploaded once (CUDA build only).
    // Default-initialised to nullptr so CPU/OpenMP builds see them safely unused.
    // `mutable` because operator() is const and uploads lazily on first use.
    mutable vit_float* d_A = nullptr;
    mutable vit_float* d_b = nullptr;
public:
    Linear(vit_size _in_features, vit_size _out_features, vit_bool _use_bias=true);
    Linear(const Linear& lin) = delete;
    Linear(Linear&& lin);
    ~Linear();

    Linear& operator= (const Linear& lin) = delete;
    Linear& operator= (Linear&& lin);
    void operator()(const Tensor& x_in, Tensor& x_out) const;

    vit_size get_in_features() const;
    vit_size get_out_features() const;
    vit_bool get_use_bias() const;

    void move_A(Matrix& _A);
    void move_b(RowVector& _b);

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



class LayerNorm {
private:
    vit_size normalized_shape;
    vit_float eps;
    vit_bool use_bias;

    RowVector g;
    RowVector b; // subject to use_bias
public:
    LayerNorm(vit_size _normalized_shape, vit_float _eps=0.00001, vit_bool _use_bias=true);
    LayerNorm(const LayerNorm& ln) = delete;
    LayerNorm(LayerNorm&& ln);
    ~LayerNorm();

    LayerNorm& operator= (const LayerNorm& ln) = delete;
    LayerNorm& operator= (LayerNorm&& ln);
    void operator()(Tensor& x) const;
    void operator()(Tensor& x, vit_size num_heads, vit_size head_dim) const;

    vit_size get_normalized_shape() const;
    vit_float get_eps() const;
    vit_bool get_use_bias() const;

    void move_g(RowVector& _g);
    void move_b(RowVector& _b);

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



class LayerScale {
private:
    vit_size dim;
    vit_float val;
public:
    LayerScale(vit_size _dim, vit_float _val=0.00001);
    LayerScale(const LayerScale& ln);
    ~LayerScale();

    LayerScale& operator= (const LayerScale& ln);
    void operator()(Tensor& x) const;

    vit_size get_dim() const;
    vit_float get_val() const;

    void set_val(vit_float _val);

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



class Activation {
private:
    vit_float (*act) (vit_float val);
public:
    Activation(vit_float (*_act)(vit_float val));
    Activation(const Activation& a);
    ~Activation();

    Activation& operator= (const Activation& a);
    void operator()(Tensor& x) const;

    void set_act(vit_float (*_act)(vit_float val));

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



vit_float ReLU(vit_float val);

vit_float GELU(vit_float val);

void global_pool_nlc (
    const Tensor& x_in, Tensor& x_out, pool_type pt,
    vit_size num_prefix_tokens=1, vit_bool reduce_include_prefix=false
);



#endif // __MODULES_H__
