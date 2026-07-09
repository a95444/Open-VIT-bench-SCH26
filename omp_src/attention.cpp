#ifdef _OPENMP
#include <omp.h>

#include "../include/attention.h"

#include "../include/datatypes.h"
#include "../include/modules.h"
#include "../include/vit_nvtx.h"

#include <utility>
#include <assert.h>
#include <math.h>



Attention::Attention (
    vit_size _dim,
    vit_size _num_heads,
    vit_bool use_qkv_bias,
    vit_bool _use_qk_norm
) :
    q_gen(_dim,_dim,use_qkv_bias),
    k_gen(_dim,_dim,use_qkv_bias),
    v_gen(_dim,_dim,use_qkv_bias),
    q_norm(_dim/_num_heads, 0.00001, true),
    k_norm(_dim/_num_heads, 0.00001, true),
    proj(_dim,_dim,true)
{
    assert(_dim % _num_heads == 0); // _dim must be divisible by _num_heads

    dim = _dim;
    num_heads = _num_heads;
    head_dim = _dim / _num_heads;
    scale = std::pow(head_dim, -0.5);
    use_qk_norm = _use_qk_norm;
}

Attention::Attention(Attention&& attn) :
    q_gen(std::move(attn.q_gen)),
    k_gen(std::move(attn.k_gen)),
    v_gen(std::move(attn.v_gen)),
    q_norm(std::move(attn.q_norm)),
    k_norm(std::move(attn.k_norm)),
    proj(std::move(attn.proj))
{
    dim = attn.dim;
    num_heads = attn.num_heads;
    head_dim = attn.head_dim;
    scale = attn.scale;
    use_qk_norm = attn.use_qk_norm;
}

Attention::~Attention() {}

Attention& Attention::operator= (Attention&& attn) {
    dim = attn.dim;
    num_heads = attn.num_heads;
    head_dim = attn.head_dim;
    scale = attn.scale;
    use_qk_norm = attn.use_qk_norm;

    q_gen = std::move(attn.q_gen);
    k_gen = std::move(attn.k_gen);
    v_gen = std::move(attn.v_gen);
    q_norm = std::move(attn.q_norm);
    k_norm = std::move(attn.k_norm);
    proj = std::move(attn.proj);

    return *this;
}

vit_size Attention::get_dim() const {
    return dim;
}

vit_size Attention::get_num_heads() const {
    return num_heads;
}

vit_size Attention::get_head_dim() const {
    return head_dim;
}

vit_float Attention::get_scale() const {
    return scale;
}

vit_bool Attention::get_use_qk_norm() const {
   return use_qk_norm;
}

void Attention::move_qkv_gen(Linear& _q_gen, Linear& _k_gen, Linear& _v_gen) {
    q_gen = std::move(_q_gen);
    k_gen = std::move(_k_gen);
    v_gen = std::move(_v_gen);
}

void Attention::move_norms(LayerNorm& _q_norm, LayerNorm& _k_norm) {
    q_norm = std::move(_q_norm);
    k_norm = std::move(_k_norm);
}

void Attention::move_proj(Linear& _proj) {
    proj = std::move(_proj);
}

void Attention::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );

    os.write( (char*) &dim, sizeof(vit_size));
    os.write( (char*) &num_heads, sizeof(vit_size));
    os.write( (char*) &head_dim, sizeof(vit_size));
    os.write( (char*) &scale, sizeof(vit_float));
    os.write( (char*) &use_qk_norm, sizeof(vit_bool));

    q_gen.to_ofstream(os);
    k_gen.to_ofstream(os);
    v_gen.to_ofstream(os);
    if (use_qk_norm == true) {
        q_norm.to_ofstream(os);
        k_norm.to_ofstream(os);
    }
    proj.to_ofstream(os);
}

void Attention::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );

    is.read( (char*) &dim, sizeof(vit_size));
    is.read( (char*) &num_heads, sizeof(vit_size));
    is.read( (char*) &head_dim, sizeof(vit_size));
    is.read( (char*) &scale, sizeof(vit_float));
    is.read( (char*) &use_qk_norm, sizeof(vit_bool));

    q_gen.from_ifstream(is);
    k_gen.from_ifstream(is);
    v_gen.from_ifstream(is);
    if (use_qk_norm == true) {
        q_norm.from_ifstream(is);
        k_norm.from_ifstream(is);
    }
    proj.from_ifstream(is);
}

void Attention::forward(const Tensor& x_in, Tensor& x_out) const {
    Tensor query, key, value;
    q_gen(x_in, query);
    k_gen(x_in, key);
    v_gen(x_in, value);

    if (use_qk_norm == true) {
        q_norm(query, num_heads, head_dim);
        k_norm(key, num_heads, head_dim);
    }

    multi_head_attention(query, key, value, scale, x_out, num_heads, head_dim);

    proj(x_out,x_out);
}

void Attention::single_head_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    vit_float _scale,
    Tensor& x_out
) const {
    this->multi_head_attention(query, key, value, _scale, x_out, 1, x_out.get_C());
}

void Attention::multi_head_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    vit_float _scale,
    Tensor& x_out,
    vit_size _num_heads,
    vit_size _head_dim
) const {
    assert(query.get_C() == _num_heads*_head_dim);

    assert(key.get_B() == query.get_B());
    assert(value.get_B() == query.get_B());
    assert(key.get_N() == query.get_N());
    assert(value.get_N() == query.get_N());
    assert(key.get_C() == query.get_C());
    assert(value.get_C() == query.get_C());

    vit_size N = query.get_N();
    Tensor qk(query.get_B(), N, N * _num_heads);
    Tensor y(query.get_B(), N, query.get_C());

    vit_float val;
    vit_float cumulative;
    VIT_NVTX_RANGE("attention"); // scaled dot-product core: QK^T, softmax, A*V
    #pragma omp parallel for collapse(2) private(val, cumulative) shared(y,N,_head_dim,query,key,_scale,qk,value) schedule(dynamic)
    for (int batch=0;batch<y.get_B();++batch) {
        for (int nh=0;nh<num_heads;++nh) {

            // qk is the matrix product query * key^T
            for (int q_n=0;q_n<N;++q_n) {
                for (int k_n=0;k_n<N;++k_n) {
                    val = 0;
                    for (int c=0;c<_head_dim;++c) {
                        val +=
                            query.at(batch, q_n, (nh*_head_dim) + c) *
                            key.at(batch, k_n, (nh*_head_dim) + c);
                    }
                    val *= _scale;
                    qk.set(batch, q_n, (nh*N) + k_n, val);
                }
            }

            // softmax of qk
            for (int qk_n=0;qk_n<N;++qk_n) {
                cumulative = 0;
                for (int qk_c=0;qk_c<N;++qk_c) { // qk is B*N*(N*nh), that's why it's C is also N
                    val = qk.at(batch, qk_n, (nh*N) + qk_c);
                    val = std::exp(val);
                    cumulative += val;
                    qk.set(batch, qk_n, (nh*N) + qk_c, val);
                }
                for (int qk_c=0;qk_c<N;++qk_c) {
                    val = qk.at(batch, qk_n, (nh*N) + qk_c);
                    val /= cumulative;
                    qk.set(batch, qk_n, (nh*N) + qk_c, val);
                }
            }

            // y is the matrix product of qk * value
            for (int qk_n=0;qk_n<N;++qk_n) {
                for (int v_c=0;v_c<_head_dim;++v_c) {
                    val = 0;
                    for (int qk_c=0;qk_c<N;++qk_c) { // qk_c is also v_n
                        val +=
                            qk.at(batch, qk_n, (nh*N) + qk_c) *
                            value.at(batch, qk_c, (nh*_head_dim) + v_c);
                    }
                    y.set(batch, qk_n, (nh*_head_dim) + v_c, val);
                }
            }
        }
    }

    x_out = std::move(y);
}

#else

#error "Error: omp sources must be compiled with -fopenmp compiler flag!"

#endif
