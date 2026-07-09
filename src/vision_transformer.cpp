#include "../include/vision_transformer.h"

#include "../include/datatypes.h"
#include "../include/modules.h"
#include "../include/block.h"
#include "../include/patch_embed.h"
#include "../include/vit_nvtx.h"

#include <vector>
#include <utility>
#include <assert.h>
#include <chrono>



VisionTransformer::VisionTransformer(
    vit_size img_size_h,
    vit_size img_size_w,
    vit_size patch_size_h,
    vit_size patch_size_w,
    vit_size in_chans,

    vit_size _num_classes,
    pool_type _global_pool,
    vit_size _embed_dim,
    vit_size _depth,

    vit_size num_heads,
    vit_float mlp_ratio,
    vit_bool use_qkv_bias,
    vit_bool use_qk_norm,
    vit_float scale_val,

    vit_bool _has_class_token,
    vit_size _num_reg_tokens,
    vit_bool _no_embed_class,

    vit_bool _use_pos_embed,
    vit_bool _use_pre_norm,
    vit_bool _use_fc_norm,

    vit_bool _dynamic_img_size,
    vit_bool dynamic_img_pad,
    vit_float (*act)(vit_float val)
) :
    cls_token(), reg_token(), pos_embed(),
    patch_embed(img_size_h, img_size_w, patch_size_h, patch_size_w, in_chans,
                _embed_dim, !_use_pre_norm, !_dynamic_img_size, dynamic_img_pad, false),
    pre_norm(_embed_dim, 0.000001, true),
    blocks(),
    norm(_embed_dim, 0.000001, true),
    fc_norm(_embed_dim, 0.000001, true),
    head(_embed_dim, _num_classes, true)
{
    assert(_has_class_token == true || _global_pool != pool_type::pool_token);
    // pool_token needs _class_token = true to work

    assert(_num_classes > 0); // _num_classes can't be zero
    num_classes = _num_classes;
    global_pool = _global_pool;
    embed_dim = _embed_dim;
    depth = _depth;

    for (int i=0;i<_depth;++i) {
        blocks.emplace_back(
            _embed_dim,
            num_heads,
            mlp_ratio,
            use_qkv_bias,
            use_qk_norm,
            scale_val,
            act
        );
    }

    has_class_token = _has_class_token;
    num_reg_tokens = _num_reg_tokens;
    num_prefix_tokens = _has_class_token==true ? 1 : 0;
    num_prefix_tokens += _num_reg_tokens;
    no_embed_class = _no_embed_class;

    // init cls_token and reg_token at zeroes, with dim:
    // 1 x 1 x _embed_dim if _class_token
    // 1 x _reg_tokens x _embed_dim if _reg_tokens > 0

    // vit_size num_patches = patch_embed.get_num_patches();
    //vit_size embed_len = no_embed_class==true ? num_patches : num_patches + num_prefix_tokens;
    // init pos_embed and at random, with dim:
    // 1 x embed_len x _embed_dim if _class_token
    // scaled by 0.02

    use_pos_embed = _use_pos_embed;
    use_pre_norm = _use_pre_norm;
    use_fc_norm = _use_fc_norm;
    dynamic_img_size = _dynamic_img_size;
}

VisionTransformer::VisionTransformer(VisionTransformer&& vit) :
    cls_token(std::move(vit.cls_token)),
    reg_token(std::move(vit.reg_token)),
    pos_embed(std::move(vit.pos_embed)),

    patch_embed(std::move(vit.patch_embed)),
    pre_norm(std::move(vit.pre_norm)),
    blocks(std::move(vit.blocks)),
    norm(std::move(vit.norm)),
    fc_norm(std::move(vit.fc_norm)),
    head(std::move(vit.head))
{
    num_classes = vit.num_classes;
    global_pool = vit.global_pool;
    embed_dim = vit.embed_dim;
    depth = vit.depth;

    has_class_token = vit.has_class_token;
    num_reg_tokens = vit.num_reg_tokens;
    num_prefix_tokens = vit.num_prefix_tokens;
    no_embed_class = vit.no_embed_class;

    use_pos_embed = vit.use_pos_embed;
    use_pre_norm = vit.use_pre_norm;
    use_fc_norm = vit.use_fc_norm;
    dynamic_img_size = vit.dynamic_img_size;
}

VisionTransformer::~VisionTransformer() {}

VisionTransformer& VisionTransformer::operator= (VisionTransformer&& vit) {
    num_classes = vit.num_classes;
    global_pool = vit.global_pool;
    embed_dim = vit.embed_dim;
    depth = vit.depth;

    has_class_token = vit.has_class_token;
    num_reg_tokens = vit.num_reg_tokens;
    num_prefix_tokens = vit.num_prefix_tokens;
    no_embed_class = vit.no_embed_class;

    use_pos_embed = vit.use_pos_embed;
    use_pre_norm = vit.use_pre_norm;
    use_fc_norm = vit.use_fc_norm;
    dynamic_img_size = vit.dynamic_img_size;

    cls_token = std::move(vit.cls_token);
    reg_token = std::move(vit.reg_token);
    pos_embed = std::move(vit.pos_embed);

    patch_embed = std::move(vit.patch_embed);
    pre_norm = std::move(vit.pre_norm);
    blocks = std::move(vit.blocks);
    norm = std::move(vit.norm);
    fc_norm = std::move(vit.fc_norm);
    head = std::move(vit.head);

    return *this;
}

vit_size VisionTransformer::get_num_classes() const { return num_classes; }
pool_type VisionTransformer::get_global_pool() const { return global_pool; }
vit_size VisionTransformer::get_embed_dim() const { return embed_dim; }
vit_size VisionTransformer::get_depth() const { return depth; }

vit_bool VisionTransformer::get_has_class_token() const { return has_class_token; }
vit_size VisionTransformer::get_num_reg_tokens() const { return num_reg_tokens; }
vit_size VisionTransformer::get_num_prefix_tokens() const { return num_prefix_tokens; }
vit_bool VisionTransformer::get_no_embed_class() const { return no_embed_class; }

vit_bool VisionTransformer::get_use_pos_embed() const { return use_pos_embed; }
vit_bool VisionTransformer::get_use_pre_norm() const { return use_pre_norm; }
vit_bool VisionTransformer::get_use_fc_norm() const { return use_fc_norm; }
vit_bool VisionTransformer::get_dynamic_img_size() const { return dynamic_img_size; }

void VisionTransformer::move_cls_token(RowVector _cls_token) {
    cls_token = std::move(_cls_token);
}

void VisionTransformer::move_reg_token(Matrix _reg_token) {
    reg_token = std::move(_reg_token);
}

void VisionTransformer::move_pos_embed(Matrix _pos_embed) {
    pos_embed = std::move(_pos_embed);
}

void VisionTransformer::move_patch_embed(PatchEmbed _patch_embed) {
    patch_embed = std::move(_patch_embed);
}

void VisionTransformer::move_pre_norm(LayerNorm _pre_norm) {
    pre_norm = std::move(_pre_norm);
}

void VisionTransformer::move_blocks(std::vector<Block> _blocks) {
    depth = _blocks.size();
    blocks = std::move(_blocks);
}

void VisionTransformer::move_norm(LayerNorm _norm) {
    norm = std::move(_norm);
}

void VisionTransformer::move_fc_norm(LayerNorm _fc_norm) {
    fc_norm = std::move(_fc_norm);
}

void VisionTransformer::move_head(Linear _head) {
    head = std::move(_head);
}

void VisionTransformer::reset_classifier(
    Linear _head, vit_size _num_classes, pool_type _global_pool
) {
    assert(_head.get_out_features() == _num_classes);
    head = std::move(_head);
    num_classes = _num_classes;
    if (_global_pool == pool_token) {
        assert(has_class_token == true);
    }
    global_pool = _global_pool;
}

void VisionTransformer::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );

    os.write( (char*) &num_classes, sizeof(vit_size));

    vit_size pool;
    switch (global_pool) {
        case pool_token: pool = 1;
        break;
        case pool_avg: pool = 2;
        break;
        case pool_avgmax: pool = 3;
        break;
        case pool_max: pool = 4;
        break;
        default: pool = 0;
        break;
    }
    os.write( (char*) &pool, sizeof(vit_size));


    os.write( (char*) &embed_dim, sizeof(vit_size));
    os.write( (char*) &depth, sizeof(vit_size));

    os.write( (char*) &has_class_token, sizeof(vit_bool));
    os.write( (char*) &num_reg_tokens, sizeof(vit_size));
    os.write( (char*) &num_prefix_tokens, sizeof(vit_size));
    os.write( (char*) &no_embed_class, sizeof(vit_bool));

    os.write( (char*) &use_pos_embed, sizeof(vit_bool));
    os.write( (char*) &use_pre_norm, sizeof(vit_bool));
    os.write( (char*) &use_fc_norm, sizeof(vit_bool));
    os.write( (char*) &dynamic_img_size, sizeof(vit_bool));

    if (has_class_token == true) { cls_token.to_ofstream(os); }
    if (num_reg_tokens > 0) { reg_token.to_ofstream(os); }
    if (use_pos_embed == true) { pos_embed.to_ofstream(os); }

    patch_embed.to_ofstream(os);
    if (use_pre_norm == true) { pre_norm.to_ofstream(os); }

    for (int i=0;i<depth;++i) {
        blocks.at(i).to_ofstream(os);
    }

    if (use_fc_norm == true) {
        fc_norm.to_ofstream(os);
    } else {
        norm.to_ofstream(os);
    }
    head.to_ofstream(os);
}

void VisionTransformer::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );

    is.read( (char*) &num_classes, sizeof(vit_size));

    vit_size pool;
    is.read( (char*) &pool, sizeof(vit_size));
    assert(pool != 0);
    switch (pool) {
        case 1: global_pool = pool_token;
        break;
        case 2: global_pool = pool_avg;
        break;
        case 3: global_pool = pool_avgmax;
        break;
        case 4: global_pool = pool_max;
        break;
    }

    is.read( (char*) &embed_dim, sizeof(vit_size));
    is.read( (char*) &depth, sizeof(vit_size));

    is.read( (char*) &has_class_token, sizeof(vit_bool));
    is.read( (char*) &num_reg_tokens, sizeof(vit_size));
    is.read( (char*) &num_prefix_tokens, sizeof(vit_size));
    is.read( (char*) &no_embed_class, sizeof(vit_bool));

    is.read( (char*) &use_pos_embed, sizeof(vit_bool));
    is.read( (char*) &use_pre_norm, sizeof(vit_bool));
    is.read( (char*) &use_fc_norm, sizeof(vit_bool));
    is.read( (char*) &dynamic_img_size, sizeof(vit_bool));

    if (has_class_token == true) { cls_token.from_ifstream(is); }
    if (num_reg_tokens > 0) { reg_token.from_ifstream(is); }
    if (use_pos_embed == true) { pos_embed.from_ifstream(is); }

    patch_embed.from_ifstream(is);
    if (use_pre_norm == true) { pre_norm.from_ifstream(is); }

    blocks.clear();
    for (int i=0;i<depth;++i) {
        // the block is initialized with garbage values, then adjusted immediatley
        blocks.emplace_back(1, 1, 1.0, false, false, 1.0, GELU);
        blocks.at(i).from_ifstream(is);
    }

    if (use_fc_norm == true) {
        fc_norm.from_ifstream(is);
    } else {
        norm.from_ifstream(is);
    }
    head.from_ifstream(is);
}

void VisionTransformer::position_embed(const Tensor& x_in, Tensor& x_out) const {
    VIT_NVTX_RANGE("pos_embed");
    Tensor y(x_in.get_B(), (x_in.get_N() + num_prefix_tokens), x_in.get_C() );
    vit_float val;

    if (has_class_token == true) {
        assert(cls_token.get_DIM() == x_in.get_C());
    }

    if (num_reg_tokens > 0) {
        assert(reg_token.get_ROWS() == num_reg_tokens);
        assert(reg_token.get_COLS() == x_in.get_C());
    }

    if (use_pos_embed == true) {
        assert(pos_embed.get_COLS() == x_in.get_C());
    }

    if (no_embed_class == true) {
        // position embed excludes prepent tokens,
        // so add pos_embed only to x values
        if (use_pos_embed == true) {
            if (dynamic_img_size == true) {
                assert(pos_embed.get_ROWS() >= x_in.get_N());
            } else {
                assert(pos_embed.get_ROWS() == x_in.get_N());
            }
        }

        for (int i=0;i<y.get_B();++i) {
            if (has_class_token == true) {
                for (int k=0;k<y.get_C();++k) {
                    val = cls_token.at(k);
                    y.set(i, 0, k, val);
                }
            }

            for (int j=0;j<reg_token.get_ROWS();++j) {
                for (int k=0;k<y.get_C();++k) {
                    val = reg_token.at(j, k);
                    if (has_class_token == true) {
                        y.set(i, j+1, k, val);
                    } else {
                        y.set(i, j, k, val);
                    }
                }
            }

            for (int j=0;j<x_in.get_N();++j) {
                for (int k=0;k<y.get_C();++k) {
                    val = x_in.at(i, j, k);
                    if (use_pos_embed == true) {
                        val += pos_embed.at(j, k);
                    }
                    y.set(i, j+num_prefix_tokens, k, val);
                }
            }
        }
    } else {
        // position embed includes prepent tokens
        // so add pos_embed to everything
        if (use_pos_embed == true) {
            if (dynamic_img_size == true) {
                assert(pos_embed.get_ROWS() >= x_in.get_N() + num_prefix_tokens);
            } else {
                assert(pos_embed.get_ROWS() == x_in.get_N() + num_prefix_tokens);
            }
        }

        for (int i=0;i<y.get_B();++i) {
            if (has_class_token == true) {
                for (int k=0;k<y.get_C();++k) {
                    val = cls_token.at(k);
                    if (use_pos_embed == true) {
                        val += pos_embed.at(0, k);
                    }
                    y.set(i, 0, k, val);
                }
            }

            for (int j=0;j<reg_token.get_ROWS();++j) {
                for (int k=0;k<y.get_C();++k) {
                    val = reg_token.at(j, k);
                    if (use_pos_embed == true) {
                        if (has_class_token == true) {
                            val += pos_embed.at(j+1, k);
                        } else {
                            val += pos_embed.at(j, k);
                        }
                    }
                    if (has_class_token == true) {
                        y.set(i, j+1, k, val);
                    } else {
                        y.set(i, j, k, val);
                    }
                }
            }

            for (int j=0;j<x_in.get_N();++j) {
                for (int k=0;k<y.get_C();++k) {
                    val = x_in.at(i, j, k);
                    if (use_pos_embed == true) {
                        val += pos_embed.at(j+num_prefix_tokens, k);
                    }
                    y.set(i, j+num_prefix_tokens, k, val);
                }
            }
        }
    }

    x_out = std::move(y);
}

void VisionTransformer::forward_features(const PictureBatch& p_in, Tensor& x_out) const {
    patch_embed.forward(p_in,x_out);

    this->position_embed(x_out,x_out);

    if (use_pre_norm == true) {
        pre_norm(x_out);
    }

    for (int i=0;i<depth;++i) {
        blocks.at(i).forward(x_out,x_out);
    }

    if (use_fc_norm == false) {
        norm(x_out);
    }
}

void VisionTransformer::pool(const Tensor& x_in, Tensor& x_out) const {
    global_pool_nlc(x_in, x_out, global_pool, num_prefix_tokens, false);
}

void VisionTransformer::forward_head(const Tensor& x_in, Tensor& x_out) const {
    this->pool(x_in, x_out);
    if (use_fc_norm == true) {
        fc_norm(x_out);
    }
    head(x_out, x_out);
}

void VisionTransformer::forward(const PictureBatch& pic, PredictionBatch& pred) const {
    Tensor x;
    this->forward_features(pic, x);
    this->forward_head(x,x);
    PredictionBatch pb(x);
    pred = std::move(pb);
}

void VisionTransformer::timed_forward(const PictureBatch& pic, PredictionBatch& pred, RowVector& times) const {
    RowVector t(4 + (2*depth));

    auto start_time = std::chrono::high_resolution_clock::now();
    Tensor x;
    {
        VIT_NVTX_RANGE("patch_embed");
        patch_embed.forward(pic, x);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    t.set(0, elapsed.count());

    start_time = std::chrono::high_resolution_clock::now();
    this->position_embed(x, x);
    if (use_pre_norm == true) {
        pre_norm(x);
    }
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;
    t.set(1, elapsed.count());

    for (int i=0;i<depth;++i) {
        std::cout << "Processing Transformer block " << i + 1 << " de " << depth << "..." << std::endl;
        vit_float attn_time, mlp_time;
        blocks.at(i).timed_forward(x, x, attn_time, mlp_time);
        t.set(2+(2*i), attn_time);
        t.set(3+(2*i), mlp_time);
    }

    start_time = std::chrono::high_resolution_clock::now();
    if (use_fc_norm == false) {
        norm(x);
    }
    this->pool(x, x);
    if (use_fc_norm == true) {
        fc_norm(x);
    }
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;
    t.set(2+(2*depth), elapsed.count()); // 2+(2*depth) is 4+2*(depth-1)

    start_time = std::chrono::high_resolution_clock::now();
    head(x, x);
    PredictionBatch pb(x);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;
    t.set(3+(2*depth), elapsed.count());

    pred = std::move(pb);
    times = std::move(t);
}
