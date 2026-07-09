#ifdef _OPENMP
#include <omp.h>

#include "../include/conv2d.h"

#include "../include/datatypes.h"
#include "../include/vit_nvtx.h"

#include <utility>
#include <assert.h>



Conv2d::Conv2d(
    vit_size _in_channels,
    vit_size _out_channels,
    vit_size _kernel_h,
    vit_size _kernel_w,
    vit_size _stride_h,
    vit_size _stride_w,
    vit_bool _use_bias
) : kernel(), bias() {
    in_channels = _in_channels;
    out_channels = _out_channels;

    kernel_h = _kernel_h;
    kernel_w = _kernel_w;
    stride_h = _stride_h;
    stride_w = _stride_w;
    use_bias = _use_bias;
}

Conv2d::Conv2d(Conv2d&& c2d) : kernel(std::move(c2d.kernel)), bias(std::move(c2d.bias)) {
    in_channels = c2d.in_channels;
    out_channels = c2d.out_channels;

    kernel_h = c2d.kernel_h;
    kernel_w = c2d.kernel_w;
    stride_h = c2d.stride_h;
    stride_w = c2d.stride_w;
    use_bias = c2d.use_bias;
}

Conv2d::~Conv2d() {}

Conv2d& Conv2d::operator= (Conv2d&& c2d) {
    in_channels = c2d.in_channels;
    out_channels = c2d.out_channels;

    kernel_h = c2d.kernel_h;
    kernel_w = c2d.kernel_w;
    stride_h = c2d.stride_h;
    stride_w = c2d.stride_w;
    use_bias = c2d.use_bias;

    kernel = std::move(c2d.kernel);
    bias = std::move(c2d.bias);

    return *this;
}

vit_size Conv2d::get_in_channels() const {
    return in_channels;
}

vit_size Conv2d::get_out_channels() const {
    return out_channels;
}

vit_size Conv2d::get_kernel_h() const {
    return kernel_h;
}

vit_size Conv2d::get_kernel_w() const {
    return kernel_w;
}

vit_size Conv2d::get_stride_h() const {
    return stride_h;
}

vit_size Conv2d::get_stride_w() const {
    return stride_w;
}

vit_bool Conv2d::get_use_bias() const {
    return use_bias;
}

void Conv2d::move_kernel(PictureBatch& _kernel) {
    kernel = std::move(_kernel);
}

void Conv2d::move_bias(RowVector& _bias) {
    bias = std::move(_bias);
}

void Conv2d::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );

    os.write( (char*) &in_channels, sizeof(vit_size));
    os.write( (char*) &out_channels, sizeof(vit_size));
    os.write( (char*) &kernel_h, sizeof(vit_size));
    os.write( (char*) &kernel_w, sizeof(vit_size));
    os.write( (char*) &stride_h, sizeof(vit_size));
    os.write( (char*) &stride_w, sizeof(vit_size));
    os.write( (char*) &use_bias, sizeof(vit_bool));

    kernel.to_ofstream(os);
    if (use_bias == true) {
        bias.to_ofstream(os);
    }
}

void Conv2d::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );

    is.read( (char*) &in_channels, sizeof(vit_size));
    is.read( (char*) &out_channels, sizeof(vit_size));
    is.read( (char*) &kernel_h, sizeof(vit_size));
    is.read( (char*) &kernel_w, sizeof(vit_size));
    is.read( (char*) &stride_h, sizeof(vit_size));
    is.read( (char*) &stride_w, sizeof(vit_size));
    is.read( (char*) &use_bias, sizeof(vit_bool));

    kernel.from_ifstream(is);
    if (use_bias == true) {
        bias.from_ifstream(is);
    }
}

void Conv2d::forward(const PictureBatch& x_in, PictureBatch& x_out) const {
    VIT_NVTX_RANGE("conv2d");
    assert(x_in.get_C() == in_channels);
    assert(kernel.get_C() == in_channels);
    assert(kernel.get_B() == out_channels);
    if (use_bias == true) {
        assert(bias.get_DIM() == out_channels);
    }

    assert( (x_in.get_H()-kernel.get_H()) % stride_h  == 0);
    vit_size out_h = ( (x_in.get_H()-kernel.get_H()) / stride_h ) + 1;

    assert( (x_in.get_W()-kernel.get_W()) % stride_w == 0);
    vit_size out_w = ( (x_in.get_W()-kernel.get_W()) / stride_w ) + 1;

    PictureBatch y(x_in.get_B(), out_channels, out_h, out_w);

    vit_float val;
    #pragma omp parallel for collapse(4) private(val) schedule(static)
    for (int batch=0;batch<y.get_B();++batch) {
        for (int y_c=0;y_c<out_channels;++y_c) {
            for (int y_h=0;y_h<out_h;++y_h) {
                for (int y_w=0;y_w<out_w;++y_w) {

                    val = use_bias==true ? bias.at(y_c) : 0;

                    for(int k_c=0;k_c<kernel.get_C();++k_c) {
                        for(int k_h=0;k_h<kernel.get_H();++k_h) {
                            for(int k_w=0;k_w<kernel.get_W();++k_w) {
                                val +=
                                    kernel.at(y_c, k_c, k_h, k_w) *
                                    x_in.at(batch, k_c, (y_h*stride_h)+k_h, (y_w*stride_w)+k_w);
                            }
                        }
                    }

                    y.set(batch, y_c, y_h, y_w, val);
                }
            }
        }
    }

    x_out = std::move(y);
}

#else

#error "Error: omp sources must be compiled with -fopenmp compiler flag!"

#endif
