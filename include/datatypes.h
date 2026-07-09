#ifndef __DATATYPES_H__
#define __DATATYPES_H__

#include <fstream>



typedef unsigned vit_size;
typedef float vit_float;
typedef bool vit_bool;
typedef enum { pool_token, pool_avg, pool_avgmax, pool_max } pool_type;



class RowVector {
private:
    vit_size DIM;
    vit_float* data;
public:
    RowVector();
    RowVector(vit_size _DIM);
    RowVector(vit_float* _data, vit_size data_dim);
    RowVector(const RowVector& v) = delete;
    RowVector(RowVector&& v);
    ~RowVector();

    RowVector& operator= (const RowVector& v) = delete;
    RowVector& operator= (RowVector&& v);
    RowVector operator+ (const RowVector& v) const;
    RowVector& operator+= (const RowVector& v);

    vit_size get_DIM() const;
    vit_float at(vit_size i) const;
    const vit_float* get_data() const; // raw contiguous buffer (for CUDA uploads)

    void set(vit_size i, vit_float val);

    void print() const;

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



class Matrix {
private:
    vit_size ROWS, COLS;
    vit_float* data;
public:
    Matrix();
    Matrix(vit_size _ROWS, vit_size _COLS);
    Matrix(vit_float* _data, vit_size data_dim, vit_size _ROWS, vit_size _COLS);
    Matrix(vit_float** _data, vit_size _ROWS, vit_size _COLS);
    Matrix(const Matrix& m) = delete;
    Matrix(Matrix&& m);
    ~Matrix();

    Matrix& operator= (const Matrix& m) = delete;
    Matrix& operator= (Matrix&& m);
    Matrix operator+ (const Matrix& m) const;
    Matrix& operator+= (const Matrix& m);

    vit_size get_ROWS() const;
    vit_size get_COLS() const;
    vit_float at(vit_size i, vit_size j) const;
    const vit_float* get_data() const; // raw contiguous buffer (for CUDA uploads)

    void set(vit_size i, vit_size j, vit_float val);

    void print() const;

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



class Tensor {
private:
    vit_size B, N, C; // We will deal with three-dimensional tensors
    vit_float* data;
public:
    Tensor();
    Tensor(vit_size _B, vit_size _N, vit_size _C);
    Tensor(vit_float* _data, vit_size data_dim, vit_size _B, vit_size _N, vit_size _C);
    Tensor(vit_float*** _data, vit_size _B, vit_size _N, vit_size _C);
    Tensor(const Tensor& t) = delete;
    Tensor(Tensor&& t);
    ~Tensor();

    Tensor& operator= (const Tensor& t) = delete;
    Tensor& operator= (Tensor&& t);
    Tensor operator+ (const Tensor& t) const;
    Tensor& operator+= (const Tensor& t);

    vit_size get_B() const;
    vit_size get_N() const;
    vit_size get_C() const;
    vit_float at(vit_size b, vit_size n, vit_size c) const;
    const vit_float* get_data() const; // raw contiguous buffer (for CUDA uploads)

    void set(vit_size b, vit_size n, vit_size c, vit_float val);
    void copy_tensor(const Tensor& t);

    void print() const;

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



class PictureBatch {
private:
    vit_size B, C, H, W;
    vit_float* data;
public:
    PictureBatch();
    PictureBatch(vit_size _B, vit_size _C, vit_size _H, vit_size _W);
    PictureBatch(
        vit_float* _data, vit_size data_dim, vit_size _B, vit_size _C, vit_size _H, vit_size _W
    );
    PictureBatch(const PictureBatch& pic) = delete;
    PictureBatch(PictureBatch&& pic);
    ~PictureBatch();

    PictureBatch& operator= (const PictureBatch& pic) = delete;
    PictureBatch& operator= (PictureBatch&& pic);

    vit_size get_B() const;
    vit_size get_C() const;
    vit_size get_H() const;
    vit_size get_W() const ;
    vit_float at(vit_size b, vit_size c, vit_size h, vit_size w) const;

    void flatten_to_tensor(Tensor& t) const;
    void get_pad(PictureBatch& pic, vit_size new_h, vit_size new_w) const;

    void set(vit_size b, vit_size c, vit_size h, vit_size w, vit_float val);

    void print() const;

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



class PredictionBatch {
private:
    vit_size B;
    vit_size CLS;

    vit_size* classes;
    vit_float* prob;
    vit_float* prob_matrix;
public:
    PredictionBatch();
    PredictionBatch(const Tensor& t);
    PredictionBatch(const PredictionBatch& pred) = delete;
    PredictionBatch(PredictionBatch&& pred);
    ~PredictionBatch();

    PredictionBatch& operator= (const PredictionBatch& pred) = delete;
    PredictionBatch& operator= (PredictionBatch&& pred);

    vit_size get_B() const;
    vit_size get_CLS() const;
    vit_size get_prediction_class(vit_size i) const;
    vit_float get_prediction_class_probability(vit_size i) const;
    vit_float get_probability_of_class(vit_size i, vit_size cls) const;

    void print() const;

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



#endif // __DATATYPES_H__
