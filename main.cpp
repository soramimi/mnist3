
#include "mnist.h"
#include "rwfile.h"
#include <random>

class SoftmaxWithLossLayer {
private:
	Matrix T;
	Matrix Y;
public:
	Matrix Loss;
	void reset(Matrix const &t)
	{
		T = t;
		Y = {};
		Loss = {};
	}
	Matrix forward(Matrix const &in)
	{
		Y = in.softmax();
		Loss = Y.cross_entropy_error(T);
		return Y;
	}
	Matrix backward(Matrix const &out)
	{
		return out.sub(T).div(out.rows());
	}
};

class AffineLayer {
private:
	Matrix X;
	Matrix dW, dB;
public:
	Matrix W, B;
	AffineLayer() = default;
	AffineLayer(int input, int output, std::function<Matrix::real_t()> const &rand)
	{
		make(input, output);

		for (size_t i = 0; i < W.size(); i++) {
			W.data()[i] = rand();
		}
	}
	void reset(Matrix const &t)
	{
		X = {};
		dW = {};
		dB = {};
	}
	void make(int input, int output)
	{
		W.make(input, output);
		B.make(1, output);
	}
	Matrix forward(Matrix const &in)
	{
		X = in;
		return in.dot(W).add(B);
	}
	Matrix backward(Matrix const &out)
	{
		Matrix dx = out.dot(W.transpose());
		dW = X.transpose().dot(out);
		dB = out.sum();
		return dx;
	}
	void learn(Matrix::real_t learning_rate)
	{
		W = W.sub(dW.mul(learning_rate));
		B = B.sub(dB.mul(learning_rate));
	}
};

class SigmoidLayer {
private:
	Matrix Y;
public:
	void reset(Matrix const &t)
	{
		Y = {};
	}
	Matrix forward(Matrix const &in)
	{
		Y = in.sigmoid();
		return Y;
	}
	Matrix backward(Matrix const &out)
	{
		if (Y.rows() != out.rows()) return {};
		if (Y.cols() != out.cols()) return {};
		Matrix dx;
		dx.make(Y.rows(), Y.cols());
		size_t n = Y.size();
		for (size_t i = 0; i < n; i++) {
			dx.data()[i] = (1 - Y.data()[i]) * Y.data()[i] * out.data()[i];
		}
		return dx;
	}
};


class TwoLayerNet {
public:
	AffineLayer affine_layer1;
	SigmoidLayer sigmoid_layer;
	AffineLayer affine_layer2;
	SoftmaxWithLossLayer softmax_layer;

	TwoLayerNet()
	{
		int input = 28 * 28;
		int hidden = 50;
		int output = 10;
		affine_layer1.make(input, hidden);
		affine_layer2.make(hidden, output);
	}

	Matrix predict(Matrix const &x)
	{
		return x.dot(affine_layer1.W).add(affine_layer1.B).sigmoid()
				.dot(affine_layer2.W).add(affine_layer2.B).softmax();
	}

	Matrix::real_t accuracy(Matrix const &x, Matrix const &t)
	{
		auto argmax = [](Matrix const &a, int row){
			int i = 0;
			for (size_t j = 1; j < a.cols(); j++) {
				if (a.at(row, j) > a.at(row, i)) {
					i = j;
				}
			}
			return i;
		};

		int rows = std::min(x.rows(), t.rows());
		Matrix y = predict(x);
		int acc = 0;
		for (int row = 0; row < rows; row++) {
			auto a = argmax(y, row);
			auto b = argmax(t, row);
			if (a == b) {
				acc++;
			}
		}
		return (Matrix::real_t)acc / rows;
	}

	void gradient(Matrix const &x, Matrix const &t)
	{
		affine_layer1.reset(t);
		sigmoid_layer.reset(t);
		affine_layer2.reset(t);
		softmax_layer.reset(t);

		Matrix a1 = affine_layer1.forward(x);
		Matrix z1 = sigmoid_layer.forward(a1);
		Matrix a2 = affine_layer2.forward(z1);
		Matrix y = softmax_layer.forward(a2);

		Matrix dy = softmax_layer.backward(y);
		Matrix dz1 = affine_layer2.backward(dy);
		Matrix da1 = sigmoid_layer.backward(dz1);
		affine_layer1.backward(da1);
	}

	void train(Matrix const &x_batch, Matrix const &t_batch, double learning_rate)
	{
		gradient(x_batch, t_batch);
		affine_layer1.learn(learning_rate);
		affine_layer2.learn(learning_rate);
	}
};

int main()
{
	mnist::DataSet train;
	if (!train.load("train-labels-idx1-ubyte", "train-images-idx3-ubyte")) {
		fprintf(stderr, "failed to load mnist images and labels\n");
		exit(1);
	}

	mnist::DataSet t10k;
	if (!t10k.load("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte")) {
		fprintf(stderr, "failed to load mnist images and labels\n");
		exit(1);
	}

	int iteration = 10000;
	int batch_size = 100;
	Matrix::real_t learning_rate = 0.1;

	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());
	std::normal_distribution<Matrix::real_t> dist(0.0, 0.1);
	auto Initialize = [&](Matrix *a){
		auto Rand = [&](){
			return dist(engine);
		};
		for (size_t i = 0; i < a->size(); i++) {
			a->data()[i] = Rand();
		}
	};

	TwoLayerNet net;
	Initialize(&net.affine_layer2.W);
	Initialize(&net.affine_layer1.W);

	unsigned int k = 0;
	for (int i = 0; i < iteration; i++) {
		Matrix x_batch;
		Matrix t_batch;
		for (int j = 0; j < batch_size; j++) {
			Matrix x, t;
			k = (k + rand()) % train.size();
			train.image_to_matrix(k, &x);
			train.label_to_matrix(k, &t);
			x_batch.add_rows(x);
			t_batch.add_rows(t);
		}

		net.train(x_batch, t_batch, learning_rate);

		if ((i + 1) % 100 == 0) {
			Matrix::real_t t = net.accuracy(x_batch, t_batch);
			printf("[train %d] %f\n", i + 1, t);
		}
	}

	{
		Matrix x_batch;
		Matrix t_batch;
		for (int j = 0; j < t10k.size(); j++) {
			Matrix x, t;
			t10k.image_to_matrix(j, &x);
			t10k.label_to_matrix(j, &t);
			x_batch.add_rows(x);
			t_batch.add_rows(t);
		}
		Matrix::real_t t = net.accuracy(x_batch, t_batch);
		printf("[t10k] %f\n", t);
	}

	return 0;
}

