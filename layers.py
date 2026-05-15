import numpy as np


class ConvolutionLayer:

    def __init__(self, num_filters, filter_size):

        self.num_filters = num_filters
        self.filter_size = filter_size

        self.filters = np.random.randn(
            num_filters,
            filter_size,
            filter_size
        ) / 9

    def iterate_regions(self, image):

        h, w = image.shape

        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):

                region = image[
                    i:(i + self.filter_size),
                    j:(j + self.filter_size)
                ]

                yield region, i, j

    def forward(self, input):

        h, w = input.shape

        output = np.zeros((
            h - self.filter_size + 1,
            w - self.filter_size + 1,
            self.num_filters
        ))

        for region, i, j in self.iterate_regions(input):

            output[i, j] = np.sum(
                region * self.filters,
                axis=(1, 2)
            )

        return output


class MaxPoolLayer:

    def iterate_regions(self, image):

        h, w, num_filters = image.shape

        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):

                region = image[
                    (i * 2):(i * 2 + 2),
                    (j * 2):(j * 2 + 2)
                ]

                yield region, i, j

    def forward(self, input):

        h, w, num_filters = input.shape

        output = np.zeros((h // 2, w // 2, num_filters))

        for region, i, j in self.iterate_regions(input):

            output[i, j] = np.amax(region, axis=(0, 1))

        return output


class Softmax:

    def __init__(self, input_len, nodes):

        self.weights = np.random.randn(
            input_len,
            nodes
        ) / input_len

        self.biases = np.zeros(nodes)

    def forward(self, input):

        input = input.flatten()

        totals = np.dot(input, self.weights) + self.biases

        exp = np.exp(totals)

        return exp / np.sum(exp, axis=0)