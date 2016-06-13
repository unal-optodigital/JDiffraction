/*
 * Copyright 2016 Universidad Nacional de Colombia
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package unal.od.jdiffraction.cpu;

import org.jtransforms.fft.DoubleFFT_2D;
import unal.od.jdiffraction.cpu.utils.ArrayUtils;

/**
 * Computes wave diffraction through Fresnel-Fourier method with double
 * precision.
 *
 * @author Pablo Piedrahita-Quintero (jppiedrahitaq@unal.edu.co)
 * @author Carlos Trujillo (catrujila@unal.edu.co)
 * @author Jorge Garcia-Sucerquia (jigarcia@unal.edu.co)
 *
 * @since JDiffraction 1.0
 */
public class DoubleFresnelFourier extends DoublePropagator {

    private final int M, N;
    private final double z, lambda, dx, dy, dxOut, dyOut;
    private final double[][] kernel, outputPhase;
    private final DoubleFFT_2D fft;

    /**
     * Creates a new instance of DoubleFresnelFourier. Also performs kernel
     * calculations.
     *
     * @param M Number of data points on x direction.
     * @param N Number of data points on y direction.
     * @param lambda Wavelength.
     * @param z Distance.
     * @param dx Sampling pitch on x direction.
     * @param dy Sampling pitch on y direction.
     */
    public DoubleFresnelFourier(int M, int N, double lambda, double z, double dx, double dy) {
        this.M = M;
        this.N = N;
        this.lambda = lambda;
        this.dx = dx;
        this.dy = dy;
        this.z = z;

        dxOut = lambda * z / (M * dx);
        dyOut = lambda * z / (N * dy);

        kernel = new double[M][2 * N];
        outputPhase = new double[M][2 * N];
        fft = new DoubleFFT_2D(M, N);

        calculateKernels();
    }

    private void calculateKernels() {

        int M2, N2, endM, endN;
        double factor, factor2, factor3, dxSq, dySq, dxOutSq, dyOutSq;

        M2 = M / 2;
        N2 = N / 2;
        endM = 2 * M2 - 1;
        endN = 2 * N2 - 1;

        dxOutSq = dxOut * dxOut;
        dyOutSq = dyOut * dyOut;

        dxSq = dx * dx;
        dySq = dy * dy;
        factor = Math.PI / (lambda * z);
        factor2 = Math.PI * 2 * z / lambda;
        factor3 = dx * dy / (lambda * z);

        for (int i = 0; i < M2; i++) {
            int i2 = i - M2 + 1;
            double p1 = i2 * i2 * dxSq;
            double p2 = i2 * i2 * dxOutSq;

            for (int j = 0; j < N2; j++) {
                int j2 = j - N2 + 1;
                double phase;

                phase = p1 + j2 * j2 * dySq;
                phase *= factor;
                kernel[i][2 * j] = kernel[endM - i][2 * j] = kernel[i][2 * (endN - j)]
                        = kernel[endM - i][2 * (endN - j)] = Math.cos(phase);

                kernel[i][2 * j + 1] = kernel[endM - i][2 * j + 1] = kernel[i][2 * (endN - j) + 1]
                        = kernel[endM - i][2 * (endN - j) + 1] = Math.sin(phase);

                phase = p2 + j2 * j2 * dyOutSq;
                phase *= factor;
                outputPhase[i][2 * j] = outputPhase[endM - i][2 * j] = outputPhase[i][2 * (endN - j)]
                        = outputPhase[endM - i][2 * (endN - j)] = Math.sin(factor2 + phase) * factor3;

                outputPhase[i][2 * j + 1] = outputPhase[endM - i][2 * j + 1] = outputPhase[i][2 * (endN - j) + 1]
                        = outputPhase[endM - i][2 * (endN - j) + 1] = -Math.cos(factor2 + phase) * factor3;
            }
        }

        if (M % 2 != 0) {
            int i2 = M - M2 + 1;
            double p1 = i2 * i2 * dxSq;
            double p2 = i2 * i2 * dxOutSq;

            for (int j = 0; j < N2; j++) {
                int j2 = j - N2 + 1;
                double phase;

                phase = p1 + j2 * j2 * dySq;
                phase *= factor;
                kernel[M - 1][2 * j] = kernel[M - 1][2 * (endN - j)] = Math.cos(phase);
                kernel[M - 1][2 * j + 1] = kernel[M - 1][2 * (endN - j) + 1] = Math.sin(phase);

                phase = p2 + j2 * j2 * dyOutSq;
                phase *= factor;
                outputPhase[M - 1][2 * j] = outputPhase[M - 1][2 * (endN - j)] = Math.sin(factor2 + phase) * factor3;
                outputPhase[M - 1][2 * j + 1] = outputPhase[M - 1][2 * (endN - j) + 1] = -Math.cos(factor2 + phase) * factor3;
            }
        }

        if (N % 2 != 0) {
            int j2 = N - N2 + 1;
            double p1 = j2 * j2 * dySq;
            double p2 = j2 * j2 * dyOutSq;

            for (int i = 0; i < N2; i++) {
                int i2 = M - M2 + 1;
                double phase;

                phase = p1 + j2 * j2 * dySq;
                phase *= factor;
                kernel[i][2 * (N - 1)] = kernel[endM - i][2 * (N - 1)] = Math.cos(phase);
                kernel[i][2 * (N - 1) + 1] = kernel[endM - i][2 * (N - 1) + 1] = Math.sin(phase);

                phase = p2 + j2 * j2 * dyOutSq;
                phase *= factor;
                outputPhase[i][2 * (N - 1)] = outputPhase[endM - i][2 * (N - 1)] = Math.sin(factor2 + phase) * factor3;
                outputPhase[i][2 * (N - 1) + 1] = outputPhase[endM - i][2 * (N - 1) + 1] = -Math.cos(factor2 + phase) * factor3;
            }
        }

        if (M % 2 != 0 && N % 2 != 0) {
            int i2 = M - M2 + 1;
            int j2 = N - N2 + 1;

            double phase;

            phase = i2 * i2 * dxSq + j2 * j2 * dySq;
            phase *= factor;
            kernel[M - 1][2 * (N - 1)] = Math.cos(phase);
            kernel[M - 1][2 * (N - 1) + 1] = Math.sin(phase);

            phase = i2 * i2 * dxOutSq + j2 * j2 * dyOutSq;
            phase *= factor;
            outputPhase[M - 1][2 * (N - 1)] = Math.sin(factor2 + phase) * factor3;
            outputPhase[M - 1][2 * (N - 1) + 1] = -Math.cos(factor2 + phase) * factor3;
        }
    }

    @Override
    public void diffract(double[][] field) {
        if (M != field.length || 2 * N != field[0].length) {
            throw new IllegalArgumentException("Array dimension must be " + M + " x " + 2 * N + ".");
        }

        ArrayUtils.complexMultiplication2(field, kernel);
        ArrayUtils.complexShift(field);
        fft.complexForward(field);
        ArrayUtils.complexShift(field);
        ArrayUtils.complexMultiplication2(field, outputPhase);
    }

    public int getM() {
        return M;
    }

    public int getN() {
        return N;
    }

    public double getZ() {
        return z;
    }

    public double getLambda() {
        return lambda;
    }

    public double getDx() {
        return dx;
    }

    public double getDy() {
        return dy;
    }

    public double getDxOut() {
        return dxOut;
    }

    public double getDyOut() {
        return dyOut;
    }

}
