/*
 * Copyright 2015 Universidad Nacional de Colombia
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
 * Computes wave diffraction through Fresnel-Bluestein method
 * (<a href="http://dx.doi.org/10.1364/AO.49.006430" target="_blank">http://dx.doi.org/10.1364/AO.49.006430</a>).
 *
 * @author Pablo Piedrahita-Quintero (jppiedrahitaq@unal.edu.co)
 * @author Jorge Garcia-Sucerquia (jigarcia@unal.edu.co)
 * 
 * @since JDiffraction 1.0
 */
public class DoubleFresnelBluestein extends DoublePropagator {

    private final int M, N;
    private final double z, lambda, dx, dy, dxOut, dyOut;
    private final double[][] kernel1, kernel2, outputPhase;
    private final DoubleFFT_2D fft;

    /**
     * Creates a new instance of DoubleFresnelBluestein. Also performs kernel
     * calculations.
     * 
     * @param M Number of data points on x direction.
     * @param N Number of data points on y direction.
     * @param lambda Wavelenght
     * @param z Distance.
     * @param dx Sampling pitch on x direction.
     * @param dy Sampling pitch on y direction.
     * @param dxOut X pitch on the output field.
     * @param dyOut Y pitch on the output field.
     */
    public DoubleFresnelBluestein(int M, int N, double lambda, double z, double dx, double dy, double dxOut, double dyOut) {

        this.M = M;
        this.N = N;
        this.lambda = lambda;
        this.dx = dx;
        this.dy = dy;
        this.dxOut = dxOut;
        this.dyOut = dyOut;
        this.z = z;

        kernel1 = new double[M][2 * N];
        kernel2 = new double[M][2 * N];
        outputPhase = new double[M][2 * N];
        fft = new DoubleFFT_2D(M, N);

        calculateKernels();
    }

    private void calculateKernels() {
        int M2, N2, endM, endN;
        double factor, factor2, factor3, kernelFactorX1, kernelFactorX2,
                kernelFactorY1, kernelFactorY2, outputFactorX, outputFactorY;

        M2 = M / 2;
        N2 = N / 2;
        endM = 2 * M2 - 1;
        endN = 2 * N2 - 1;

        factor =  Math.PI / (lambda * z);
        factor2 =  Math.PI * 2 * z / lambda;
        factor3 = lambda * z;

        kernelFactorX1 = dx * (dx - dxOut);
        kernelFactorY1 = dy * (dy - dyOut);
        kernelFactorX2 = dx * dxOut;
        kernelFactorY2 = dy * dyOut;

        outputFactorX = dxOut * (dx - dxOut);
        outputFactorY = dyOut * (dy - dyOut);

        for (int i = 0; i < M2; i++) {
            int i2 = i - M2 + 1;
            double c1 = i2 * i2 * kernelFactorX1;
            double c2 = i2 * i2 * kernelFactorX2;
            double p1 = i2 * i2 * outputFactorX;

            for (int j = 0; j < N2; j++) {
                int j2 = j - N2 + 1;
                double c3 = j2 * j2 * kernelFactorY1;
                double c4 = j2 * j2 * kernelFactorY2;
                double p2 = j2 * j2 * outputFactorY;

                double kernelPhase1, kernelPhase2, phase;

                kernelPhase1 = factor * (c1 + c3);
                kernelPhase2 = factor * (c2 + c4);

                kernel1[i][2 * j] = kernel1[endM - i][2 * j] = kernel1[i][2 * (endN - j)]
                        = kernel1[endM - i][2 * (endN - j)] =  Math.cos(kernelPhase1);
                kernel1[i][2 * j + 1] = kernel1[endM - i][2 * j + 1] = kernel1[i][2 * (endN - j) + 1]
                        = kernel1[endM - i][2 * (endN - j) + 1] =  Math.sin(kernelPhase1);

                kernel2[i][2 * j] = kernel2[endM - i][2 * j] = kernel2[i][2 * (endN - j)]
                        = kernel2[endM - i][2 * (endN - j)] =  Math.cos(kernelPhase2);
                kernel2[i][2 * j + 1] = kernel2[endM - i][2 * j + 1] = kernel2[i][2 * (endN - j) + 1]
                        = kernel2[endM - i][2 * (endN - j) + 1] =  Math.sin(kernelPhase2);

                phase = -factor * (p1 + p2);

                outputPhase[i][2 * j] = outputPhase[endM - i][2 * j] = outputPhase[i][2 * (endN - j)]
                        = outputPhase[endM - i][2 * (endN - j)] =  Math.sin(factor2 + phase) / factor3;
                outputPhase[i][2 * j + 1] = outputPhase[endM - i][2 * j + 1] = outputPhase[i][2 * (endN - j) + 1]
                        = outputPhase[endM - i][2 * (endN - j) + 1] =  -Math.cos(factor2 + phase) / factor3;
            }
        }

        if (M % 2 != 0) {
            int i2 = M - M2 + 1;
            double c1 = i2 * i2 * kernelFactorX1;
            double c2 = i2 * i2 * kernelFactorX2;
            double p1 = i2 * i2 * outputFactorX;

            for (int j = 0; j < N2; j++) {
                int j2 = j - N2 + 1;
                double c3 = j2 * j2 * kernelFactorY1;
                double c4 = j2 * j2 * kernelFactorY2;
                double p2 = j2 * j2 * outputFactorY;

                double kernelPhase1, kernelPhase2, phase;

                kernelPhase1 = factor * (c1 + c3);
                kernelPhase2 = factor * (c2 + c4);

                kernel1[M - 1][2 * j] = kernel1[M - 1][2 * (endN - j)] =  Math.cos(kernelPhase1);
                kernel1[M - 1][2 * j + 1] = kernel1[M - 1][2 * (endN - j) + 1] =  Math.sin(kernelPhase1);

                kernel2[M - 1][2 * j] = kernel2[M - 1][2 * (endN - j)] =  Math.cos(kernelPhase2);
                kernel2[M - 1][2 * j + 1] = kernel2[M - 1][2 * (endN - j) + 1] =  Math.sin(kernelPhase2);

                phase = -factor * (p1 + p2);

                outputPhase[M - 1][2 * j] = outputPhase[M - 1][2 * (endN - j)] =  Math.sin(factor2 + phase) / factor3;
                outputPhase[M - 1][2 * j + 1] = outputPhase[M - 1][2 * (endN - j) + 1] =  -Math.cos(factor2 + phase) / factor3;
            }
        }

        if (N % 2 != 0) {
            int j2 = N - N2 + 1;
            double c1 = j2 * j2 * kernelFactorY1;
            double c2 = j2 * j2 * kernelFactorY2;
            double p1 = j2 * j2 * outputFactorY;

            for (int i = 0; i < N2; i++) {
                int i2 = M - M2 + 1;
                double c3 = i2 * i2 * kernelFactorX1;
                double c4 = i2 * i2 * kernelFactorX2;
                double p2 = i2 * i2 * outputFactorX;

                double kernelPhase1, kernelPhase2, phase;

                kernelPhase1 = factor * (c1 + c3);
                kernelPhase2 = factor * (c2 + c4);

                kernel1[i][2 * (N - 1)] = kernel1[endM - i][2 * (N - 1)] =  Math.cos(kernelPhase1);
                kernel1[i][2 * (N - 1) + 1] = kernel1[endM - i][2 * (N - 1) + 1] =  Math.sin(kernelPhase1);

                kernel2[i][2 * (N - 1)] = kernel2[endM - i][2 * (N - 1)] =  Math.cos(kernelPhase2);
                kernel2[i][2 * (N - 1) + 1] = kernel2[endM - i][2 * (N - 1) + 1] =  Math.sin(kernelPhase2);

                phase = -factor * (p1 + p2);

                outputPhase[i][2 * (N - 1)] = outputPhase[endM - i][2 * (N - 1)] =  Math.sin(factor2 + phase) / factor3;
                outputPhase[i][2 * (N - 1) + 1] = outputPhase[endM - i][2 * (N - 1) + 1] =  -Math.cos(factor2 + phase) / factor3;
            }
        }

        if (M % 2 != 0 && N % 2 != 0) {
            int i2 = M - M2 + 1;
            int j2 = N - N2 + 1;

            double c1 = j2 * j2 * kernelFactorY1;
            double c2 = j2 * j2 * kernelFactorY2;
            double p1 = j2 * j2 * outputFactorY;

            double c3 = i2 * i2 * kernelFactorX1;
            double c4 = i2 * i2 * kernelFactorX2;
            double p2 = i2 * i2 * outputFactorX;

            double kernelPhase1, kernelPhase2, phase;

            kernelPhase1 = factor * (c1 + c3);
            kernelPhase2 = factor * (c2 + c4);

            kernel1[M - 1][2 * (N - 1)] =  Math.cos(kernelPhase1);
            kernel1[M - 1][2 * (N - 1) + 1] =  Math.sin(kernelPhase1);

            kernel2[M - 1][2 * (N - 1)] =  Math.cos(kernelPhase2);
            kernel2[M - 1][2 * (N - 1) + 1] =  Math.sin(kernelPhase2);

            phase = -factor * (p1 + p2);

            outputPhase[M - 1][2 * (N - 1)] =  Math.sin(factor2 + phase) / factor3;
            outputPhase[M - 1][2 * (N - 1) + 1] =  -Math.cos(factor2 + phase) / factor3;
        }

        ArrayUtils.complexShift(kernel2);
        fft.complexForward(kernel2);
        ArrayUtils.complexShift(kernel2);
    }

    @Override
    public void diffract(double[][] field) {
        if (M != field.length || N != (field[0].length / 2)) {
            throw new IllegalArgumentException("Array dimension must be " + M + " x " + 2 * N + ".");
        }

        ArrayUtils.complexMultiplication2(field, kernel1);
        ArrayUtils.complexShift(field);
        fft.complexForward(field);
        ArrayUtils.complexShift(field);
        ArrayUtils.complexMultiplication2(field, kernel2);
        ArrayUtils.complexShift(field);
        fft.complexInverse(field, false);
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
