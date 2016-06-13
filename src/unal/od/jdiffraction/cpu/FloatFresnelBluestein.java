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

import org.jtransforms.fft.FloatFFT_2D;
import unal.od.jdiffraction.cpu.utils.ArrayUtils;

/**
 * Computes wave diffraction through
 * <a href="http://dx.doi.org/10.1364/AO.49.006430" target="_blank">Fresnel-Bluestein</a>
 * method
 *
 * with single precision.
 *
 * @author Pablo Piedrahita-Quintero (jppiedrahitaq@unal.edu.co)
 * @author Carlos Trujillo (catrujila@unal.edu.co)
 * @author Jorge Garcia-Sucerquia (jigarcia@unal.edu.co)
 *
 * @since JDiffraction 1.0
 */
public class FloatFresnelBluestein extends FloatPropagator {

    private final int M, N;
    private final float z, lambda, dx, dy, dxOut, dyOut;
    private final float[][] kernel1, kernel2, outputPhase;
    private final FloatFFT_2D fft;

    /**
     * Creates a new instance of FloatFresnelBluestein. Also performs kernel
     * calculations.
     *
     * @param M Number of data points on x direction.
     * @param N Number of data points on y direction.
     * @param lambda Wavelength.
     * @param z Distance.
     * @param dx Sampling pitch on x direction.
     * @param dy Sampling pitch on y direction.
     * @param dxOut X pitch on the output field.
     * @param dyOut Y pitch on the output field.
     */
    public FloatFresnelBluestein(int M, int N, float lambda, float z, float dx,
            float dy, float dxOut, float dyOut) {

        this.M = M;
        this.N = N;
        this.lambda = lambda;
        this.dx = dx;
        this.dy = dy;
        this.dxOut = dxOut;
        this.dyOut = dyOut;
        this.z = z;

        kernel1 = new float[M][2 * N];
        kernel2 = new float[M][2 * N];
        outputPhase = new float[M][2 * N];
        fft = new FloatFFT_2D(M, N);

        calculateKernels();
    }

    private void calculateKernels() {
        int M2, N2, endM, endN;
        float factor, factor2, factor3, kernelFactorX1, kernelFactorX2,
                kernelFactorY1, kernelFactorY2, outputFactorX, outputFactorY;

        M2 = M / 2;
        N2 = N / 2;
        endM = 2 * M2 - 1;
        endN = 2 * N2 - 1;

        factor = (float) Math.PI / (lambda * z);
        factor2 = (float) Math.PI * 2 * z / lambda;
        factor3 = lambda * z;

        kernelFactorX1 = dx * (dx - dxOut);
        kernelFactorY1 = dy * (dy - dyOut);
        kernelFactorX2 = dx * dxOut;
        kernelFactorY2 = dy * dyOut;

        outputFactorX = dxOut * (dx - dxOut);
        outputFactorY = dyOut * (dy - dyOut);

        for (int i = 0; i < M2; i++) {
            int i2 = i - M2 + 1;
            float c1 = i2 * i2 * kernelFactorX1;
            float c2 = i2 * i2 * kernelFactorX2;
            float p1 = i2 * i2 * outputFactorX;

            for (int j = 0; j < N2; j++) {
                int j2 = j - N2 + 1;
                float c3 = j2 * j2 * kernelFactorY1;
                float c4 = j2 * j2 * kernelFactorY2;
                float p2 = j2 * j2 * outputFactorY;

                float kernelPhase1, kernelPhase2, phase;

                kernelPhase1 = factor * (c1 + c3);
                kernelPhase2 = factor * (c2 + c4);

                kernel1[i][2 * j] = kernel1[endM - i][2 * j] = kernel1[i][2 * (endN - j)]
                        = kernel1[endM - i][2 * (endN - j)] = (float) Math.cos(kernelPhase1);
                kernel1[i][2 * j + 1] = kernel1[endM - i][2 * j + 1] = kernel1[i][2 * (endN - j) + 1]
                        = kernel1[endM - i][2 * (endN - j) + 1] = (float) Math.sin(kernelPhase1);

                kernel2[i][2 * j] = kernel2[endM - i][2 * j] = kernel2[i][2 * (endN - j)]
                        = kernel2[endM - i][2 * (endN - j)] = (float) Math.cos(kernelPhase2);
                kernel2[i][2 * j + 1] = kernel2[endM - i][2 * j + 1] = kernel2[i][2 * (endN - j) + 1]
                        = kernel2[endM - i][2 * (endN - j) + 1] = (float) Math.sin(kernelPhase2);

                phase = -factor * (p1 + p2);

                outputPhase[i][2 * j] = outputPhase[endM - i][2 * j] = outputPhase[i][2 * (endN - j)]
                        = outputPhase[endM - i][2 * (endN - j)] = (float) Math.sin(factor2 + phase) / factor3;
                outputPhase[i][2 * j + 1] = outputPhase[endM - i][2 * j + 1] = outputPhase[i][2 * (endN - j) + 1]
                        = outputPhase[endM - i][2 * (endN - j) + 1] = (float) -Math.cos(factor2 + phase) / factor3;
            }
        }

        if (M % 2 != 0) {
            int i2 = M - M2 + 1;
            float c1 = i2 * i2 * kernelFactorX1;
            float c2 = i2 * i2 * kernelFactorX2;
            float p1 = i2 * i2 * outputFactorX;

            for (int j = 0; j < N2; j++) {
                int j2 = j - N2 + 1;
                float c3 = j2 * j2 * kernelFactorY1;
                float c4 = j2 * j2 * kernelFactorY2;
                float p2 = j2 * j2 * outputFactorY;

                float kernelPhase1, kernelPhase2, phase;

                kernelPhase1 = factor * (c1 + c3);
                kernelPhase2 = factor * (c2 + c4);

                kernel1[M - 1][2 * j] = kernel1[M - 1][2 * (endN - j)] = (float) Math.cos(kernelPhase1);
                kernel1[M - 1][2 * j + 1] = kernel1[M - 1][2 * (endN - j) + 1] = (float) Math.sin(kernelPhase1);

                kernel2[M - 1][2 * j] = kernel2[M - 1][2 * (endN - j)] = (float) Math.cos(kernelPhase2);
                kernel2[M - 1][2 * j + 1] = kernel2[M - 1][2 * (endN - j) + 1] = (float) Math.sin(kernelPhase2);

                phase = -factor * (p1 + p2);

                outputPhase[M - 1][2 * j] = outputPhase[M - 1][2 * (endN - j)] = (float) Math.sin(factor2 + phase) / factor3;
                outputPhase[M - 1][2 * j + 1] = outputPhase[M - 1][2 * (endN - j) + 1] = (float) -Math.cos(factor2 + phase) / factor3;
            }
        }

        if (N % 2 != 0) {
            int j2 = N - N2 + 1;
            float c1 = j2 * j2 * kernelFactorY1;
            float c2 = j2 * j2 * kernelFactorY2;
            float p1 = j2 * j2 * outputFactorY;

            for (int i = 0; i < N2; i++) {
                int i2 = M - M2 + 1;
                float c3 = i2 * i2 * kernelFactorX1;
                float c4 = i2 * i2 * kernelFactorX2;
                float p2 = i2 * i2 * outputFactorX;

                float kernelPhase1, kernelPhase2, phase;

                kernelPhase1 = factor * (c1 + c3);
                kernelPhase2 = factor * (c2 + c4);

                kernel1[i][2 * (N - 1)] = kernel1[endM - i][2 * (N - 1)] = (float) Math.cos(kernelPhase1);
                kernel1[i][2 * (N - 1) + 1] = kernel1[endM - i][2 * (N - 1) + 1] = (float) Math.sin(kernelPhase1);

                kernel2[i][2 * (N - 1)] = kernel2[endM - i][2 * (N - 1)] = (float) Math.cos(kernelPhase2);
                kernel2[i][2 * (N - 1) + 1] = kernel2[endM - i][2 * (N - 1) + 1] = (float) Math.sin(kernelPhase2);

                phase = -factor * (p1 + p2);

                outputPhase[i][2 * (N - 1)] = outputPhase[endM - i][2 * (N - 1)] = (float) Math.sin(factor2 + phase) / factor3;
                outputPhase[i][2 * (N - 1) + 1] = outputPhase[endM - i][2 * (N - 1) + 1] = (float) -Math.cos(factor2 + phase) / factor3;
            }
        }

        if (M % 2 != 0 && N % 2 != 0) {
            int i2 = M - M2 + 1;
            int j2 = N - N2 + 1;

            float c1 = j2 * j2 * kernelFactorY1;
            float c2 = j2 * j2 * kernelFactorY2;
            float p1 = j2 * j2 * outputFactorY;

            float c3 = i2 * i2 * kernelFactorX1;
            float c4 = i2 * i2 * kernelFactorX2;
            float p2 = i2 * i2 * outputFactorX;

            float kernelPhase1, kernelPhase2, phase;

            kernelPhase1 = factor * (c1 + c3);
            kernelPhase2 = factor * (c2 + c4);

            kernel1[M - 1][2 * (N - 1)] = (float) Math.cos(kernelPhase1);
            kernel1[M - 1][2 * (N - 1) + 1] = (float) Math.sin(kernelPhase1);

            kernel2[M - 1][2 * (N - 1)] = (float) Math.cos(kernelPhase2);
            kernel2[M - 1][2 * (N - 1) + 1] = (float) Math.sin(kernelPhase2);

            phase = -factor * (p1 + p2);

            outputPhase[M - 1][2 * (N - 1)] = (float) Math.sin(factor2 + phase) / factor3;
            outputPhase[M - 1][2 * (N - 1) + 1] = (float) -Math.cos(factor2 + phase) / factor3;
        }

        fft.complexForward(kernel2);
    }

    @Override
    public void diffract(float[][] field) {
        if (M != field.length || 2 * N != field[0].length) {
            throw new IllegalArgumentException("Array dimension must be " + M + " x " + 2 * N + ".");
        }

        ArrayUtils.complexMultiplication2(field, kernel1);
        fft.complexForward(field);
        ArrayUtils.complexMultiplication2(field, kernel2);
        fft.complexInverse(field, true);
        ArrayUtils.complexShift(field);
        ArrayUtils.complexMultiplication2(field, outputPhase);
    }

    public int getM() {
        return M;
    }

    public int getN() {
        return N;
    }

    public float getZ() {
        return z;
    }

    public float getLambda() {
        return lambda;
    }

    public float getDx() {
        return dx;
    }

    public float getDy() {
        return dy;
    }

    public float getDxOut() {
        return dxOut;
    }

    public float getDyOut() {
        return dyOut;
    }

}
