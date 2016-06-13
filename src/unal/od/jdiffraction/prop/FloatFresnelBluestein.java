/*
 * Copyright 2014 Universidad Nacional de Colombia
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
package unal.od.jdiffraction.prop;

import org.jtransforms.fft.FloatFFT_2D;
import unal.od.jdiffraction.utils.FloatArrayUtils;

/**
 *
 * @author Pablo Piedrahita-Quintero (jppiedrahitaq@unal.edu.co)
 * @author Jorge Garcia-Sucerquia (jigarcia@unal.edu.co)
 */
public class FloatFresnelBluestein extends FloatPropagator {

    private final int M, N;
    private final float z, lambda, dx, dy, dxOut, dyOut;
    private final float[][] kernel1, kernel2, outputPhase;
    private final FloatFFT_2D fft;

    /**
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
        int M2, N2;
        float factor, factor2, factor3, kernelFactorX1, kernelFactorX2,
                kernelFactorY1, kernelFactorY2, outputFactorX, outputFactorY;

        M2 = M / 2;
        N2 = N / 2;
        
        factor = (float) Math.PI / (lambda * z);
        factor2 = (float) Math.PI * 2 * z / lambda;
        factor3 = lambda * z;
        
        kernelFactorX1 = dx * (dx - dxOut);
        kernelFactorY1 = dy * (dy - dyOut);
        kernelFactorX2 = dx * dxOut;
        kernelFactorY2 = dy * dyOut;

        outputFactorX = dxOut * (dx - dxOut);
        outputFactorY = dyOut * (dy - dyOut);

        for (int i = 0; i < M; i++) {
            int i2 = i - M2 + 1;
            float c1 = i2 * i2 * kernelFactorX1;
            float c2 = i2 * i2 * kernelFactorX2;
            float p1 = i2 * i2 * outputFactorX;

            for (int j = 0; j < N; j++) {
                int j2 = j - N2 + 1;
                float c3 = j2 * j2 * kernelFactorY1;
                float c4 = j2 * j2 * kernelFactorY2;
                float p2 = j2 * j2 * outputFactorY;

                float kernelPhase1, kernelPhase2, phase;

                kernelPhase1 = factor * (c1 + c3);
                kernelPhase2 = factor * (c2 + c4);

                kernel1[i][2 * j] = (float) Math.cos(kernelPhase1);
                kernel1[i][2 * j + 1] = (float) Math.sin(kernelPhase1);
                kernel2[i][2 * j] = (float) Math.cos(kernelPhase2);
                kernel2[i][2 * j + 1] = (float) Math.sin(kernelPhase2);

                phase = -factor * (p1 + p2);

                outputPhase[i][2 * j] = (float) Math.sin(factor2 + phase) / factor3;
                outputPhase[i][2 * j + 1] = (float) -Math.cos(factor2 + phase) / factor3;
            }
        }

        FloatArrayUtils.complexShift(kernel2);
        fft.complexForward(kernel2);
        FloatArrayUtils.complexShift(kernel2);
    }

    @Override
    public void diffract(float[][] field) {
        if (M != field.length || N != (field[0].length / 2)) {
            throw new IllegalArgumentException("Array dimension must be " + M + " x " + 2 * N + ".");
        }

        FloatArrayUtils.complexMultiplication2(field, kernel1);
        FloatArrayUtils.complexShift(field);
        fft.complexForward(field);
        FloatArrayUtils.complexShift(field);
        FloatArrayUtils.complexMultiplication2(field, kernel2);
        FloatArrayUtils.complexShift(field);
        fft.complexInverse(field, false);
        FloatArrayUtils.complexShift(field);
        FloatArrayUtils.complexMultiplication2(field, outputPhase);
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
