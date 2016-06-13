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
package unal.od.jdiffraction.gpu.utils;

import java.io.IOException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static unal.od.jdiffraction.gpu.utils.CUDAUtils.preparePtxFile;

/**
 * Utilities for working with complex and real data arrays. The physical layout
 * of the complex data must be the same as in JTransforms:
 * <p>
 * {@code
 * field[i * 2 * N + 2 * j] = Re[i][j],
 * field[i * 2 * N + 2 * j + 1] = Im[i][j]; 0 &lt;= i &lt; M, 0 &lt;= j &lt; N
 * }
 *
 * @author Pablo Piedrahita-Quintero (jppiedrahitaq@unal.edu.co)
 * @author Carlos Trujillo (catrujila@unal.edu.co)
 * @author Jorge Garcia-Sucerquia (jigarcia@unal.edu.co)
 *
 * @since JDiffraction 1.2
 */
public class ArrayUtilsGPU {

    private static ArrayUtilsGPU INSTANCE = null;

    private final CUDAUtils cuUtils;

    private static CUcontext context;
    private static CUdevice device;
    private static CUmodule module;

    private static CUfunction phase, phaseD;
    private static CUfunction modulus, modulusD;
    private static CUfunction modulusSq, modulusSqD;
    private static CUfunction complexMultiplication, complexMultiplicationD;
    private static CUfunction complexMultiplication2, complexMultiplication2D;
    private static CUfunction complexAmplitude, complexAmplitudeD;
    private static CUfunction complexAmplitudeAmp, complexAmplitudeAmpD;
    private static CUfunction complexAmplitudePhase, complexAmplitudePhaseD;
    private static CUfunction complexAmplitude2, complexAmplitude2D;
    private static CUfunction complexAmplitude2Real, complexAmplitude2RealD;
    private static CUfunction complexAmplitude2Img, complexAmplitude2ImgD;
    private static CUfunction real, realD;
    private static CUfunction imaginary, imaginaryD;
    private static CUfunction complexShift, complexShiftD;
    private static CUfunction realShift, realShiftD;
    private static CUfunction multiply, multiplyD;
    private static CUfunction divide, divideD;
    private static CUfunction log10, log10D;
    private static CUfunction max, maxD;
    private static CUfunction min, minD;
    private static CUfunction scale, scaleD;
    private static CUfunction scale2, scale2D;

    private static int maxThreads;
    private static int threadsPerDimension;

    private static final String VERSION = "1.2";

    /**
     * Creates a new instance of ArrayUtilsGPU.
     *
     * @throws IOException
     */
    private ArrayUtilsGPU() throws IOException {
        cuUtils = CUDAUtils.getInstance();

        cuUtils.initCUDA();

        device = cuUtils.getDevice(0);

        maxThreads = cuUtils.getMaxThreads(device);
        threadsPerDimension = (int) Math.sqrt(maxThreads);

        context = cuUtils.getContext(device);

        declareKernels();
    }

    private synchronized static void createInstance() throws IOException {
        if (INSTANCE == null) {
            INSTANCE = new ArrayUtilsGPU();
        }
    }

    /**
     * Returns the current instance of ArrayUtilsGPU. If the current instance is
     * <code>null</code>, a new instance is created.
     *
     * @return instance of ArrayUtilsGPU
     * @throws IOException
     */
    public static ArrayUtilsGPU getInstance() throws IOException {
        if (INSTANCE == null) {
            createInstance();
        }
        return INSTANCE;
    }

    private void declareKernels() throws IOException {
        String filename = preparePtxFile("Utils.cu");

        module = new CUmodule();
        cuModuleLoad(module, filename);

        phase = new CUfunction();
        cuModuleGetFunction(phase, module, "phase");

        phaseD = new CUfunction();
        cuModuleGetFunction(phaseD, module, "phaseD");

        modulus = new CUfunction();
        cuModuleGetFunction(modulus, module, "modulus");

        modulusD = new CUfunction();
        cuModuleGetFunction(modulusD, module, "modulusD");

        modulusSq = new CUfunction();
        cuModuleGetFunction(modulusSq, module, "modulusSq");

        modulusSqD = new CUfunction();
        cuModuleGetFunction(modulusSqD, module, "modulusSqD");

        complexMultiplication = new CUfunction();
        cuModuleGetFunction(complexMultiplication, module, "complexMultiplication");

        complexMultiplicationD = new CUfunction();
        cuModuleGetFunction(complexMultiplicationD, module, "complexMultiplicationD");

        complexMultiplication2 = new CUfunction();
        cuModuleGetFunction(complexMultiplication2, module, "complexMultiplication2");

        complexMultiplication2D = new CUfunction();
        cuModuleGetFunction(complexMultiplication2D, module, "complexMultiplication2D");

        complexAmplitude = new CUfunction();
        cuModuleGetFunction(complexAmplitude, module, "complexAmplitude");

        complexAmplitudeD = new CUfunction();
        cuModuleGetFunction(complexAmplitudeD, module, "complexAmplitudeD");

        complexAmplitudeAmp = new CUfunction();
        cuModuleGetFunction(complexAmplitudeAmp, module, "complexAmplitudeAmp");

        complexAmplitudeAmpD = new CUfunction();
        cuModuleGetFunction(complexAmplitudeAmpD, module, "complexAmplitudeAmpD");

        complexAmplitudePhase = new CUfunction();
        cuModuleGetFunction(complexAmplitudePhase, module, "complexAmplitudePhase");

        complexAmplitudePhaseD = new CUfunction();
        cuModuleGetFunction(complexAmplitudePhaseD, module, "complexAmplitudePhaseD");

        complexAmplitude2 = new CUfunction();
        cuModuleGetFunction(complexAmplitude2, module, "complexAmplitude2");

        complexAmplitude2D = new CUfunction();
        cuModuleGetFunction(complexAmplitude2D, module, "complexAmplitude2D");

        complexAmplitude2Real = new CUfunction();
        cuModuleGetFunction(complexAmplitude2Real, module, "complexAmplitude2Real");

        complexAmplitude2RealD = new CUfunction();
        cuModuleGetFunction(complexAmplitude2RealD, module, "complexAmplitude2RealD");

        complexAmplitude2Img = new CUfunction();
        cuModuleGetFunction(complexAmplitude2Img, module, "complexAmplitude2Img");

        complexAmplitude2ImgD = new CUfunction();
        cuModuleGetFunction(complexAmplitude2ImgD, module, "complexAmplitude2ImgD");

        real = new CUfunction();
        cuModuleGetFunction(real, module, "real");

        realD = new CUfunction();
        cuModuleGetFunction(realD, module, "realD");

        imaginary = new CUfunction();
        cuModuleGetFunction(imaginary, module, "imaginary");

        imaginaryD = new CUfunction();
        cuModuleGetFunction(imaginaryD, module, "imaginaryD");

        complexShift = new CUfunction();
        cuModuleGetFunction(complexShift, module, "complexShift");

        complexShiftD = new CUfunction();
        cuModuleGetFunction(complexShiftD, module, "complexShiftD");

        realShift = new CUfunction();
        cuModuleGetFunction(realShift, module, "realShift");

        realShiftD = new CUfunction();
        cuModuleGetFunction(realShiftD, module, "realShiftD");

        multiply = new CUfunction();
        cuModuleGetFunction(multiply, module, "multiply");

        multiplyD = new CUfunction();
        cuModuleGetFunction(multiplyD, module, "multiplyD");

        divide = new CUfunction();
        cuModuleGetFunction(divide, module, "divide");

        divideD = new CUfunction();
        cuModuleGetFunction(divideD, module, "divideD");

        log10 = new CUfunction();
        cuModuleGetFunction(log10, module, "log10F");

        log10D = new CUfunction();
        cuModuleGetFunction(log10D, module, "log10D");

        max = new CUfunction();
        cuModuleGetFunction(max, module, "maxF");

        maxD = new CUfunction();
        cuModuleGetFunction(maxD, module, "maxD");

        min = new CUfunction();
        cuModuleGetFunction(min, module, "minF");

        minD = new CUfunction();
        cuModuleGetFunction(minD, module, "minD");

        scale = new CUfunction();
        cuModuleGetFunction(scale, module, "scale");

        scaleD = new CUfunction();
        cuModuleGetFunction(scaleD, module, "scaleD");

        scale2 = new CUfunction();
        cuModuleGetFunction(scale2, module, "scale2");

        scale2D = new CUfunction();
        cuModuleGetFunction(scale2D, module, "scale2D");
    }

    private static void checkDimension(int M, int N, boolean complex, float[] a) {
        if (a.length == 0) {
            throw new IllegalArgumentException("Arrays dimension must be greater than 0.");
        }

        if (complex && a.length != (M * 2 * N)) {
            throw new IllegalArgumentException("Arrays dimension must be M * 2 * N.");
        } else if (!complex && a.length != (M * N)) {
            throw new IllegalArgumentException("Arrays dimension must be M * N.");
        }
    }

    private static void checkDimension(int M, int N, boolean complex, double[] a) {
        if (a.length == 0) {
            throw new IllegalArgumentException("Arrays dimension must be greater than 0.");
        }

        if (complex && a.length != (M * 2 * N)) {
            throw new IllegalArgumentException("Arrays dimension must be M * 2 * N.");
        } else if (!complex && a.length != (M * N)) {
            throw new IllegalArgumentException("Arrays dimension must be M * N.");
        }
    }

    /**
     * Returns the library version as a String.
     *
     * @return library version
     */
    public static String jDiffractionVersion() {
        return VERSION;
    }

    /**
     * Computes the phase (angle) of a complex array.
     * <p>
     * {@code
     * phase[i * N + j] = atan(Im[i][j] / Re[i][j])
     * }
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @return phase (angle) array
     */
    public float[] phase(int M, int N, float[] a) {
        checkDimension(M, N, true, a);

        float[] tmp = new float[M * N];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.FLOAT);

        CUdeviceptr devPhase = new CUdeviceptr();
        cuMemAlloc(devPhase, M * N * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devPhase)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(phase,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(tmp), devPhase, M * N * Sizeof.FLOAT);

        cuMemFree(devA);
        cuMemFree(devPhase);

        return tmp;
    }

    /**
     * Computes the phase (angle) of a complex array.
     * <p>
     * {@code
     * phase[i * N + j] = atan(Im[i][j] / Re[i][j])
     * }
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @return phase (angle) array
     */
    public double[] phase(int M, int N, double[] a) {
        checkDimension(M, N, true, a);

        double[] tmp = new double[M * N];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.DOUBLE);

        CUdeviceptr devPhase = new CUdeviceptr();
        cuMemAlloc(devPhase, M * N * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devPhase)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(phaseD,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(tmp), devPhase, M * N * Sizeof.DOUBLE);

        cuMemFree(devA);
        cuMemFree(devPhase);

        return tmp;
    }

//    /**
//     * Computes the phase (angle) of a complex array in the GPU memory.
//     * <p>
//     * {@code
//     * phase[i * N + j] = atan(Im[i][j] / Re[i][j])
//     * }
//     *
//     * @param M number of rows
//     * @param N number of columns
//     * @param devA pointer to complex array in the GPU memory
//     * @param isFloat boolean to indicate the type of the data
//     * @return pointer to phase (angle) array in the GPU memory
//     */
//    public CUdeviceptr phaseGPU(int M, int N, CUdeviceptr devA, boolean isFloat) {
//        CUdeviceptr devPhase = new CUdeviceptr();
//        if (isFloat) {
//            cuMemAlloc(devPhase, M * N * Sizeof.FLOAT);
//        } else {
//            cuMemAlloc(devPhase, M * N * Sizeof.DOUBLE);
//        }
//
//        Pointer parameters = Pointer.to(
//                Pointer.to(new int[]{M}),
//                Pointer.to(new int[]{N}),
//                Pointer.to(devA),
//                Pointer.to(devPhase)
//        );
//
//        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//        cuLaunchKernel(isFloat ? phase : phaseD,
//                gridX, gridY, 1,
//                threadsPerDimension, threadsPerDimension, 1,
//                0, null,
//                parameters, null
//        );
//        cuCtxSynchronize();
//
//        return devPhase;
//    }
//
    /**
     * Computes the modulus of a complex array.
     * <p>
     * {@code
     * modulus[i * N + j] = sqrt(Re[i][j]^2 + Im[i][j]^2)
     * }
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @return modulus array
     */
    public float[] modulus(int M, int N, float[] a) {
        checkDimension(M, N, true, a);

        float[] tmp = new float[M * N];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.FLOAT);

        CUdeviceptr devModulus = new CUdeviceptr();
        cuMemAlloc(devModulus, M * N * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devModulus)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(modulus,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(tmp), devModulus, M * N * Sizeof.FLOAT);

        cuMemFree(devA);
        cuMemFree(devModulus);

        return tmp;
    }

    /**
     * Computes the modulus of a complex array.
     * <p>
     * {@code
     * modulus[i][j] = sqrt(Re[i][j]^2 + Im[i][j]^2)
     * }
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @return modulus array
     */
    public double[] modulus(int M, int N, double[] a) {
        checkDimension(M, N, true, a);

        double[] tmp = new double[M * N];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.DOUBLE);

        CUdeviceptr devModulus = new CUdeviceptr();
        cuMemAlloc(devModulus, M * N * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devModulus)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(modulusD,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(tmp), devModulus, M * N * Sizeof.DOUBLE);

        cuMemFree(devA);
        cuMemFree(devModulus);

        return tmp;
    }

//    /**
//     * Computes the modulus of a complex array in the GPU memory.
//     * <p>
//     * {@code
//     * modulus[i * N + j] = sqrt(Re[i][j]^2 + Im[i][j]^2)
//     * }
//     *
//     * @param M number of rows
//     * @param N number of columns
//     * @param devA complex array
//     * @param isFloat boolean to indicate the type of the data
//     * @return modulus array
//     */
//    public CUdeviceptr modulusGPU(int M, int N, CUdeviceptr devA, boolean isFloat) {
//        CUdeviceptr devModulus = new CUdeviceptr();
//        if (isFloat) {
//            cuMemAlloc(devModulus, M * N * Sizeof.FLOAT);
//        } else {
//            cuMemAlloc(devModulus, M * N * Sizeof.DOUBLE);
//        }
//
//        Pointer parameters = Pointer.to(
//                Pointer.to(new int[]{M}),
//                Pointer.to(new int[]{N}),
//                Pointer.to(devA),
//                Pointer.to(devModulus)
//        );
//
//        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//        cuLaunchKernel(isFloat ? modulus : modulusD,
//                gridX, gridY, 1,
//                threadsPerDimension, threadsPerDimension, 1,
//                0, null,
//                parameters, null
//        );
//        cuCtxSynchronize();
//
//        return devModulus;
//    }
//
    /**
     * Computes the squared modulus of a complex array.
     * <p>
     * {@code
     * modulusSq[i * N + j] = Re[i][j]^2 + Im[i][j]^2
     * }
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @return modulus squared array
     */
    public float[] modulusSq(int M, int N, float[] a) {
        checkDimension(M, N, true, a);

        float[] tmp = new float[M * N];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.FLOAT);

        CUdeviceptr devModulusSq = new CUdeviceptr();
        cuMemAlloc(devModulusSq, M * N * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devModulusSq)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(modulusSq,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(tmp), devModulusSq, M * N * Sizeof.FLOAT);

        cuMemFree(devA);
        cuMemFree(devModulusSq);

        return tmp;
    }

    /**
     * Computes the squared modulus of a complex array.
     * <p>
     * {@code
     * modulusSq[i * N + j] = Re[i][j]^2 + Im[i][j]^2
     * }
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @return modulus squared array
     */
    public double[] modulusSq(int M, int N, double[] a) {
        checkDimension(M, N, true, a);

        double[] tmp = new double[M * N];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.DOUBLE);

        CUdeviceptr devModulusSq = new CUdeviceptr();
        cuMemAlloc(devModulusSq, M * N * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devModulusSq)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(modulusSqD,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(tmp), devModulusSq, M * N * Sizeof.DOUBLE);

        cuMemFree(devA);
        cuMemFree(devModulusSq);

        return tmp;
    }

//    /**
//     * Computes the squared modulus of a complex array.
//     * <p>
//     * {@code
//     * modulusSq[i * N + j] = Re[i][j]^2 + Im[i][j]^2
//     * }
//     *
//     * @param M number of rows
//     * @param N number of columns
//     * @param devA complex array
//     * @param isFloat
//     * @return modulus squared array
//     */
//    public CUdeviceptr modulusSqGPU(int M, int N, CUdeviceptr devA, boolean isFloat) {
//        CUdeviceptr devModulusSq = new CUdeviceptr();
//        if (isFloat) {
//            cuMemAlloc(devModulusSq, M * N * Sizeof.FLOAT);
//        } else {
//            cuMemAlloc(devModulusSq, M * N * Sizeof.DOUBLE);
//        }
//
//        Pointer parameters = Pointer.to(
//                Pointer.to(new int[]{M}),
//                Pointer.to(new int[]{N}),
//                Pointer.to(devA),
//                Pointer.to(devModulusSq)
//        );
//
//        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//        cuLaunchKernel(isFloat ? modulusSq : modulusSqD,
//                gridX, gridY, 1,
//                threadsPerDimension, threadsPerDimension, 1,
//                0, null,
//                parameters, null
//        );
//        cuCtxSynchronize();
//
//        return devModulusSq;
//    }
//
    /**
     * Computes the pointwise complex multiplication of 2 arrays.
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @param b complex array
     * @return multiplication
     */
    public float[] complexMultiplication(int M, int N, float[] a, float[] b) {
        checkDimension(M, N, true, a);
        checkDimension(M, N, true, b);

        float[] multiplied = new float[M * 2 * N];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.FLOAT);

        CUdeviceptr devB = new CUdeviceptr();
        cuMemAlloc(devB, M * 2 * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devB, Pointer.to(b), M * 2 * N * Sizeof.FLOAT);

        CUdeviceptr devMultiplied = new CUdeviceptr();
        cuMemAlloc(devMultiplied, M * 2 * N * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devB),
                Pointer.to(devMultiplied)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(complexMultiplication,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(multiplied), devMultiplied, M * 2 * N * Sizeof.FLOAT);

        cuMemFree(devA);
        cuMemFree(devB);
        cuMemFree(devMultiplied);

        return multiplied;
    }

    /**
     * Computes the pointwise complex multiplication of 2 arrays.
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @param b complex array
     * @return multiplication
     */
    public double[] complexMultiplication(int M, int N, double[] a, double[] b) {
        checkDimension(M, N, true, a);
        checkDimension(M, N, true, b);

        double[] multiplied = new double[M * 2 * N];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.DOUBLE);

        CUdeviceptr devB = new CUdeviceptr();
        cuMemAlloc(devB, M * 2 * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devB, Pointer.to(b), M * 2 * N * Sizeof.DOUBLE);

        CUdeviceptr devMultiplied = new CUdeviceptr();
        cuMemAlloc(devMultiplied, M * 2 * N * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devB),
                Pointer.to(devMultiplied)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(complexMultiplicationD,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(multiplied), devMultiplied, M * 2 * N * Sizeof.DOUBLE);

        cuMemFree(devA);
        cuMemFree(devB);
        cuMemFree(devMultiplied);

        return multiplied;
    }

//    /**
//     * Computes the pointwise complex multiplication of 2 arrays.
//     *
//     * @param M number of rows
//     * @param N number of columns
//     * @param devA complex array
//     * @param devB complex array
//     * @param isFloat
//     * @return multiplication
//     */
//    public CUdeviceptr complexMultiplicationGPU(int M, int N, CUdeviceptr devA, CUdeviceptr devB, boolean isFloat) {
//        CUdeviceptr devMultiplied = new CUdeviceptr();
//        if (isFloat) {
//            cuMemAlloc(devMultiplied, M * 2 * N * Sizeof.FLOAT);
//        } else {
//            cuMemAlloc(devMultiplied, M * 2 * N * Sizeof.DOUBLE);
//        }
//
//        Pointer parameters = Pointer.to(
//                Pointer.to(new int[]{M}),
//                Pointer.to(new int[]{N}),
//                Pointer.to(devA),
//                Pointer.to(devB),
//                Pointer.to(devMultiplied)
//        );
//
//        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//        cuLaunchKernel(isFloat ? complexMultiplication : complexMultiplicationD,
//                gridX, gridY, 1,
//                threadsPerDimension, threadsPerDimension, 1,
//                0, null,
//                parameters, null
//        );
//        cuCtxSynchronize();
//
//        return devMultiplied;
//    }
//
    /**
     * Computes the pointwise complex multiplication of 2 arrays leaving the
     * result in {@code a}.
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @param b complex array
     */
    public void complexMultiplication2(int M, int N, float[] a, float[] b) {
        checkDimension(M, N, true, a);
        checkDimension(M, N, true, b);

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.FLOAT);

        CUdeviceptr devB = new CUdeviceptr();
        cuMemAlloc(devB, M * 2 * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devB, Pointer.to(b), M * 2 * N * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devB)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(complexMultiplication2,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, M * 2 * N * Sizeof.FLOAT);

        cuMemFree(devA);
        cuMemFree(devB);
    }

    /**
     * Computes the pointwise complex multiplication of 2 arrays leaving the
     * result in {@code a}.
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @param b complex array
     */
    public void complexMultiplication2(int M, int N, double[] a, double[] b) {
        checkDimension(M, N, true, a);
        checkDimension(M, N, true, b);

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.DOUBLE);

        CUdeviceptr devB = new CUdeviceptr();
        cuMemAlloc(devB, M * 2 * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devB, Pointer.to(b), M * 2 * N * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devB)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(complexMultiplication2D,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, M * 2 * N * Sizeof.DOUBLE);

        cuMemFree(devA);
        cuMemFree(devB);
    }

//    /**
//     * Computes the pointwise complex multiplication of 2 arrays leaving the
//     * result in {@code a}.
//     *
//     * @param M number of rows
//     * @param N number of columns
//     * @param devA complex array
//     * @param devB complex array
//     * @param isFloat
//     */
//    public void complexMultiplication2GPU(int M, int N, CUdeviceptr devA, CUdeviceptr devB, boolean isFloat) {
//        Pointer parameters = Pointer.to(
//                Pointer.to(new int[]{M}),
//                Pointer.to(new int[]{N}),
//                Pointer.to(devA),
//                Pointer.to(devB)
//        );
//
//        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//        cuLaunchKernel(isFloat ? complexMultiplication2 : complexMultiplication2D,
//                gridX, gridY, 1,
//                threadsPerDimension, threadsPerDimension, 1,
//                0, null,
//                parameters, null
//        );
//        cuCtxSynchronize();
//    }
//
    /**
     * Creates a complex array pointwise. The computation is done calculating
     * {@code amp * exp(i * phase)}.
     *
     * @param M number of rows
     * @param N number of columns
     * @param phase phase array
     * @param amp amplitude array
     * @return complex array
     */
    public float[] complexAmplitude(int M, int N, float[] phase, float[] amp) {
        checkDimension(M, N, false, phase);
        checkDimension(M, N, false, amp);

        float[] complexAmp = new float[M * 2 * N];

        CUdeviceptr devPhase = new CUdeviceptr();
        cuMemAlloc(devPhase, M * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devPhase, Pointer.to(phase), M * N * Sizeof.FLOAT);

        CUdeviceptr devAmp = new CUdeviceptr();
        cuMemAlloc(devAmp, M * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devAmp, Pointer.to(amp), M * N * Sizeof.FLOAT);

        CUdeviceptr devComplexAmp = new CUdeviceptr();
        cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devPhase),
                Pointer.to(devAmp),
                Pointer.to(devComplexAmp)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(complexAmplitude,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(complexAmp), devComplexAmp, M * 2 * N * Sizeof.FLOAT);

        cuMemFree(devPhase);
        cuMemFree(devAmp);
        cuMemFree(devComplexAmp);

        return complexAmp;
    }

    /**
     * Creates a complex array pointwise. The computation is done calculating
     * {@code amp * exp(i * phase)}.
     *
     * @param M number of rows
     * @param N number of columns
     * @param phase phase array
     * @param amp amplitude array
     * @return complex array
     */
    public double[] complexAmplitude(int M, int N, double[] phase, double[] amp) {
        checkDimension(M, N, false, phase);
        checkDimension(M, N, false, amp);

        double[] complexAmp = new double[M * 2 * N];

        CUdeviceptr devPhase = new CUdeviceptr();
        cuMemAlloc(devPhase, M * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devPhase, Pointer.to(phase), M * N * Sizeof.DOUBLE);

        CUdeviceptr devAmp = new CUdeviceptr();
        cuMemAlloc(devAmp, M * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devAmp, Pointer.to(amp), M * N * Sizeof.DOUBLE);

        CUdeviceptr devComplexAmp = new CUdeviceptr();
        cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devPhase),
                Pointer.to(devAmp),
                Pointer.to(devComplexAmp)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(complexAmplitudeD,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(complexAmp), devComplexAmp, M * 2 * N * Sizeof.DOUBLE);

        cuMemFree(devPhase);
        cuMemFree(devAmp);
        cuMemFree(devComplexAmp);

        return complexAmp;
    }

//    /**
//     * Creates a complex array pointwise. The computation is done calculating
//     * {@code amp * exp(i * phase)}.
//     *
//     * @param M number of rows
//     * @param N number of columns
//     * @param devPhase phase array
//     * @param devAmp amplitude array
//     * @param isFloat
//     * @return complex array
//     */
//    public CUdeviceptr complexAmplitudeGPU(int M, int N, CUdeviceptr devPhase, CUdeviceptr devAmp, boolean isFloat) {
//        CUdeviceptr devComplexAmp = new CUdeviceptr();
//        if (isFloat) {
//            cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.FLOAT);
//        } else {
//            cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.DOUBLE);
//        }
//
//        Pointer parameters = Pointer.to(
//                Pointer.to(new int[]{M}),
//                Pointer.to(new int[]{N}),
//                Pointer.to(devPhase),
//                Pointer.to(devAmp),
//                Pointer.to(devComplexAmp)
//        );
//
//        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//        cuLaunchKernel(isFloat ? complexAmplitude : complexAmplitudeD,
//                gridX, gridY, 1,
//                threadsPerDimension, threadsPerDimension, 1,
//                0, null,
//                parameters, null
//        );
//        cuCtxSynchronize();
//
//        return devComplexAmp;
//    }
//
    /**
     * Creates a complex array element by element. The computation is done
     * calculating <code>amp * exp(i * phase)</code>
     *
     * @param M number of rows
     * @param N number of columns
     * @param phase phase array
     * @param amp amplitude
     * @return complex array
     */
    public float[] complexAmplitude(int M, int N, float[] phase, float amp) {
        checkDimension(M, N, false, phase);

        float[] complexAmp = new float[M * 2 * N];

        CUdeviceptr devPhase = new CUdeviceptr();
        cuMemAlloc(devPhase, M * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devPhase, Pointer.to(phase), M * N * Sizeof.FLOAT);

        CUdeviceptr devComplexAmp = new CUdeviceptr();
        cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devPhase),
                Pointer.to(new float[]{amp}),
                Pointer.to(devComplexAmp)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(complexAmplitudeAmp,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(complexAmp), devComplexAmp, M * 2 * N * Sizeof.FLOAT);

        cuMemFree(devPhase);
        cuMemFree(devComplexAmp);

        return complexAmp;
    }

    /**
     * Creates a complex array pointwise. The computation is done calculating
     * {@code amp * exp(i * phase)}.
     *
     * @param M number of rows
     * @param N number of columns
     * @param phase phase array
     * @param amp amplitude
     * @return complex array
     */
    public double[] complexAmplitude(int M, int N, double[] phase, double amp) {
        checkDimension(M, N, false, phase);

        double[] complexAmp = new double[M * 2 * N];

        CUdeviceptr devPhase = new CUdeviceptr();
        cuMemAlloc(devPhase, M * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devPhase, Pointer.to(phase), M * N * Sizeof.DOUBLE);

        CUdeviceptr devComplexAmp = new CUdeviceptr();
        cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devPhase),
                Pointer.to(new double[]{amp}),
                Pointer.to(devComplexAmp)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(complexAmplitudeAmpD,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(complexAmp), devComplexAmp, M * 2 * N * Sizeof.DOUBLE);

        cuMemFree(devPhase);
        cuMemFree(devComplexAmp);

        return complexAmp;
    }

//    /**
//     * Creates a complex array element by element. The computation is done
//     * calculating <code>amp * exp(i * phase)</code>
//     *
//     * @param M number of rows
//     * @param N number of columns
//     * @param devPhase phase array
//     * @param amp amplitude
//     * @param isFloat
//     * @return complex array
//     */
//    public CUdeviceptr complexAmplitudeGPU(int M, int N, CUdeviceptr devPhase, float amp, boolean isFloat) {
//        CUdeviceptr devComplexAmp = new CUdeviceptr();
//        if (isFloat) {
//            cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.FLOAT);
//        } else {
//            cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.DOUBLE);
//        }
//
//        Pointer parameters = Pointer.to(
//                Pointer.to(new int[]{M}),
//                Pointer.to(new int[]{N}),
//                Pointer.to(devPhase),
//                Pointer.to(new float[]{amp}),
//                Pointer.to(devComplexAmp)
//        );
//
//        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//        cuLaunchKernel(isFloat ? complexAmplitudeAmp : complexAmplitudeAmpD,
//                gridX, gridY, 1,
//                threadsPerDimension, threadsPerDimension, 1,
//                0, null,
//                parameters, null
//        );
//        cuCtxSynchronize();
//
//        return devComplexAmp;
//    }
//
    /**
     * Creates a complex array pointwise. The computation is done calculating
     * {@code amp * exp(i * phase)}.
     *
     * @param M number of rows
     * @param N number of columns
     * @param phase phase
     * @param amp amplitude array
     * @return complex array
     */
    public float[] complexAmplitude(int M, int N, float phase, float[] amp) {
        checkDimension(M, N, false, amp);

        float[] complexAmp = new float[M * 2 * N];

        CUdeviceptr devAmp = new CUdeviceptr();
        cuMemAlloc(devAmp, M * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devAmp, Pointer.to(amp), M * N * Sizeof.FLOAT);

        CUdeviceptr devComplexAmp = new CUdeviceptr();
        cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(new float[]{phase}),
                Pointer.to(devAmp),
                Pointer.to(devComplexAmp)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(complexAmplitudePhase,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(complexAmp), devComplexAmp, M * 2 * N * Sizeof.FLOAT);

        cuMemFree(devAmp);
        cuMemFree(devComplexAmp);

        return complexAmp;
    }

    /**
     * Creates a complex array pointwise. The computation is done calculating
     * {@code amp * exp(i * phase)}.
     *
     * @param M number of rows
     * @param N number of columns
     * @param phase phase
     * @param amp amplitude array
     * @return complex array
     */
    public double[] complexAmplitude(int M, int N, double phase, double[] amp) {
        checkDimension(M, N, false, amp);

        double[] complexAmp = new double[M * 2 * N];

        CUdeviceptr devAmp = new CUdeviceptr();
        cuMemAlloc(devAmp, M * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devAmp, Pointer.to(amp), M * N * Sizeof.DOUBLE);

        CUdeviceptr devComplexAmp = new CUdeviceptr();
        cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(new double[]{phase}),
                Pointer.to(devAmp),
                Pointer.to(devComplexAmp)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(complexAmplitudePhaseD,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(complexAmp), devComplexAmp, M * 2 * N * Sizeof.DOUBLE);

        cuMemFree(devAmp);
        cuMemFree(devComplexAmp);

        return complexAmp;
    }

//    /**
//     * Creates a complex array pointwise. The computation is done calculating
//     * {@code amp * exp(i * phase)}.
//     *
//     * @param M number of rows
//     * @param N number of columns
//     * @param phase phase
//     * @param devAmp amplitude array
//     * @param isFloat
//     * @return complex array
//     */
//    public CUdeviceptr complexAmplitudeGPU(int M, int N, float phase, CUdeviceptr devAmp, boolean isFloat) {
//        CUdeviceptr devComplexAmp = new CUdeviceptr();
//        if (isFloat) {
//            cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.FLOAT);
//        } else {
//            cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.DOUBLE);
//        }
//
//        Pointer parameters = Pointer.to(
//                Pointer.to(new int[]{M}),
//                Pointer.to(new int[]{N}),
//                Pointer.to(new float[]{phase}),
//                Pointer.to(devAmp),
//                Pointer.to(devComplexAmp)
//        );
//
//        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//        cuLaunchKernel(isFloat ? complexAmplitudePhase : complexAmplitudePhaseD,
//                gridX, gridY, 1,
//                threadsPerDimension, threadsPerDimension, 1,
//                0, null,
//                parameters, null
//        );
//        cuCtxSynchronize();
//
//        return devComplexAmp;
//    }
//
    /**
     * Takes the real and imaginary parts and returns a complex array. If one of
     * the input arrays is null, that part is filled with zeros.
     *
     * @param M number of rows
     * @param N number of columns
     * @param real real part array
     * @param imaginary imaginary part array
     * @return complex array
     */
    public float[] complexAmplitude2(int M, int N, float[] real, float[] imaginary) {
        boolean hasReal = real != null;
        boolean hasImaginary = imaginary != null;

        float[] complexAmp = new float[M * 2 * N];

        CUdeviceptr devComplexAmp = new CUdeviceptr();
        cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.FLOAT);

        if (!hasReal && !hasImaginary) {
            throw new IllegalArgumentException("Both arguments can't be null.");
        } else if (hasReal && hasImaginary) {
            checkDimension(M, N, false, real);
            checkDimension(M, N, false, imaginary);

            CUdeviceptr devReal = new CUdeviceptr();
            cuMemAlloc(devReal, M * N * Sizeof.FLOAT);
            cuMemcpyHtoD(devReal, Pointer.to(real), M * N * Sizeof.FLOAT);

            CUdeviceptr devImg = new CUdeviceptr();
            cuMemAlloc(devImg, M * N * Sizeof.FLOAT);
            cuMemcpyHtoD(devImg, Pointer.to(imaginary), M * N * Sizeof.FLOAT);

            Pointer parameters = Pointer.to(
                    Pointer.to(new int[]{M}),
                    Pointer.to(new int[]{N}),
                    Pointer.to(devReal),
                    Pointer.to(devImg),
                    Pointer.to(devComplexAmp)
            );

            int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
            int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

            cuLaunchKernel(complexAmplitude2,
                    gridX, gridY, 1,
                    threadsPerDimension, threadsPerDimension, 1,
                    0, null,
                    parameters, null
            );

            cuMemFree(devReal);
            cuMemFree(devImg);
        } else if (hasReal && !hasImaginary) {
            checkDimension(M, N, false, real);

            CUdeviceptr devReal = new CUdeviceptr();
            cuMemAlloc(devReal, M * N * Sizeof.FLOAT);
            cuMemcpyHtoD(devReal, Pointer.to(real), M * N * Sizeof.FLOAT);

            Pointer parameters = Pointer.to(
                    Pointer.to(new int[]{M}),
                    Pointer.to(new int[]{N}),
                    Pointer.to(devReal),
                    Pointer.to(devComplexAmp)
            );

            int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
            int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

            cuLaunchKernel(complexAmplitude2Real,
                    gridX, gridY, 1,
                    threadsPerDimension, threadsPerDimension, 1,
                    0, null,
                    parameters, null
            );

            cuMemFree(devReal);
        } else if (!hasReal && hasImaginary) {
            checkDimension(M, N, false, imaginary);

            CUdeviceptr devImg = new CUdeviceptr();
            cuMemAlloc(devImg, M * N * Sizeof.FLOAT);
            cuMemcpyHtoD(devImg, Pointer.to(imaginary), M * N * Sizeof.FLOAT);

            Pointer parameters = Pointer.to(
                    Pointer.to(new int[]{M}),
                    Pointer.to(new int[]{N}),
                    Pointer.to(devImg),
                    Pointer.to(devComplexAmp)
            );

            int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
            int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

            cuLaunchKernel(complexAmplitude2Img,
                    gridX, gridY, 1,
                    threadsPerDimension, threadsPerDimension, 1,
                    0, null,
                    parameters, null
            );

            cuMemFree(devImg);
        }

        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(complexAmp), devComplexAmp, M * 2 * N * Sizeof.FLOAT);

        cuMemFree(devComplexAmp);

        return complexAmp;
    }

    /**
     * Takes the real and imaginary parts and returns a complex array. If one of
     * the input arrays is null, that part is filled with zeros.
     *
     * @param M number of rows
     * @param N number of columns
     * @param real real part array
     * @param imaginary imaginary part array
     * @return complex array
     */
    public double[] complexAmplitude2(int M, int N, double[] real, double[] imaginary) {
        boolean hasReal = real != null;
        boolean hasImaginary = imaginary != null;

        double[] complexAmp = new double[M * 2 * N];

        CUdeviceptr devComplexAmp = new CUdeviceptr();
        cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.DOUBLE);

        if (!hasReal && !hasImaginary) {
            throw new IllegalArgumentException("Both arguments can't be null.");
        } else if (hasReal && hasImaginary) {
            checkDimension(M, N, false, real);
            checkDimension(M, N, false, imaginary);

            CUdeviceptr devReal = new CUdeviceptr();
            cuMemAlloc(devReal, M * N * Sizeof.DOUBLE);
            cuMemcpyHtoD(devReal, Pointer.to(real), M * N * Sizeof.DOUBLE);

            CUdeviceptr devImg = new CUdeviceptr();
            cuMemAlloc(devImg, M * N * Sizeof.DOUBLE);
            cuMemcpyHtoD(devImg, Pointer.to(imaginary), M * N * Sizeof.DOUBLE);

            Pointer parameters = Pointer.to(
                    Pointer.to(new int[]{M}),
                    Pointer.to(new int[]{N}),
                    Pointer.to(devReal),
                    Pointer.to(devImg),
                    Pointer.to(devComplexAmp)
            );

            int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
            int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

            cuLaunchKernel(complexAmplitude2D,
                    gridX, gridY, 1,
                    threadsPerDimension, threadsPerDimension, 1,
                    0, null,
                    parameters, null
            );

            cuMemFree(devReal);
            cuMemFree(devImg);
        } else if (hasReal && !hasImaginary) {
            checkDimension(M, N, false, real);

            CUdeviceptr devReal = new CUdeviceptr();
            cuMemAlloc(devReal, M * N * Sizeof.DOUBLE);
            cuMemcpyHtoD(devReal, Pointer.to(real), M * N * Sizeof.DOUBLE);

            Pointer parameters = Pointer.to(
                    Pointer.to(new int[]{M}),
                    Pointer.to(new int[]{N}),
                    Pointer.to(devReal),
                    Pointer.to(devComplexAmp)
            );

            int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
            int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

            cuLaunchKernel(complexAmplitude2RealD,
                    gridX, gridY, 1,
                    threadsPerDimension, threadsPerDimension, 1,
                    0, null,
                    parameters, null
            );

            cuMemFree(devReal);
        } else if (!hasReal && hasImaginary) {
            checkDimension(M, N, false, imaginary);

            CUdeviceptr devImg = new CUdeviceptr();
            cuMemAlloc(devImg, M * N * Sizeof.DOUBLE);
            cuMemcpyHtoD(devImg, Pointer.to(imaginary), M * N * Sizeof.DOUBLE);

            Pointer parameters = Pointer.to(
                    Pointer.to(new int[]{M}),
                    Pointer.to(new int[]{N}),
                    Pointer.to(devImg),
                    Pointer.to(devComplexAmp)
            );

            int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
            int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

            cuLaunchKernel(complexAmplitude2ImgD,
                    gridX, gridY, 1,
                    threadsPerDimension, threadsPerDimension, 1,
                    0, null,
                    parameters, null
            );

            cuMemFree(devImg);
        }

        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(complexAmp), devComplexAmp, M * 2 * N * Sizeof.DOUBLE);

        cuMemFree(devComplexAmp);

        return complexAmp;
    }

//    /**
//     * Takes the real and imaginary parts and returns a complex array. If one of
//     * the input arrays is null, that part is filled with zeros.
//     *
//     * @param M number of rows
//     * @param N number of columns
//     * @param devReal real part array
//     * @param devImg imaginary part array
//     * @param isFloat
//     * @return complex array
//     */
//    public CUdeviceptr complexAmplitude2GPU(int M, int N, CUdeviceptr devReal, CUdeviceptr devImg, boolean isFloat) {
//        boolean hasReal = devReal != null;
//        boolean hasImaginary = devImg != null;
//
//        CUdeviceptr devComplexAmp = new CUdeviceptr();
//        if (isFloat) {
//            cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.FLOAT);
//        } else {
//            cuMemAlloc(devComplexAmp, M * 2 * N * Sizeof.DOUBLE);
//        }
//
//        if (!hasReal && !hasImaginary) {
//            throw new IllegalArgumentException("Both arguments can't be null.");
//        } else if (hasReal && hasImaginary) {
//            Pointer parameters = Pointer.to(
//                    Pointer.to(new int[]{M}),
//                    Pointer.to(new int[]{N}),
//                    Pointer.to(devReal),
//                    Pointer.to(devImg),
//                    Pointer.to(devComplexAmp)
//            );
//
//            int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//            int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//            cuLaunchKernel(isFloat ? complexAmplitude2 : complexAmplitude2D,
//                    gridX, gridY, 1,
//                    threadsPerDimension, threadsPerDimension, 1,
//                    0, null,
//                    parameters, null
//            );
//
//            cuMemFree(devReal);
//            cuMemFree(devImg);
//        } else if (hasReal && !hasImaginary) {
//            Pointer parameters = Pointer.to(
//                    Pointer.to(new int[]{M}),
//                    Pointer.to(new int[]{N}),
//                    Pointer.to(devReal),
//                    Pointer.to(devComplexAmp)
//            );
//
//            int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//            int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//            cuLaunchKernel(isFloat ? complexAmplitude2Real : complexAmplitude2RealD,
//                    gridX, gridY, 1,
//                    threadsPerDimension, threadsPerDimension, 1,
//                    0, null,
//                    parameters, null
//            );
//
//            cuMemFree(devReal);
//        } else if (!hasReal && hasImaginary) {
//            Pointer parameters = Pointer.to(
//                    Pointer.to(new int[]{M}),
//                    Pointer.to(new int[]{N}),
//                    Pointer.to(devImg),
//                    Pointer.to(devComplexAmp)
//            );
//
//            int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//            int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//            cuLaunchKernel(isFloat ? complexAmplitude2Img : complexAmplitude2ImgD,
//                    gridX, gridY, 1,
//                    threadsPerDimension, threadsPerDimension, 1,
//                    0, null,
//                    parameters, null
//            );
//
//            cuMemFree(devImg);
//        }
//
//        cuCtxSynchronize();
//
//        return devComplexAmp;
//    }
//
    /**
     * Extracts the real part of a complex array.
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @return real array
     */
    public float[] real(int M, int N, float[] a) {
        checkDimension(M, N, true, a);

        float[] tmp = new float[M * N];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.FLOAT);

        CUdeviceptr devReal = new CUdeviceptr();
        cuMemAlloc(devReal, M * N * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devReal)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(real,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(tmp), devReal, M * N * Sizeof.FLOAT);

        cuMemFree(devA);
        cuMemFree(devReal);

        return tmp;
    }

    /**
     * Extracts the real part of a complex array.
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @return real array
     */
    public static double[] real(int M, int N, double[] a) {
        checkDimension(M, N, true, a);

        double[] tmp = new double[M * N];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.DOUBLE);

        CUdeviceptr devReal = new CUdeviceptr();
        cuMemAlloc(devReal, M * N * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devReal)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(realD,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(tmp), devReal, M * N * Sizeof.DOUBLE);

        cuMemFree(devA);
        cuMemFree(devReal);

        return tmp;
    }

//    /**
//     * Extracts the real part of a complex array.
//     *
//     * @param M number of rows
//     * @param N number of columns
//     * @param devA complex array
//     * @param isFloat
//     * @return real array
//     */
//    public CUdeviceptr realGPU(int M, int N, CUdeviceptr devA, boolean isFloat) {
//        CUdeviceptr devReal = new CUdeviceptr();
//        if (isFloat) {
//            cuMemAlloc(devReal, M * N * Sizeof.FLOAT);
//        } else {
//            cuMemAlloc(devReal, M * N * Sizeof.DOUBLE);
//        }
//
//        Pointer parameters = Pointer.to(
//                Pointer.to(new int[]{M}),
//                Pointer.to(new int[]{N}),
//                Pointer.to(devA),
//                Pointer.to(devReal)
//        );
//
//        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//        cuLaunchKernel(isFloat ? real : realD,
//                gridX, gridY, 1,
//                threadsPerDimension, threadsPerDimension, 1,
//                0, null,
//                parameters, null
//        );
//        cuCtxSynchronize();
//
//        return devReal;
//    }
//
    /**
     * Extracts the imaginary part of a complex array.
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @return imaginary array
     */
    public float[] imaginary(int M, int N, float[] a) {
        checkDimension(M, N, true, a);

        float[] tmp = new float[M * N];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.FLOAT);

        CUdeviceptr devImg = new CUdeviceptr();
        cuMemAlloc(devImg, M * N * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devImg)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(imaginary,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(tmp), devImg, M * N * Sizeof.FLOAT);

        cuMemFree(devA);
        cuMemFree(devImg);

        return tmp;
    }

    /**
     * Extracts the imaginary part of a complex array.
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     * @return imaginary array
     */
    public double[] imaginary(int M, int N, double[] a) {
        checkDimension(M, N, true, a);

        double[] tmp = new double[M * N];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.DOUBLE);

        CUdeviceptr devImg = new CUdeviceptr();
        cuMemAlloc(devImg, M * N * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA),
                Pointer.to(devImg)
        );

        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(imaginaryD,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(tmp), devImg, M * N * Sizeof.DOUBLE);

        cuMemFree(devA);
        cuMemFree(devImg);

        return tmp;
    }

//    /**
//     * Extracts the imaginary part of a complex array.
//     *
//     * @param M number of rows
//     * @param N number of columns
//     * @param devA complex array
//     * @param isFloat
//     * @return imaginary array
//     */
//    public CUdeviceptr imaginaryGPU(int M, int N, CUdeviceptr devA, boolean isFloat) {
//        CUdeviceptr devImg = new CUdeviceptr();
//        if (isFloat) {
//            cuMemAlloc(devImg, M * N * Sizeof.FLOAT);
//        } else {
//            cuMemAlloc(devImg, M * N * Sizeof.DOUBLE);
//        }
//
//        Pointer parameters = Pointer.to(
//                Pointer.to(new int[]{M}),
//                Pointer.to(new int[]{N}),
//                Pointer.to(devA),
//                Pointer.to(devImg)
//        );
//
//        int gridX = (M + threadsPerDimension - 1) / threadsPerDimension;
//        int gridY = (N + threadsPerDimension - 1) / threadsPerDimension;
//
//        cuLaunchKernel(isFloat ? imaginary : imaginaryD,
//                gridX, gridY, 1,
//                threadsPerDimension, threadsPerDimension, 1,
//                0, null,
//                parameters, null
//        );
//        cuCtxSynchronize();
//
//        return devImg;
//    }
//
    /**
     * Performs the circular shifting of a complex array, leaving the result in
     * {@code a}.
     * <p>
     * {@code
     * a b -&gt; d c
     * c d       b a
     * }
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     */
    public void complexShift(int M, int N, float[] a) {
        checkDimension(M, N, true, a);

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA)
        );

        int gridX = (M / 2 + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N / 2 + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(complexShift,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, M * 2 * N * Sizeof.FLOAT);

        cuMemFree(devA);
    }

    /**
     * Performs the circular shifting of a complex array, leaving the result in
     * {@code a}.
     * <p>
     * {@code
     * a b -&gt; d c
     * c d       b a
     * }
     *
     * @param M number of rows
     * @param N number of columns
     * @param a complex array
     */
    public void complexShift(int M, int N, double[] a) {
        checkDimension(M, N, true, a);

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * 2 * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA)
        );

        int gridX = (M / 2 + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N / 2 + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(complexShiftD,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, M * 2 * N * Sizeof.DOUBLE);

        cuMemFree(devA);
    }

    /**
     * Performs the circular shifting of a complex array, leaving the result in
     * {@code a}.
     * <p>
     * {@code
     * a b -&gt; d c
     * c d       b a
     * }
     *
     * @param M number of rows
     * @param N number of columns
     * @param devA complex array
     * @param isFloat
     */
    public void complexShiftGPU(int M, int N, CUdeviceptr devA, boolean isFloat) {
        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA)
        );

        int gridX = (M / 2 + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N / 2 + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(isFloat ? complexShift : complexShiftD,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();
    }

    /**
     * Performs the circular shifting of a real array, leaving the result in
     * {@code a}.
     * <p>
     * {@code
     * a b -&gt; d c
     * c d       b a
     * }
     *
     * @param M number of rows
     * @param N number of columns
     * @param a real array
     */
    public void realShift(int M, int N, float[] a) {
        checkDimension(M, N, false, a);

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * N * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA)
        );

        int gridX = (M / 2 + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N / 2 + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(realShift,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, M * N * Sizeof.FLOAT);

        cuMemFree(devA);
    }

    /**
     * Performs the circular shifting of a real array, leaving the result in
     * {@code a}.
     * <p>
     * {@code
     * a b -&gt; d c
     * c d       b a
     * }
     *
     * @param M number of rows
     * @param N number of columns
     * @param a real array
     */
    public void realShift(int M, int N, double[] a) {
        checkDimension(M, N, false, a);

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, M * N * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), M * 2 * N * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{M}),
                Pointer.to(new int[]{N}),
                Pointer.to(devA)
        );

        int gridX = (M / 2 + threadsPerDimension - 1) / threadsPerDimension;
        int gridY = (N / 2 + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(realShiftD,
                gridX, gridY, 1,
                threadsPerDimension, threadsPerDimension, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, M * N * Sizeof.DOUBLE);

        cuMemFree(devA);
    }

//    /**
//     * Performs the circular shifting of a real array, leaving the result in
//     * {@code a}.
//     * <p>
//     * {@code
//     * a b -&gt; d c
//     * c d       b a
//     * }
//     *
//     * @param M number of rows
//     * @param N number of columns
//     * @param devA real array
//     * @param isFloat
//     */
//    public void realShiftGPU(int M, int N, CUdeviceptr devA, boolean isFloat) {
//        Pointer parameters = Pointer.to(
//                Pointer.to(new int[]{M}),
//                Pointer.to(new int[]{N}),
//                Pointer.to(devA)
//        );
//
//        int gridX = (M / 2 + threadsPerDimension - 1) / threadsPerDimension;
//        int gridY = (N / 2 + threadsPerDimension - 1) / threadsPerDimension;
//
//        cuLaunchKernel(isFloat ? realShift : realShiftD,
//                gridX, gridY, 1,
//                threadsPerDimension, threadsPerDimension, 1,
//                0, null,
//                parameters, null
//        );
//        cuCtxSynchronize();
//    }
//    
    /**
     * Computes the pointwise multiplication of an array by {@code num}, leaving
     * the result in {@code a}.
     *
     * @param a array
     * @param num number
     */
    public void multiply(float[] a, float num) {
        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(new float[]{num})
        );

        int gridX = (a.length + maxThreads - 1) / maxThreads;

        cuLaunchKernel(multiply,
                gridX, 1, 1,
                maxThreads, 1, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, a.length * Sizeof.FLOAT);

        cuMemFree(devA);
    }

    /**
     * Computes the pointwise multiplication of an array by {@code num}, leaving
     * the result in {@code a}.
     *
     * @param a array
     * @param num number
     */
    public void multiply(double[] a, double num) {
        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(new double[]{num})
        );

        int gridX = (a.length + maxThreads - 1) / maxThreads;

        cuLaunchKernel(multiplyD,
                gridX, 1, 1,
                maxThreads, 1, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, a.length * Sizeof.DOUBLE);

        cuMemFree(devA);
    }

    /**
     * Computes the pointwise division of an array by {@code num}, leaving the
     * result in {@code a}.
     *
     * @param a array
     * @param num number
     */
    public void divide(float[] a, float num) {
        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(new float[]{num})
        );

        int gridX = (a.length + maxThreads - 1) / maxThreads;

        cuLaunchKernel(divide,
                gridX, 1, 1,
                maxThreads, 1, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, a.length * Sizeof.FLOAT);

        cuMemFree(devA);
    }

    /**
     * Computes the pointwise division of an array by {@code num}, leaving the
     * result in {@code a}.
     *
     * @param a array
     * @param num number
     */
    public void divide(double[] a, double num) {
        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(new double[]{num})
        );

        int gridX = (a.length + maxThreads - 1) / maxThreads;

        cuLaunchKernel(divideD,
                gridX, 1, 1,
                maxThreads, 1, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, a.length * Sizeof.DOUBLE);

        cuMemFree(devA);
    }

    /**
     * Computes log10 of a real array.
     *
     * @param a array
     * @return array containing log10(a)
     */
    public float[] log10(float[] a) {
        float[] b = new float[a.length];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.FLOAT);

        CUdeviceptr devB = new CUdeviceptr();
        cuMemAlloc(devB, a.length * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(devB)
        );

        int gridX = (a.length + maxThreads - 1) / maxThreads;

        cuLaunchKernel(log10,
                gridX, 1, 1,
                maxThreads, 1, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(b), devB, a.length * Sizeof.FLOAT);

        cuMemFree(devA);
        cuMemFree(devB);

        return b;
    }

    /**
     * Computes log10 of a real array.
     *
     * @param a array
     * @return array containing log10(a)
     */
    public double[] log10(double[] a) {
        double[] b = new double[a.length];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.DOUBLE);

        CUdeviceptr devB = new CUdeviceptr();
        cuMemAlloc(devB, a.length * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(devB)
        );

        int gridX = (a.length + maxThreads - 1) / maxThreads;

        cuLaunchKernel(log10D,
                gridX, 1, 1,
                maxThreads, 1, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(b), devB, a.length * Sizeof.DOUBLE);

        cuMemFree(devA);
        cuMemFree(devB);

        return b;
    }

    /**
     * Gets the max value of a real array.
     *
     * @param a array
     * @return max
     */
    public float max(float[] a) {
        int gridX = (a.length + maxThreads - 1) / maxThreads;
        int sharedMemSize = maxThreads * Sizeof.FLOAT;

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.FLOAT);

        CUdeviceptr devMax = new CUdeviceptr();
        cuMemAlloc(devMax, gridX * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(devMax)
        );

        cuLaunchKernel(max,
                gridX, 1, 1,
                maxThreads, 1, 1,
                sharedMemSize, null,
                parameters, null
        );
        cuCtxSynchronize();

        float[] maxArray = new float[gridX];

        cuMemcpyDtoH(Pointer.to(maxArray), devMax, gridX * Sizeof.FLOAT);

        cuMemFree(devA);
        cuMemFree(devMax);

        float maxVal = a[0];

        for (int i = 0; i < gridX; i++) {
            maxVal = Math.max(maxVal, a[i]);
        }
        return maxVal;
    }

    /**
     * Gets the max value of a real array.
     *
     * @param a array
     * @return max
     */
    public double max(double[] a) {
        int gridX = (a.length + maxThreads - 1) / maxThreads;
        int sharedMemSize = maxThreads * Sizeof.DOUBLE;

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.DOUBLE);

        CUdeviceptr devMax = new CUdeviceptr();
        cuMemAlloc(devMax, gridX * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(devMax)
        );

        cuLaunchKernel(maxD,
                gridX, 1, 1,
                maxThreads, 1, 1,
                sharedMemSize, null,
                parameters, null
        );
        cuCtxSynchronize();

        double[] maxArray = new double[gridX];

        cuMemcpyDtoH(Pointer.to(maxArray), devMax, gridX * Sizeof.DOUBLE);

        cuMemFree(devA);
        cuMemFree(devMax);

        double maxVal = a[0];

        for (int i = 0; i < gridX; i++) {
            maxVal = Math.max(maxVal, a[i]);
        }
        return maxVal;
    }

    /**
     * Gets the min value of a real array.
     *
     * @param a array
     * @return min
     */
    public float min(float[] a) {
        int gridX = (a.length + maxThreads - 1) / maxThreads;
        int sharedMemSize = maxThreads * Sizeof.FLOAT;

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.FLOAT);

        CUdeviceptr devMin = new CUdeviceptr();
        cuMemAlloc(devMin, gridX * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(devMin)
        );

        cuLaunchKernel(min,
                gridX, 1, 1,
                maxThreads, 1, 1,
                sharedMemSize, null,
                parameters, null
        );
        cuCtxSynchronize();

        float[] minArray = new float[gridX];

        cuMemcpyDtoH(Pointer.to(minArray), devMin, gridX * Sizeof.FLOAT);

        cuMemFree(devA);
        cuMemFree(devMin);

        float minVal = a[0];

        for (int i = 0; i < gridX; i++) {
            minVal = Math.min(minVal, a[i]);
        }
        return minVal;
    }

    /**
     * Gets the min value of a real array.
     *
     * @param a array
     * @return min
     */
    public double min(double[] a) {
        int gridX = (a.length + maxThreads - 1) / maxThreads;
        int sharedMemSize = maxThreads * Sizeof.DOUBLE;

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.DOUBLE);

        CUdeviceptr devMin = new CUdeviceptr();
        cuMemAlloc(devMin, gridX * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(devMin)
        );

        cuLaunchKernel(minD,
                gridX, 1, 1,
                maxThreads, 1, 1,
                sharedMemSize, null,
                parameters, null
        );
        cuCtxSynchronize();

        double[] minArray = new double[gridX];

        cuMemcpyDtoH(Pointer.to(minArray), devMin, gridX * Sizeof.DOUBLE);

        cuMemFree(devA);
        cuMemFree(devMin);

        double minVal = a[0];

        for (int i = 0; i < gridX; i++) {
            minVal = Math.min(minVal, a[i]);
        }
        return minVal;
    }

    /**
     * Scales a real array to {@code [0 - maxScale]}.
     *
     * @param a array
     * @param max array's max value
     * @param min array's min value
     * @param maxScale max value of the output array
     * @return scaled array
     */
    public float[] scale(float[] a, float max, float min, float maxScale) {
        float[] scaled = new float[a.length];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.FLOAT);

        CUdeviceptr devScaled = new CUdeviceptr();
        cuMemAlloc(devScaled, a.length * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(new float[]{max}),
                Pointer.to(new float[]{min}),
                Pointer.to(new float[]{maxScale}),
                Pointer.to(devScaled)
        );

        int gridX = (a.length + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(scale,
                gridX, 1, 1,
                maxThreads, 1, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, a.length * Sizeof.FLOAT);

        cuMemFree(devA);
        cuMemFree(devScaled);

        return scaled;
    }

    /**
     * Scales a real array to {@code [0 - maxScale]}.
     *
     * @param a array
     * @param max array's max value
     * @param min array's min value
     * @param maxScale max value of the output array
     * @return scaled array
     */
    public double[] scale(double[] a, double max, double min, double maxScale) {
        double[] scaled = new double[a.length];

        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.DOUBLE);

        CUdeviceptr devScaled = new CUdeviceptr();
        cuMemAlloc(devScaled, a.length * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(new double[]{max}),
                Pointer.to(new double[]{min}),
                Pointer.to(new double[]{maxScale}),
                Pointer.to(devScaled)
        );

        int gridX = (a.length + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(scaleD,
                gridX, 1, 1,
                maxThreads, 1, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, a.length * Sizeof.DOUBLE);

        cuMemFree(devA);
        cuMemFree(devScaled);

        return scaled;
    }

    /**
     * Scales a real array to {@code [0 - maxScale]}. Array's max and min values
     * are found using {@link #max(float[])} and {@link #min(float[])}.
     *
     * @param a array
     * @param maxScale max value of the output array
     * @return scaled array
     */
    public float[] scale(float[] a, float maxScale) {
        float max = max(a);
        float min = min(a);

        return scale(a, max, min, maxScale);
    }

    /**
     * Scales a real array to {@code [0 - maxScale]}. Array's max and min values
     * are found using {@link #max(double[])} and {@link #min(double[])}.
     *
     * @param a array
     * @param maxScale max value of the output array
     * @return scaled array
     */
    public double[] scale(double[] a, double maxScale) {
        double max = max(a);
        double min = min(a);

        return scale(a, max, min, maxScale);
    }

    /**
     * Scales a real array to {@code [0 - maxScale]} leaving the result in
     * {@code a}.
     *
     * @param a array
     * @param max array's max value
     * @param min array's min value
     * @param maxScale max value of the output array
     */
    public void scale2(float[] a, float max, float min, float maxScale) {
        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.FLOAT);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.FLOAT);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(new float[]{max}),
                Pointer.to(new float[]{min}),
                Pointer.to(new float[]{maxScale})
        );

        int gridX = (a.length + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(scale2,
                gridX, 1, 1,
                maxThreads, 1, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, a.length * Sizeof.FLOAT);

        cuMemFree(devA);
    }

    /**
     * Scales a real array to {@code [0 - maxScale]} leaving the result in
     * {@code a}.
     *
     * @param a array
     * @param max array's max value
     * @param min array's min value
     * @param maxScale max value of the output array
     */
    public void scale2(double[] a, double max, double min, double maxScale) {
        CUdeviceptr devA = new CUdeviceptr();
        cuMemAlloc(devA, a.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(devA, Pointer.to(a), a.length * Sizeof.DOUBLE);

        Pointer parameters = Pointer.to(
                Pointer.to(new int[]{a.length}),
                Pointer.to(devA),
                Pointer.to(new double[]{max}),
                Pointer.to(new double[]{min}),
                Pointer.to(new double[]{maxScale})
        );

        int gridX = (a.length + threadsPerDimension - 1) / threadsPerDimension;

        cuLaunchKernel(scale2D,
                gridX, 1, 1,
                maxThreads, 1, 1,
                0, null,
                parameters, null
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(a), devA, a.length * Sizeof.DOUBLE);

        cuMemFree(devA);
    }

    /**
     * Scales a real array to {@code [0 - maxScale]} leaving the result in
     * {@code a}. Array's max and min values are found using
     * {@link #max(float[])} and {@link #min(float[])}.
     *
     * @param a array
     * @param maxScale max value of the output array
     */
    public void scale2(float[] a, float maxScale) {
        float max = max(a);
        float min = min(a);

        scale2(a, max, min, maxScale);
    }

    /**
     * Scales a real array to {@code [0 - maxScale]} leaving the result in
     * {@code a}. Array's max and min values are found using
     * {@link #max(double[])} and {@link #min(double[])}.
     *
     * @param a array
     * @param maxScale max value of the output array
     */
    public void scale2(double[] a, double maxScale) {
        double max = max(a);
        double min = min(a);

        scale2(a, max, min, maxScale);
    }

    /**
     * Converts a 1D array into a 2D array leaving the result in {@code b}. It
     * is assumed that the information on the 1D array is distributed as the
     * rows of the 2D array in sequence.
     *
     * @param M number of data points in x direction
     * @param N number of data points in y direction
     * @param a 1D array
     * @param b 2D array
     */
    public static void vectorToMatrixArray(int M, int N, float[] a, float[][] b) {
        if (a.length == 0 || b.length == 0 || b[0].length == 0 || a.length != (M * N) || b.length != M || b[0].length != N) {
            throw new IllegalArgumentException("The number of data points in both arrays must be equal and different from zero.");
        }

        for (int i = 0; i < M; i++) {
            System.arraycopy(a, N * i, b[i], 0, N);
        }
    }

    /**
     * Converts a 1D array into a 2D array leaving the result in {@code b}. It
     * is assumed that the information on the 1D array is distributed as the
     * rows of the 2D array in sequence.
     *
     * @param M number of data points in x direction
     * @param N number of data points in y direction
     * @param a 1D array
     * @param b 2D array
     */
    public static void vectorToMatrixArray(int M, int N, double[] a, double[][] b) {
        if (a.length == 0 || b.length == 0 || b[0].length == 0 || a.length != (M * N) || b.length != M || b[0].length != N) {
            throw new IllegalArgumentException("The number of data points in both arrays must be equal and different from zero.");
        }

        for (int i = 0; i < M; i++) {
            System.arraycopy(a, N * i, b[i], 0, N);
        }
    }

    /**
     * Converts a 2D array into a 1D array leaving the result in {@code b}. The
     * information on the 1D array is distributed as the rows of the 2D array in
     * sequence.
     *
     * @param M number of data points in x direction
     * @param N number of data points in y direction
     * @param a 2D array
     * @param b 1D array
     */
    public static void matrixToVectorArray(int M, int N, float[][] a, float[] b) {
        if (b.length == 0 || a.length == 0 || a[0].length == 0 || b.length != (M * N) || a.length != M || a[0].length != N) {
            throw new IllegalArgumentException("The number of data points in both arrays must be equal and different from zero.");
        }

        for (int i = 0; i < M; i++) {
            System.arraycopy(a[i], 0, b, N * i, N);
        }
    }

    /**
     * Converts a 2D array into a 1D array leaving the result in {@code b}. The
     * information on the 1D array is distributed as the rows of the 2D array in
     * sequence.
     *
     * @param M number of data points in x direction
     * @param N number of data points in y direction
     * @param a 2D array
     * @param b 1D array
     */
    public static void matrixToVectorArray(int M, int N, double[][] a, double[] b) {
        if (b.length == 0 || a.length == 0 || a[0].length == 0 || b.length != (M * N) || a.length != M || a[0].length != N) {
            throw new IllegalArgumentException("The number of data points in both arrays must be equal and different from zero.");
        }

        for (int i = 0; i < M; i++) {
            System.arraycopy(a[i], 0, b, N * i, N);
        }
    }
}
